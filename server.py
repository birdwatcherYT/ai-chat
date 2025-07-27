# server.py

import asyncio
import base64
import json
import logging
import traceback
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

from src.app_context import AppContext, load_config
from src.llm.common import history_to_text
from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__, level=logging.INFO)

app = FastAPI()

ctx: AppContext | None = None
history: list[dict[str, str]] | None = None
session_images: list[str] = [] # ★ 追加: セッションごとの画像履歴

llm_text_queue = asyncio.Queue()
audio_data_queue = asyncio.Queue()


def load_config_and_init():
    global ctx, history
    try:
        config = load_config()
        ctx = AppContext(config)
        history = list(ctx.initial_history)
        app.mount(ctx.cfg.chat.image.url_path, StaticFiles(directory=ctx.cfg.chat.image.save_dir), name="images")
        app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
        logger.info(f"✅ [SYSTEM] 初期化完了 (ASR: {ctx.cfg.chat.user.input})")
    except Exception as e:
        logger.error(f"❌ [SYSTEM] 初期化エラー: {e}")
        traceback.print_exc()

class ConnectionManager:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def send_json(self, data: dict):
        if self.active_connections:
            try:
                await self.active_connections[0].send_json(data)
            except Exception:
                pass

manager = ConnectionManager()

def llm_stream_blocking_task(turn: str, current_history: list, loop: asyncio.AbstractEventLoop, image_data_list: list[str] | None = None):
    full_response, answer_segment = "", ""
    try:
        utter_chain = ctx.llms.get_utter_chain(image_data_list=image_data_list)
        utter_prompt_vars = {"speaker": turn, "messages": history_to_text(current_history)}
        for chunk in utter_chain.stream(utter_prompt_vars):
            content = getattr(chunk, 'content', '')
            full_response += content
            answer_segment += content
            asyncio.run_coroutine_threadsafe(manager.send_json({"type": "chunk", "data": {"name": turn, "content": content}}), loop)
            if answer_segment and answer_segment[-1] in ctx.cfg.chat.streaming_voice_output:
                asyncio.run_coroutine_threadsafe(llm_text_queue.put((turn, answer_segment)), loop)
                answer_segment = ""
        if answer_segment:
            asyncio.run_coroutine_threadsafe(llm_text_queue.put((turn, answer_segment)), loop)
        return {"name": turn, "content": full_response}
    except Exception:
        traceback.print_exc()
        return None
    finally:
        asyncio.run_coroutine_threadsafe(llm_text_queue.put(None), loop)

async def synthesis_consumer():
    while True:
        task = await llm_text_queue.get()
        if task is None:
            await audio_data_queue.put(None)
            break
        speaker_name, text = task
        voice_config = ctx.ai_config.get(speaker_name)
        if not voice_config or not voice_config.engine:
            continue
        try:
            data, sr = await ctx.tts_engines[voice_config.engine].synthesize_async(text, **vars(voice_config.config))
            if data is not None and sr is not None:
                await audio_data_queue.put((data, sr, str(data.dtype)))
        except Exception as e:
            logger.error(f"❌ [SYNTH] 音声合成エラー: {e}")

async def audio_sender_consumer():
    while True:
        task = await audio_data_queue.get()
        if task is None:
            break
        data, sr, dtype = task
        try:
            encoded_audio = base64.b64encode(data.tobytes()).decode("utf-8")
            await manager.send_json({"type": "audio", "data": {"audio": encoded_audio, "samplerate": sr, "dtype": dtype}})
        except Exception:
            pass

async def image_generation_task(current_history):
    task_id = "image_gen"
    await manager.send_json({"type": "status_update", "data": {"id": task_id, "text": "🎨 画像を生成しています..."}})
    try:
        image_url, _ = await asyncio.to_thread(ctx.img_generator.generate_image, current_history, ctx.cfg.chat.image.edit)
        if image_url:
            await manager.send_json({"type": "image", "url": image_url})
    except Exception as e:
        logger.error(f"❌ [IMAGE] 画像生成中にエラーが発生しました: {e}")
        await manager.send_json({"type": "status_update", "data": {"id": f"{task_id}_error", "text": "🎨 画像の生成に失敗しました。"}})
        await asyncio.sleep(3)
        await manager.send_json({"type": "status_remove", "data": {"id": f"{task_id}_error"}})
    finally:
        await manager.send_json({"type": "status_remove", "data": {"id": task_id}})

async def main_pipeline_task(turn: str, loop: asyncio.AbstractEventLoop, image_data_list: list[str] | None = None):
    llm_task = asyncio.to_thread(llm_stream_blocking_task, turn, list(history), loop, image_data_list)
    audio_task = asyncio.gather(synthesis_consumer(), audio_sender_consumer())
    results = await asyncio.gather(llm_task, audio_task)
    return results[0]

async def run_single_turn(turn: str, image_data_list: list[str] | None = None):
    global history
    loop = asyncio.get_running_loop()
    history_len_before_turn = len(history)
    await manager.send_json({"type": "next_speaker", "data": turn})
    main_response = await main_pipeline_task(turn, loop, image_data_list)
    await manager.send_json({"type": "utterance_end", "data": turn})
    if main_response:
        history.append(main_response)
        history_len_after_turn = len(history)
        interval = ctx.cfg.chat.image.interval
        if (history_len_before_turn // interval) < (history_len_after_turn // interval):
            asyncio.create_task(image_generation_task(list(history)))

async def run_ai_conversation_flow(initial_turn: str):
    turn = initial_turn
    while turn != ctx.llmcfg.user_name:
        # ★ 修正: 常にセッション全体の画像リストを渡す
        await run_single_turn(turn, image_data_list=session_images)
        turn = await asyncio.to_thread(ctx.llms.get_next_speaker, list(history), except_names=[turn])
    await manager.send_json({"type": "conversation_end"})

async def process_user_audio(audio_bytes: bytes) -> str:
    if not ctx.asr_engine:
        logger.warning("⚠️ [SYSTEM] 音声データ受信、しかしASRエンジンが無効です。")
        return ""
    try:
        audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
        audio_segment = audio_segment.set_frame_rate(16000).set_sample_width(2).set_channels(1)
        pcm_audio_bytes = audio_segment.raw_data
        logger.info(f"🎤 [DECODE] 音声デコード成功: {len(pcm_audio_bytes)} bytes")
        user_message = await asyncio.to_thread(ctx.asr_engine.process_audio, pcm_audio_bytes)
        return user_message
    except Exception as e:
        logger.error(f"❌ [ASR] 音声処理中にエラー: {e}")
        traceback.print_exc()
        return ""

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global history, session_images
    # ★ 修正: 新しい接続のためにセッションデータをリセット
    history = list(ctx.initial_history)
    session_images = []
    
    if ctx is None:
        logger.error("❌ [SYSTEM] アプリケーションが初期化されていません。")
        await websocket.close(code=1011, reason="Server not initialized")
        return

    await manager.send_json({"type": "config", "data": {"user_input_mode": ctx.cfg.chat.user.input, "user_name": ctx.llmcfg.user_name}})
    for message in history:
        await manager.send_json({"type": "history", "data": message})

    try:
        if ctx.cfg.chat.user.input == "ai":
            logger.info("🤖 [SYSTEM] 全自動AIモードで起動します。")
            turn = ctx.initial_turn
            while True:
                await run_single_turn(turn, image_data_list=session_images)
                turn = await asyncio.to_thread(ctx.llms.get_next_speaker, list(history), except_names=[])
                await asyncio.sleep(1)
        else:
            initial_turn = ctx.initial_turn
            if initial_turn != ctx.llmcfg.user_name:
                await run_ai_conversation_flow(initial_turn)
            else:
                await manager.send_json({"type": "conversation_end"})
            
            while True:
                raw_message = await websocket.receive()
                user_message_text = ""
                user_message_image = None
                
                if "text" in raw_message:
                    try:
                        data = json.loads(raw_message["text"])
                        user_message_text = data.get("text", "")
                        user_message_image = data.get("image")
                    except (json.JSONDecodeError, TypeError):
                        user_message_text = raw_message["text"]
                elif "bytes" in raw_message:
                    user_message_text = await process_user_audio(raw_message["bytes"])
                    await manager.send_json({"type": "user_transcription", "data": user_message_text})

                if not user_message_text and not user_message_image:
                    continue

                # 画像がある場合の処理
                if user_message_image:
                    session_images.append(user_message_image)
                    user_message_text += "\n(画像添付)"

                user_name = ctx.llmcfg.user_name
                history.append({"name": user_name, "content": user_message_text})

                next_turn = await asyncio.to_thread(ctx.llms.get_next_speaker, list(history), except_names=[user_name])
                asyncio.create_task(run_ai_conversation_flow(next_turn))

    except WebSocketDisconnect:
        logger.info("👋 [SYSTEM] クライアントが切断しました。")
    except Exception as e:
        logger.error(f"❌ [SYSTEM] WebSocketループエラー: {e}")
        traceback.print_exc()
    finally:
        manager.disconnect(websocket)

@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")