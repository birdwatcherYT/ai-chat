# server.py

import asyncio
import base64
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

GENERATED_IMAGES_DIR = "generated_images"
GENERATED_IMAGES_URL_PATH = "/images"
load_dotenv()
logger = get_logger(__name__, level="INFO")

app = FastAPI()

ctx: AppContext | None = None
# historyはセッションごとにリセットできるよう、AppContextからコピーして使う
history: list[dict[str, str]] | None = None

llm_text_queue = asyncio.Queue()
audio_data_queue = asyncio.Queue()


# AppContextを使った初期化
def load_config_and_init():
    """アプリケーションコンテキストを初期化する"""
    global ctx, history
    try:
        config = load_config()
        ctx = AppContext(config)
        history = list(ctx.initial_history)  # 初期履歴をコピー
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


def llm_stream_blocking_task(
    turn: str, current_history: list, loop: asyncio.AbstractEventLoop
):
    full_response, answer_segment = "", ""
    try:
        utter_chain = ctx.llms.get_utter_chain()
        utter_prompt_vars = {
            "speaker": turn,
            "messages": history_to_text(current_history),
        }
        for chunk in utter_chain.stream(utter_prompt_vars):
            full_response += chunk.content
            answer_segment += chunk.content
            asyncio.run_coroutine_threadsafe(
                manager.send_json(
                    {"type": "chunk", "data": {"name": turn, "content": chunk.content}}
                ),
                loop,
            )
            if (
                answer_segment
                and answer_segment[-1] in ctx.cfg.chat.streaming_voice_output
            ):
                asyncio.run_coroutine_threadsafe(
                    llm_text_queue.put((turn, answer_segment)), loop
                )
                answer_segment = ""
        if answer_segment:
            asyncio.run_coroutine_threadsafe(
                llm_text_queue.put((turn, answer_segment)), loop
            )
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
        if not voice_config:
            continue
        try:
            data, sr = await ctx.tts_engines[voice_config.engine].synthesize_async(
                text, **vars(voice_config.config)
            )
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
            await manager.send_json(
                {
                    "type": "audio",
                    "data": {"audio": encoded_audio, "samplerate": sr, "dtype": dtype},
                }
            )
        except Exception:
            pass


async def image_generation_task(current_history):
    task_id = "image_gen"
    await manager.send_json(
        {
            "type": "status_update",
            "data": {"id": task_id, "text": "🎨 画像を生成しています..."},
        }
    )
    try:
        image_url, _ = await asyncio.to_thread(
            ctx.img_generator.generate_image, current_history, ctx.cfg.chat.image.edit
        )
        if image_url:
            await manager.send_json({"type": "image", "url": image_url})
    except Exception as e:
        logger.error(f"❌ [IMAGE] 画像生成中にエラーが発生しました: {e}")
        await manager.send_json(
            {
                "type": "status_update",
                "data": {
                    "id": f"{task_id}_error",
                    "text": "🎨 画像の生成に失敗しました。",
                },
            }
        )
        await asyncio.sleep(3)
        await manager.send_json(
            {"type": "status_remove", "data": {"id": f"{task_id}_error"}}
        )
    finally:
        await manager.send_json({"type": "status_remove", "data": {"id": task_id}})


async def main_pipeline_task(turn: str, loop: asyncio.AbstractEventLoop):
    llm_task = asyncio.to_thread(llm_stream_blocking_task, turn, list(history), loop)
    audio_task = asyncio.gather(synthesis_consumer(), audio_sender_consumer())
    results = await asyncio.gather(llm_task, audio_task)
    return results[0]


async def run_ai_turn(turn: str, history_len_before_user_turn: int):
    global history
    loop = asyncio.get_running_loop()

    history_len_after_user_turn = len(history)
    interval = ctx.cfg.chat.image.interval
    should_generate_image = (history_len_before_user_turn // interval) < (
        history_len_after_user_turn // interval
    )

    if should_generate_image:
        asyncio.create_task(image_generation_task(list(history)))

    main_response = await main_pipeline_task(turn, loop)

    if main_response:
        history.append(main_response)

    await manager.send_json({"type": "stream_end"})


async def process_user_audio(audio_bytes: bytes) -> str:
    if not ctx.asr_engine:
        logger.warning("⚠️ [SYSTEM] 音声データ受信、しかしASRエンジンが無効です。")
        return ""
    try:
        audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
        audio_segment = (
            audio_segment.set_frame_rate(16000).set_sample_width(2).set_channels(1)
        )
        pcm_audio_bytes = audio_segment.raw_data
        logger.info(f"🎤 [DECODE] 音声デコード成功: {len(pcm_audio_bytes)} bytes")

        user_message = await asyncio.to_thread(
            ctx.asr_engine.process_audio, pcm_audio_bytes
        )
        return user_message
    except Exception as e:
        logger.error(f"❌ [ASR] 音声処理中にエラー: {e}")
        traceback.print_exc()
        return ""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global history

    if ctx is None or history is None:
        logger.error("❌ [SYSTEM] アプリケーションが初期化されていません。")
        await websocket.close(code=1011, reason="Server not initialized")
        return

    # クライアントに設定情報を送信
    await manager.send_json(
        {
            "type": "config",
            "data": {
                "user_input_mode": ctx.cfg.chat.user.input,
                "user_name": ctx.llmcfg.user_name,
            },
        }
    )

    for message in history:
        await manager.send_json({"type": "history", "data": message})
    try:
        initial_turn, history_len_before = (
            ctx.initial_turn,
            len(history),
        )
        if (history_len_before + 1) % ctx.cfg.chat.image.interval == 0:
            await manager.send_json({"type": "next_speaker", "data": initial_turn})
            asyncio.create_task(run_ai_turn(initial_turn, history_len_before))
        while True:
            message = await websocket.receive()
            user_message = ""
            if "text" in message:
                user_message = message["text"]
            elif "bytes" in message:
                user_message = await process_user_audio(message["bytes"])
                await manager.send_json(
                    {"type": "user_transcription", "data": user_message}
                )

            if not user_message:
                continue

            history_len_before_user_turn, user_name = len(history), ctx.llmcfg.user_name
            history.append({"name": user_name, "content": user_message})
            next_turn = ctx.llms.get_next_speaker(history, except_names=[user_name])
            await manager.send_json({"type": "next_speaker", "data": next_turn})
            asyncio.create_task(run_ai_turn(next_turn, history_len_before_user_turn))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"❌ [SYSTEM] WebSocketループエラー: {e}")
        traceback.print_exc()
    finally:
        manager.disconnect(websocket)


app.mount(
    GENERATED_IMAGES_URL_PATH,
    StaticFiles(directory=GENERATED_IMAGES_DIR),
    name="images",
)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")
