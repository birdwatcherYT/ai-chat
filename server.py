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

from src.app_context import AppContext
from src.logger import get_logger

load_dotenv()
logger = get_logger(__name__, level=logging.INFO)

app = FastAPI()

# --- グローバル変数 ---
ctx: AppContext | None = None
history: list[dict] | None = None
effective_gui_mode: str = "browser_asr"

llm_text_queue = asyncio.Queue()
audio_data_queue = asyncio.Queue()


def initialize(app_context: AppContext):
    """サーバーの状態とGUIモードを初期化する"""
    global ctx, history, effective_gui_mode
    try:
        ctx = app_context
        history = list(ctx.initial_history)
        app.mount(
            ctx.cfg.chat.image.url_path,
            StaticFiles(directory=ctx.cfg.chat.image.save_dir),
            name="images",
        )
        app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

        base_input_mode = ctx.cfg.chat.user.input
        if base_input_mode == "ai":
            effective_gui_mode = "ai"
        else:
            asr_engine = getattr(ctx.cfg.chat.user, "asr_engine", "browser")
            if asr_engine in ["vosk", "whisper", "gemini_asr"]:
                effective_gui_mode = "server_asr"
            else:
                effective_gui_mode = "browser_asr"

        logger.info(
            f"✅ [SYSTEM] 初期化完了 (Config Input: {base_input_mode}, Effective GUI Mode: {effective_gui_mode})"
        )

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
    turn: str,
    current_history: list,
    loop: asyncio.AbstractEventLoop,
    webcam_capture: str | None = None,
):
    full_response, answer_segment = "", ""
    try:
        utter_chain = ctx.llms.get_utter_chain(current_history, webcam_capture)
        utter_prompt_vars = {"speaker": turn}
        for content in utter_chain.stream(utter_prompt_vars):
            full_response += content
            answer_segment += content
            asyncio.run_coroutine_threadsafe(
                manager.send_json(
                    {"type": "chunk", "data": {"name": turn, "content": content}}
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
        return {"name": turn, "type": "text", "content": full_response}
    except Exception:
        traceback.print_exc()
        return None
    finally:
        # 発話の終わりにNoneをキューに入れて、音声処理の完了を待てるようにする
        asyncio.run_coroutine_threadsafe(llm_text_queue.put(None), loop)


async def synthesis_consumer():
    """(ワーカー) llm_text_queueからテキストを取得し、音声合成してaudio_data_queueに渡す"""
    logger.info("✅ [WORKER] Synthesis consumer started.")
    while True:
        task = await llm_text_queue.get()
        if task is None:
            await audio_data_queue.put(None)
            llm_text_queue.task_done()
            continue  # Noneを受け取ってもワーカーは終了しない

        speaker_name, text = task
        voice_config = ctx.ai_config.get(speaker_name)

        if not voice_config or not voice_config.engine:
            llm_text_queue.task_done()
            continue
        try:
            data, sr = await ctx.tts_engines[voice_config.engine].synthesize_async(
                text, **vars(voice_config.config)
            )
            if data is not None and sr is not None:
                await audio_data_queue.put((data, sr, str(data.dtype)))
        except Exception as e:
            logger.error(f"❌ [SYNTH] 音声合成エラー: {e}")
        finally:
            llm_text_queue.task_done()


async def audio_sender_consumer():
    """(ワーカー) audio_data_queueから音声データを取得し、クライアントに送信する"""
    logger.info("✅ [WORKER] Audio sender consumer started.")
    while True:
        task = await audio_data_queue.get()
        if task is None:
            audio_data_queue.task_done()
            continue  # Noneを受け取ってもワーカーは終了しない

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
        finally:
            audio_data_queue.task_done()


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


async def main_pipeline_task(
    turn: str, loop: asyncio.AbstractEventLoop, webcam_capture: str | None = None
):
    """LLMによるテキスト生成を実行し、音声送信の完了を待つ"""
    llm_task = asyncio.to_thread(
        llm_stream_blocking_task, turn, list(history), loop, webcam_capture
    )
    main_response = await llm_task
    # すべてのテキストが処理され、音声がクライアントに送られるのを待つ
    await llm_text_queue.join()
    await audio_data_queue.join()
    return main_response


async def run_single_turn(turn: str, webcam_capture: str | None = None):
    global history
    loop = asyncio.get_running_loop()
    history_len_before_turn = len(history)
    await manager.send_json({"type": "next_speaker", "data": turn})
    main_response = await main_pipeline_task(turn, loop, webcam_capture)
    await manager.send_json({"type": "utterance_end", "data": turn})
    if main_response:
        history.append(main_response)
        history_len_after_turn = len(history)
        interval = ctx.cfg.chat.image.interval
        if (history_len_before_turn // interval) < (history_len_after_turn // interval):
            asyncio.create_task(image_generation_task(list(history)))


async def conversation_flow(initial_turn: str, webcam_capture: str | None = None):
    """会話のフローを管理する。AIモードと通常モードの両方に対応。"""
    turn = initial_turn
    is_ai_mode = effective_gui_mode == "ai"
    try:
        while True:
            if not is_ai_mode and turn == ctx.llmcfg.user_name:
                break
            await run_single_turn(turn, webcam_capture)
            last_speaker = turn
            turn = await asyncio.to_thread(
                ctx.turn_manager.get_next_speaker,
                list(history),
                last_speaker=last_speaker,
            )
            if is_ai_mode:
                await asyncio.sleep(1)
        if not is_ai_mode:
            await manager.send_json({"type": "conversation_end"})
    except asyncio.CancelledError:
        logger.info("🤖 [SYSTEM] Conversation flow was cancelled.")
        await manager.send_json({"type": "conversation_stopped"})
    except Exception as e:
        logger.error(f"❌ [SYSTEM] Conversation flow error: {e}")
        traceback.print_exc()


async def process_user_audio(audio_bytes: bytes) -> str:
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
    global history, effective_gui_mode

    if ctx is None:
        logger.error("❌ [SYSTEM] アプリケーションが初期化されていません。")
        await websocket.close(code=1011, reason="Server not initialized")
        return

    ctx.turn_manager.reset()
    history = list(ctx.initial_history)
    conversation_task = None

    # ワーカータスクをここで起動し、接続中はずっと常駐させる
    synth_task = asyncio.create_task(synthesis_consumer())
    sender_task = asyncio.create_task(audio_sender_consumer())

    await manager.send_json(
        {
            "type": "config",
            "data": {
                "user_input_mode": effective_gui_mode,
                "user_name": ctx.llmcfg.user_name,
                "character_icons": ctx.character_icons,
            },
        }
    )
    for message in history:
        text_message = {"name": message["name"], "content": message["content"]}
        if message["type"] == "image":
            text_message["content"] = "(画像添付)"
        await manager.send_json({"type": "history", "data": text_message})

    try:
        # 常にクライアントからのメッセージを待つループ
        while True:
            raw_message = await websocket.receive()
            user_message_text = ""
            user_message_image = None
            webcam_capture = None

            if "bytes" in raw_message:
                if not ctx.asr_engine:
                    logger.warning(
                        "⚠️ [SYSTEM] サーバーサイドASRが無効な状態で音声データを受信しました。無視します。"
                    )
                    continue
                user_message_text = await process_user_audio(raw_message["bytes"])
                if user_message_text:
                    await manager.send_json(
                        {"type": "user_transcription", "data": user_message_text}
                    )
                else:
                    logger.info(
                        "🎤 [ASR] 認識結果が空のため、クライアントにリトライを要求します。"
                    )
                    await manager.send_json({"type": "retry_audio_input"})
                    continue

            elif "text" in raw_message:
                data = json.loads(raw_message["text"])
                msg_type = data.get("type")

                if msg_type == "start_ai_conversation":
                    if effective_gui_mode == "ai":
                        logger.info(
                            "🤖 [SYSTEM] Client requested to start AI conversation."
                        )
                        if conversation_task and not conversation_task.done():
                            conversation_task.cancel()
                        initial_turn = ctx.initial_turn
                        conversation_task = asyncio.create_task(
                            conversation_flow(initial_turn)
                        )
                    continue

                elif msg_type == "stop_ai_conversation":
                    if conversation_task and not conversation_task.done():
                        logger.info(
                            "🤖 [SYSTEM] AI conversation stop requested by client."
                        )
                        conversation_task.cancel()
                    continue

                user_message_text = data.get("text", "")
                user_message_image = data.get("image")
                webcam_capture = data.get("webcam_capture")

            if not user_message_text and not user_message_image and not webcam_capture:
                continue

            user_name = ctx.llmcfg.user_name
            if user_message_text:
                history.append(
                    {"name": user_name, "type": "text", "content": user_message_text}
                )
            if user_message_image:
                history.append(
                    {"name": user_name, "type": "image", "content": user_message_image}
                )

            next_turn = await asyncio.to_thread(
                ctx.turn_manager.get_next_speaker, list(history), last_speaker=user_name
            )
            if conversation_task and not conversation_task.done():
                conversation_task.cancel()
            conversation_task = asyncio.create_task(
                conversation_flow(next_turn, webcam_capture)
            )

    except WebSocketDisconnect:
        logger.info("👋 [SYSTEM] クライアントが切断しました。")
    except Exception as e:
        logger.error(f"❌ [SYSTEM] WebSocketループエラー: {e}")
        traceback.print_exc()
    finally:
        logger.info("Cancelling worker and conversation tasks...")
        if conversation_task and not conversation_task.done():
            conversation_task.cancel()
        synth_task.cancel()
        sender_task.cancel()
        await asyncio.gather(
            conversation_task, synth_task, sender_task, return_exceptions=True
        )
        logger.info("All tasks cancelled.")
        manager.disconnect(websocket)


@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")
