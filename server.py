import asyncio
import base64
import json
import traceback
from io import BytesIO
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

from src.img.base import ImageGenerator
from src.img.fastsd import FastSD
from src.img.gemini_img import GeminiImg
from src.lmm.common import LLMConfig, history_to_text
from src.lmm.llm import LLMs
from src.logger import get_logger
from src.tts.aivisspeech import AivisSpeech
from src.tts.coeiroink import CoeiroInk
from src.tts.voicevox import VoiceVox

GENERATED_IMAGES_DIR = "generated_images"
GENERATED_IMAGES_URL_PATH = "/images"
load_dotenv()
logger = get_logger(__name__, level="INFO")

app = FastAPI()
cfg, llmcfg, llms, engines, ai_config, history, img_generator, asr_engine = [None] * 8
llm_text_queue = asyncio.Queue()
audio_data_queue = asyncio.Queue()


def load_config_and_init():
    global cfg, llmcfg, llms, engines, ai_config, history, img_generator, asr_engine
    try:
        with open("config.yaml", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
            cfg = json.loads(
                json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d)
            )
        llmcfg = LLMConfig(cfg)
        llms = LLMs(llmcfg)

        image_model = cfg.chat.image.model
        img_generator = None
        if image_model == "fastsd":
            img_generator = FastSD(llms, **vars(cfg.fastsd))
        elif image_model == "gemini_image":
            img_generator = GeminiImg(llms, **vars(cfg.gemini_image))
        elif image_model == "mock":
            img_generator = ImageGenerator(llms)

        engines = {
            "voicevox": VoiceVox(),
            "coeiroink": CoeiroInk(),
            "aivisspeech": AivisSpeech(),
        }
        ai_config = {ai.name: ai.voice for ai in cfg.chat.ai}
        history = [
            {
                "name": llmcfg.format(item.name),
                "content": llmcfg.format(item.content),
            }
            for item in cfg.chat.initial_message
        ]
        user_input_mode = cfg.chat.user.input
        if user_input_mode == "vosk":
            from src.asr.vosk_asr import VoskASR

            asr_engine = VoskASR(**vars(cfg.vosk))
        elif user_input_mode == "whisper":
            from src.asr.whisper_asr import WhisperASR

            asr_engine = WhisperASR(**vars(cfg.whisper), **vars(cfg.webrtcvad))
        elif user_input_mode == "gemini":
            from src.asr.gemini_asr import GeminiASR

            asr_engine = GeminiASR(cfg.gemini.model, **vars(cfg.webrtcvad))
        else:
            asr_engine = None
        logger.info(f"✅ [SYSTEM] 初期化完了 (ASR: {user_input_mode})")
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
        utter_chain = llms.get_utter_chain()
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
            if answer_segment and answer_segment[-1] in cfg.chat.streaming_voice_output:
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
        voice_config = ai_config.get(speaker_name)
        if not voice_config:
            continue
        try:
            data, sr = await engines[voice_config.engine].synthesize_async(
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
    # generate_imageは(URL, パス)のタプルを返すので、URLのみ受け取る
    image_url, _ = await asyncio.to_thread(
        img_generator.generate_image, current_history, cfg.chat.image.edit
    )
    if image_url:
        await manager.send_json({"type": "image", "url": image_url})
    await manager.send_json({"type": "status_remove", "data": {"id": task_id}})


async def main_pipeline_task(turn: str, loop: asyncio.AbstractEventLoop):
    llm_task = asyncio.to_thread(llm_stream_blocking_task, turn, list(history), loop)
    audio_task = asyncio.gather(synthesis_consumer(), audio_sender_consumer())
    results = await asyncio.gather(llm_task, audio_task)
    return results[0]


async def run_ai_turn(turn: str, history_len_before_user_turn: int):
    global history
    tasks_to_run, loop = [], asyncio.get_running_loop()
    main_pipeline = main_pipeline_task(turn, loop)
    tasks_to_run.append(main_pipeline)
    history_len_after_user_turn = len(history)
    interval = cfg.chat.image.interval
    if (history_len_before_user_turn // interval) < (
        history_len_after_user_turn // interval
    ):
        tasks_to_run.append(image_generation_task(list(history)))
    results = await asyncio.gather(*tasks_to_run)
    if results[0]:
        history.append(results[0])
    await manager.send_json({"type": "stream_end"})


async def process_user_audio(audio_bytes: bytes) -> str:
    if not asr_engine:
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
            asr_engine.process_audio, pcm_audio_bytes
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

    # クライアントに設定情報を送信
    await manager.send_json(
        {
            "type": "config",
            "data": {
                "user_input_mode": cfg.chat.user.input,
                "user_name": llmcfg.user_name,
            },
        }
    )

    for message in history:
        await manager.send_json({"type": "history", "data": message})
    try:
        initial_turn, history_len_before = (
            llmcfg.format(cfg.chat.initial_turn),
            len(history),
        )
        if (history_len_before + 1) % cfg.chat.image.interval == 0:
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

            history_len_before_user_turn, user_name = len(history), llmcfg.user_name
            history.append({"name": user_name, "content": user_message})
            next_turn = llms.get_next_speaker(history, except_names=[user_name])
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
