import asyncio
import base64
import os
import uvicorn
import traceback
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from invoke.config import Config
from dotenv import load_dotenv

load_dotenv()

from src.tts.base import TextToSpeech
from src.tts.voicevox import VoiceVox
from src.tts.coeiroink import CoeiroInk
from src.tts.aivisspeech import AivisSpeech
from src.lmm.common import LLMConfig, history_to_text
from src.lmm.llm import LLMs
from src.lmm.img import ImageGenerator

app = FastAPI()
cfg, llmcfg, llms, engines, ai_config, history, image_generator = [None] * 7
GENERATED_IMAGES_DIR = "generated_images"
GENERATED_IMAGES_URL_PATH = "/images"
llm_text_queue = asyncio.Queue()
audio_data_queue = asyncio.Queue()

def load_config_and_init():
    global cfg, llmcfg, llms, engines, ai_config, history, image_generator
    try:
        with open("invoke-utf8.yaml", "r", encoding="utf-8") as f:
            cfg = Config(yaml.safe_load(f))
        llmcfg = LLMConfig(cfg)
        llms = LLMs(llmcfg)
        os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
        image_generator = ImageGenerator(llmcfg, GENERATED_IMAGES_DIR, GENERATED_IMAGES_URL_PATH)
        engines = {"voicevox": VoiceVox(), "coeiroink": CoeiroInk(), "aivisspeech": AivisSpeech()}
        ai_config = {ai["name"]: ai["voice"] for ai in cfg.chat.ai}
        history = [{"name": llmcfg.format(item["name"]), "content": llmcfg.format(item["content"])} for item in cfg.chat.initial_message]
        print("✅ [SYSTEM] 設定の読み込みと初期化が完了しました。")
    except Exception as e: print(f"❌ [SYSTEM] 初期化中にエラーが発生しました: {e}"); traceback.print_exc()

class ConnectionManager:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None: cls._instance = super(ConnectionManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    def __init__(self): self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept(); self.active_connections.append(websocket); print("✅ [SYSTEM] クライアントが接続しました。")
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections: self.active_connections.remove(websocket)
        print("🔌 [SYSTEM] クライアントが切断されました。")
    async def send_json(self, data: dict):
        if self.active_connections:
            try: await self.active_connections[0].send_json(data)
            except Exception: print(f"❌ [SYSTEM] WebSocket送信エラー。")

manager = ConnectionManager()

def llm_stream_blocking_task(turn: str, current_history: list, loop: asyncio.AbstractEventLoop):
    print(f"🧠 [AI-THREAD] '{turn}'の発言生成を開始します。")
    full_response, answer_segment = "", ""
    try:
        utter_chain = llms.get_utter_chain()
        utter_prompt_vars = {"speaker": turn, "messages": history_to_text(current_history)}
        for chunk in utter_chain.stream(utter_prompt_vars):
            full_response += chunk.content; answer_segment += chunk.content
            asyncio.run_coroutine_threadsafe(manager.send_json({"type": "chunk", "data": {"name": turn, "content": chunk.content}}), loop)
            if answer_segment and answer_segment[-1] in cfg.chat.streaming_voice_output:
                asyncio.run_coroutine_threadsafe(llm_text_queue.put((turn, answer_segment)), loop); answer_segment = ""
        if answer_segment: asyncio.run_coroutine_threadsafe(llm_text_queue.put((turn, answer_segment)), loop)
        print(f"📝 [AI-THREAD] '{turn}'の発言が完了しました。")
        return {"name": turn, "content": full_response}
    except Exception as e: print(f"❌ [AI-THREAD] LLMプロデューサーでエラー: {e}"); traceback.print_exc(); return None
    finally: asyncio.run_coroutine_threadsafe(llm_text_queue.put(None), loop)

async def synthesis_consumer():
    while True:
        task = await llm_text_queue.get();
        if task is None: await audio_data_queue.put(None); break
        speaker_name, text = task
        voice_config = ai_config.get(speaker_name)
        if not voice_config or not voice_config.get("engine"): continue
        try:
            data, sr = await engines[voice_config["engine"]].synthesize_async(text, **voice_config["config"])
            if data is not None and sr is not None: await audio_data_queue.put((data, sr, str(data.dtype)))
        except Exception as e: print(f"❌ [SYNTH] 音声合成エラー: {e}")

async def audio_sender_consumer():
    while True:
        task = await audio_data_queue.get();
        if task is None: break
        data, sr, dtype = task
        try:
            encoded_audio = base64.b64encode(data.tobytes()).decode('utf-8')
            await manager.send_json({"type": "audio", "data": {"audio": encoded_audio, "samplerate": sr, "dtype": dtype}})
        except Exception as e: print(f"❌ [SEND] 音声送信エラー: {e}")

async def image_generation_task(current_history):
    task_id = "image_gen"
    print(f"⏳ [IMAGE-TASK] 画像生成タスクを開始します。")
    await manager.send_json({"type": "status_update", "data": {"id": task_id, "text": "🎨 画像を生成しています..."}})
    image_url = await asyncio.to_thread(image_generator.generate_image, current_history, cfg.chat.image.edit)
    if image_url: await manager.send_json({"type": "image", "url": image_url}); print(f"🖼️ [IMAGE-TASK] 画像URLを送信しました。")
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
    if (history_len_before_user_turn // interval) < (history_len_after_user_turn // interval):
        print(f"⏳ [IMAGE] 画像生成インターバルを検出。タスクを準備します。")
        tasks_to_run.append(image_generation_task(list(history)))

    print(f"🚀 [SYSTEM] {len(tasks_to_run)}個のタスクを並列実行します。")
    results = await asyncio.gather(*tasks_to_run)
    
    # main_pipelineの結果（LLMの応答）を取得
    llm_response = results[0]
    if llm_response: history.append(llm_response)

    await manager.send_json({"type": "stream_end"})
    print(f"🏁 [SYSTEM] '{turn}'のターン処理がすべて完了しました。")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global history
    for message in history: await manager.send_json({"type": "history", "data": message})
    try:
        initial_turn, history_len_before = llmcfg.format(cfg.chat.initial_turn), len(history)
        if (history_len_before + 1) % cfg.chat.image.interval == 0:
             await manager.send_json({"type": "next_speaker", "data": initial_turn})
             asyncio.create_task(run_ai_turn(initial_turn, history_len_before))
        while True:
            user_message = await websocket.receive_text()
            history_len_before_user_turn, user_name = len(history), llmcfg.user_name
            history.append({"name": user_name, "content": user_message})
            next_turn = llms.get_next_speaker(history, except_names=[user_name])
            print(f"👉 [SYSTEM] 次の話者: '{next_turn}'")
            await manager.send_json({"type": "next_speaker", "data": next_turn})
            asyncio.create_task(run_ai_turn(next_turn, history_len_before_user_turn))
    except WebSocketDisconnect: print("クライアント切断(WebSocketDisconnect)")
    except Exception as e: print(f"❌ [SYSTEM] WebSocketループエラー: {e}"); traceback.print_exc()
    finally: manager.disconnect(websocket)

app.mount(GENERATED_IMAGES_URL_PATH, StaticFiles(directory=GENERATED_IMAGES_DIR), name="images")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
@app.get("/")
async def read_root(): return FileResponse('frontend/index.html')

if __name__ == "__main__":
    load_config_and_init()
    uvicorn.run(app, host="0.0.0.0", port=8000)
