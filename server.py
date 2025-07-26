import asyncio
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from invoke.config import Config
import yaml
import traceback

from dotenv import load_dotenv

load_dotenv()

# 既存のロジックをインポート
from src.tts.base import TextToSpeech
from src.tts.voicevox import VoiceVox
from src.tts.coeiroink import CoeiroInk
from src.tts.aivisspeech import AivisSpeech
from src.lmm.common import LLMConfig, history_to_text
from src.lmm.llm import LLMs

# --- グローバル変数の設定 ---
app = FastAPI()
cfg = None
llmcfg = None
llms = None
engines = {}
ai_config = {}
history = []

# パイプライン処理のためのキュー
llm_text_queue = asyncio.Queue()
audio_data_queue = asyncio.Queue()


# --- 初期化処理 ---
def load_config_and_init():
    global cfg, llmcfg, llms, engines, ai_config, history
    try:
        with open("invoke-utf8.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        cfg = Config(config_data)

        llmcfg = LLMConfig(cfg)
        llms = LLMs(llmcfg)

        engines = {
            "voicevox": VoiceVox(),
            "coeiroink": CoeiroInk(),
            "aivisspeech": AivisSpeech(),
        }
        ai_config = {ai["name"]: ai["voice"] for ai in cfg.chat.ai}

        history = [
            {
                "name": llmcfg.format(item["name"]),
                "content": llmcfg.format(item["content"]),
            }
            for item in cfg.chat.initial_message
        ]
        print("✅ [SYSTEM] 設定の読み込みと初期化が完了しました。")
    except Exception as e:
        print(f"❌ [SYSTEM] 初期化中にエラーが発生しました: {e}")
        traceback.print_exc()


# --- WebSocketの接続管理 ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print("✅ [SYSTEM] クライアントが接続しました。")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("🔌 [SYSTEM] クライアントが切断されました。")

    async def send_json(self, data: dict):
        if self.active_connections:
            try:
                await self.active_connections[0].send_json(data)
            except Exception as e:
                print(f"❌ [SYSTEM] WebSocket送信エラー: {e}")


manager = ConnectionManager()

# --- パイプラインのワーカー定義 ---


# LLMからのテキストを生成し、キューに入れるプロデューサー
async def llm_stream_producer(turn: str):
    global history
    print(f"🧠 [AI] '{turn}'の発言生成を開始します。")
    full_response = ""
    answer_segment = ""

    utter_chain = llms.get_utter_chain()
    utter_prompt_vars = {"speaker": turn, "messages": history_to_text(history)}

    try:
        for chunk in utter_chain.stream(utter_prompt_vars):
            chunk_content = chunk.content
            full_response += chunk_content
            answer_segment += chunk_content
            await manager.send_json(
                {"type": "chunk", "data": {"name": turn, "content": chunk_content}}
            )

            if answer_segment and answer_segment[-1] in cfg.chat.streaming_voice_output:
                print(
                    f"📦 [AI] テキストセグメントを合成キューに投入: '{answer_segment[:20]}...'"
                )
                await llm_text_queue.put((turn, answer_segment))
                answer_segment = ""

        if answer_segment:
            print(
                f"📦 [AI] 残りのテキストセグメントを合成キューに投入: '{answer_segment[:20]}...'"
            )
            await llm_text_queue.put((turn, answer_segment))

        history.append({"name": turn, "content": full_response})
        print(f"📝 [AI] '{turn}'の発言が完了しました。")

    except Exception as e:
        print(f"❌ [AI] LLMプロデューサーでエラー: {e}")
        traceback.print_exc()
    finally:
        await llm_text_queue.put(None)  # パイプライン終了シグナル


# テキストを音声合成し、音声キューに入れるコンシューマー
async def synthesis_consumer():
    print("🔊 [SYSTEM] 音声合成ワーカーを開始します。")
    while True:
        task = await llm_text_queue.get()
        if task is None:
            await audio_data_queue.put(None)
            break

        speaker_name, text = task
        print(f"🎙️ [SYNTH] 合成開始: [{speaker_name}] '{text[:20]}...'")
        voice_config = ai_config.get(speaker_name)
        if not voice_config or not voice_config.get("engine"):
            continue

        tts_engine: TextToSpeech = engines[voice_config["engine"]]
        try:
            data, sr = await tts_engine.synthesize_async(text, **voice_config["config"])
            if data is not None and sr is not None:
                await audio_data_queue.put((data, sr, str(data.dtype)))
                print(f"🎤 [SYNTH] 合成成功、音声キューに投入。")
        except Exception as e:
            print(f"❌ [SYNTH] 音声合成でエラー: {e}")
            traceback.print_exc()
    print("⛔ [SYSTEM] 音声合成ワーカーを終了します。")


# 音声データをクライアントに送信するコンシューマー
async def audio_sender_consumer():
    print("📡 [SYSTEM] 音声送信ワーカーを開始します。")
    while True:
        task = await audio_data_queue.get()
        if task is None:
            await manager.send_json({"type": "stream_end"})
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
            print(f"✅ [SEND] 音声データをクライアントに送信しました。")
        except Exception as e:
            print(f"❌ [SEND] 音声送信でエラー: {e}")
            traceback.print_exc()
    print("⛔ [SYSTEM] 音声送信ワーカーを終了します。")


# --- WebSocketエンドポイント ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global history

    # 接続時に現在のチャット履歴を送信
    for message in history:
        await websocket.send_json({"type": "history", "data": message})

    try:
        # 最初のターンがAIなら実行
        initial_turn = llmcfg.format(cfg.chat.initial_turn)
        if initial_turn != llmcfg.user_name:
            await manager.send_json({"type": "next_speaker", "data": initial_turn})
            asyncio.create_task(run_ai_pipeline(initial_turn))

        # クライアントからのメッセージを待つ
        while True:
            user_message = await websocket.receive_text()
            print(f"👤 [USER] 受信メッセージ: '{user_message}'")

            user_name = llmcfg.user_name
            history.append({"name": user_name, "content": user_message})

            # 話者決定
            print("🤔 [SYSTEM] 次の話者を決定中...")
            next_turn = llms.get_next_speaker(history, except_names=[user_name])
            print(f"👉 [SYSTEM] 次の話者は '{next_turn}' です。")

            # 次の話者をクライアントに通知
            await manager.send_json({"type": "next_speaker", "data": next_turn})

            # AIパイプラインを開始
            asyncio.create_task(run_ai_pipeline(next_turn))

    except WebSocketDisconnect:
        pass  # manager.disconnectはfinallyで処理
    finally:
        manager.disconnect(websocket)


async def run_ai_pipeline(turn: str):
    # パイプラインのワーカータスクを開始
    synthesis_task = asyncio.create_task(synthesis_consumer())
    audio_sender_task = asyncio.create_task(audio_sender_consumer())

    # LLMプロデューサーが完了するのを待つ
    await llm_stream_producer(turn)

    # 全てのワーカーが終了するのを待つ
    await synthesis_task
    await audio_sender_task
    print("🏁 [SYSTEM] AIのターン処理がすべて完了しました。")


# --- 静的ファイルの配信設定 ---
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")


# --- サーバー起動 ---
if __name__ == "__main__":
    load_config_and_init()
    uvicorn.run(app, host="0.0.0.0", port=8000)
