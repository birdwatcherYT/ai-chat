import asyncio
import os
import platform
import random
import subprocess
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import sounddevice as sd
from dotenv import load_dotenv

from src.asr.base import SpeechToText
from src.lmm.common import LLMConfig, history_to_text
from src.lmm.img import ImageGenerator
from src.lmm.llm import LLMs
from src.logger import get_logger
from src.tts.aivisspeech import AivisSpeech
from src.tts.base import TextToSpeech
from src.tts.coeiroink import CoeiroInk
from src.tts.voicevox import VoiceVox

load_dotenv()
logger = get_logger(__name__, level="INFO")


# --- HELPERS (変更なし) ---
def log_task(name: str, event: str, details: str = ""):
    details_str = f"| Details: {details}" if details else ""
    logger.debug(f"\n>>>> LOG | Task: {name:<20} | Event: {event:<15} {details_str}")


def open_image(image_path: str):
    log_task("OPEN_IMAGE", "START", image_path)
    system = platform.system()
    if not os.path.exists(image_path):
        logger.warning(f"⚠️ [SYSTEM] 画像ファイルが見つかりません: {image_path}")
        return
    try:
        if system == "Windows":
            subprocess.run(["start", "", image_path], check=True, shell=True)
        elif system == "Darwin":
            subprocess.run(["open", image_path], check=True)
        else:
            subprocess.run(["xdg-open", image_path], check=True)
    except Exception as e:
        logger.error(f"❌ [SYSTEM] 画像を開けませんでした: {e}")
    log_task("OPEN_IMAGE", "END", image_path)


# --- ASYNC WORKERS (一部修正) ---
async def playback_worker(queue: asyncio.Queue):
    while True:
        data, sr = await queue.get()
        log_task("PLAYBACK_WORKER", "START")
        # sd.playは非ブロッキングなのでto_threadは不要
        sd.play(data, sr)
        # sd.waitはブロッキングなのでto_threadが必要
        await asyncio.to_thread(sd.wait)
        log_task("PLAYBACK_WORKER", "END")
        queue.task_done()


async def synthesis_worker(
    synthesis_queue: asyncio.Queue,
    playback_queue: asyncio.Queue,
    engines: dict[str, TextToSpeech],
    ai_config: dict[str, SimpleNamespace],
):
    while True:
        name, text_segment = await synthesis_queue.get()
        log_task("SYNTHESIS_WORKER", "START", f"'{text_segment[:15]}...'")
        cfg = ai_config.get(name)
        if cfg and hasattr(cfg, "engine"):
            try:
                # SimpleNamespaceをvars()で辞書に変換してから展開
                data, sr = await engines[cfg.engine].synthesize_async(
                    text_segment, **vars(cfg.config)
                )
                if data is not None:
                    await playback_queue.put((data, sr))
            except Exception as e:
                logger.error(f"❌ [SYNTH] 音声合成エラー: {e}")
        log_task("SYNTHESIS_WORKER", "END", f"'{text_segment[:15]}...'")
        synthesis_queue.task_done()


def llm_stream_task(
    loop: asyncio.AbstractEventLoop,
    synthesis_queue: asyncio.Queue,
    utter_chain: Coroutine,
    utter_prompt_vars: dict,
    turn: str,
    streaming_chars: list[str],
) -> str:
    log_task("LLM_STREAM", "THREAD_START")
    full_response, answer_segment = "", ""
    try:
        for chunk in utter_chain.stream(utter_prompt_vars):
            content = chunk.content
            print(content, end="", flush=True)
            full_response += content
            answer_segment += content
            if answer_segment and answer_segment[-1] in streaming_chars:
                log_task("LLM_STREAM", "QUEUE_PUT", f"'{answer_segment[:15]}...'")
                asyncio.run_coroutine_threadsafe(
                    synthesis_queue.put((turn, answer_segment)), loop
                )
                answer_segment = ""
        if answer_segment:
            log_task("LLM_STREAM", "QUEUE_PUT", f"'{answer_segment[:15]}...'")
            asyncio.run_coroutine_threadsafe(
                synthesis_queue.put((turn, answer_segment)), loop
            )
        log_task("LLM_STREAM", "THREAD_END")
        return full_response
    except Exception as e:
        logger.error(f"❌ [LLM-THREAD] テキスト生成中にエラーが発生しました: {e}")
        return ""


# --- ARCHITECTURE (変更なし) ---
class ChatState:
    def __init__(
        self,
        cfg: SimpleNamespace,
        llmcfg: LLMConfig,
        llms: LLMs,
        asr: SpeechToText | None,
        image_generator: ImageGenerator,
    ):
        self.cfg = cfg
        self.llmcfg = llmcfg
        self.llms = llms
        self.asr = asr
        self.image_generator = image_generator
        self.history = [
            {
                "name": llmcfg.format(item.name),
                "content": llmcfg.format(item.content),
            }
            for item in cfg.chat.initial_message
        ]
        self.turn = llmcfg.format(cfg.chat.initial_turn)
        self.loop = asyncio.get_running_loop()
        self.synthesis_queue = asyncio.Queue()


async def chat_loop(state: ChatState):
    log_task("CHAT_LOOP", "TURN_START", f"Speaker: {state.turn}")
    print(f"\n[{state.turn}] > ", end="", flush=True)

    current_message = ""
    if state.turn == state.llmcfg.user_name and state.cfg.chat.user.input != "ai":
        if state.asr:
            user_input = await asyncio.to_thread(state.asr.audio_input)
            print(user_input, flush=True)
        else:
            user_input = await asyncio.to_thread(input)
        current_message = user_input
    else:
        utter_chain = state.llms.get_utter_chain()
        utter_prompt_vars = {
            "speaker": state.turn,
            "messages": history_to_text(state.history),
        }
        log_task("CHAT_LOOP", "LLM_TASK_START")
        full_response = await asyncio.to_thread(
            llm_stream_task,
            state.loop,
            state.synthesis_queue,
            utter_chain,
            utter_prompt_vars,
            state.turn,
            state.cfg.chat.streaming_voice_output,
        )
        print()
        log_task("CHAT_LOOP", "LLM_TASK_END")
        current_message = full_response

    if current_message:
        state.history.append({"name": state.turn, "content": current_message})
        if len(state.history) % state.cfg.chat.image.interval == 0:
            asyncio.create_task(
                generate_and_open_image(
                    state.image_generator,
                    list(state.history),
                    state.cfg.chat.image.edit,
                )
            )

        log_task("NEXT_TURN_TASK", "CREATE")
        except_names = [state.turn]
        try:
            next_speaker = await asyncio.to_thread(
                state.llms.get_next_speaker, list(state.history), except_names
            )
            log_task("NEXT_TURN_TASK", "RESOLVED", f"New speaker: {next_speaker}")
            state.turn = next_speaker
        except Exception as e:
            log_task("NEXT_TURN_TASK", "ERROR", str(e))
            ai_names = list(state.llmcfg.ai_names.values())
            state.turn = random.choice(ai_names) if ai_names else state.llmcfg.user_name

    asyncio.create_task(chat_loop(state))


async def generate_and_open_image(
    image_generator: ImageGenerator, history: list[dict[str, str]], edit: bool
):
    log_task("IMAGE_GENERATION", "TASK_STARTED")
    image_url, image_path = await asyncio.to_thread(
        image_generator.generate_image, history, edit
    )
    if image_path:
        await asyncio.to_thread(open_image, image_path)
    log_task("IMAGE_GENERATION", "TASK_FINISHED")


async def chat_start(cfg: SimpleNamespace):
    """非同期タスクをセットアップし、チャットループを開始する"""
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 5 or 32)
    loop.set_default_executor(executor)

    llmcfg = LLMConfig(cfg)
    llms = LLMs(llmcfg)
    image_generator = ImageGenerator(llmcfg, "generated_images", "/images")
    asr: SpeechToText | None = None
    user_input_mode = cfg.chat.user.input
    log_task("MAIN", "INIT", f"User input mode: {user_input_mode}")
    if user_input_mode == "vosk":
        from src.asr.vosk_asr import VoskASR

        # SimpleNamespaceをvars()で辞書に変換してから展開
        asr = VoskASR(**vars(cfg.vosk))
    elif user_input_mode == "whisper":
        from src.asr.whisper_asr import WhisperASR

        # SimpleNamespaceをvars()で辞書に変換してから展開
        asr = WhisperASR(**vars(cfg.whisper), **vars(cfg.webrtcvad))
    elif user_input_mode == "gemini":
        from src.asr.gemini_asr import GeminiASR

        # SimpleNamespaceをvars()で辞書に変換してから展開
        asr = GeminiASR(cfg.gemini.model, **vars(cfg.webrtcvad))

    state = ChatState(cfg, llmcfg, llms, asr, image_generator)

    engines = {
        "voicevox": VoiceVox(),
        "coeiroink": CoeiroInk(),
        "aivisspeech": AivisSpeech(),
    }
    ai_config = {ai.name: ai.voice for ai in cfg.chat.ai}
    if cfg.chat.user.input == "ai":
        ai_config[llmcfg.user_name] = cfg.chat.user.voice
    playback_queue = asyncio.Queue()
    state.synthesis_queue = asyncio.Queue()
    asyncio.create_task(playback_worker(playback_queue))
    asyncio.create_task(
        synthesis_worker(state.synthesis_queue, playback_queue, engines, ai_config)
    )

    print("=" * 20, "チャットを開始します", "=" * 20)

    # 最初のチャットループタスクを開始
    asyncio.create_task(chat_loop(state))

    # プログラムが終了しないように、無限に待機する
    # Ctrl+Cで終了させる
    log_task("MAIN", "RUNNING")
    try:
        # asyncio.Event().wait() を使って、Ctrl+C を待つ
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        log_task("MAIN", "SHUTDOWN")
        print("\nチャットを終了します。")
