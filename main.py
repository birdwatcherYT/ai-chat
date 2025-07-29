# main.py

import asyncio
import logging
import os
import random
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import sounddevice as sd
from dotenv import load_dotenv

from src.app_context import AppContext
from src.asr.base import SpeechToText
from src.img.base import ImageGenerator
from src.llm.common import LLMConfig
from src.llm.llm import LLMs
from src.logger import get_logger
from src.tts.base import TextToSpeech
from src.turn_manager import TurnManager
from src.utils import open_image

load_dotenv()
logger = get_logger(__name__, level=logging.INFO)


def log_task(name: str, event: str, details: str = ""):
    details_str = f"| Details: {details}" if details else ""
    logger.debug(f"\n>>>> LOG | Task: {name:<20} | Event: {event:<15} {details_str}")


async def playback_worker(queue: asyncio.Queue):
    while True:
        data, sr = await queue.get()
        log_task("PLAYBACK_WORKER", "START")
        sd.play(data, sr)
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

        # cfgが存在し、かつengineがNoneでない場合のみ音声合成を実行
        if not cfg or not cfg.engine:
            log_task("SYNTHESIS_WORKER", "SKIP", "No voice engine configured")
            synthesis_queue.task_done()
            continue

        try:
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
        for content in utter_chain.stream(utter_prompt_vars):
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


class ChatState:
    def __init__(
        self,
        cfg: SimpleNamespace,
        llmcfg: LLMConfig,
        llms: LLMs,
        turn_manager: TurnManager,
        asr: SpeechToText | None,
        img_generator: ImageGenerator,
        initial_history: list[dict[str, str]],
        initial_turn: str,
    ):
        self.cfg = cfg
        self.llmcfg = llmcfg
        self.llms = llms
        self.turn_manager = turn_manager
        self.asr = asr
        self.img_generator = img_generator
        self.history = initial_history
        self.turn = initial_turn
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
        utter_chain = state.llms.get_utter_chain(state.history)
        utter_prompt_vars = {"speaker": state.turn}
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
        state.history.append(
            {"name": state.turn, "content": current_message, "type": "text"}
        )
        if len(state.history) % state.cfg.chat.image.interval == 0:
            asyncio.create_task(
                generate_and_open_image(
                    state.img_generator,
                    list(state.history),
                    state.cfg.chat.image.edit,
                )
            )

        log_task("NEXT_TURN_TASK", "CREATE")
        try:
            next_speaker = await asyncio.to_thread(
                state.turn_manager.get_next_speaker,
                list(state.history),
                state.turn,
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

    # AppContextで初期化をまとめる
    ctx = AppContext(cfg)

    # ログ出力の修正
    asr_mode = None
    if ctx.cfg.chat.user.input == "voice":
        asr_mode = getattr(ctx.cfg.chat.user, "asr_engine", "None")
    logger.info(
        f"✅ [SYSTEM] 初期化完了 (Input: {ctx.cfg.chat.user.input}, ASR: {asr_mode})"
    )

    # ChatStateに必要なものを渡して初期化
    state = ChatState(
        cfg=ctx.cfg,
        llmcfg=ctx.llmcfg,
        llms=ctx.llms,
        turn_manager=ctx.turn_manager,
        asr=ctx.asr_engine,
        img_generator=ctx.img_generator,
        initial_history=list(ctx.initial_history),
        initial_turn=ctx.initial_turn,
    )

    playback_queue = asyncio.Queue()
    asyncio.create_task(playback_worker(playback_queue))
    # synthesis_workerに必要なエンジンと設定を渡す
    asyncio.create_task(
        synthesis_worker(
            state.synthesis_queue, playback_queue, ctx.tts_engines, ctx.ai_config
        )
    )

    print("=" * 20, "チャットを開始します", "=" * 20)

    # 最初のチャットループタスクを開始
    asyncio.create_task(chat_loop(state))

    log_task("MAIN", "RUNNING")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        log_task("MAIN", "SHUTDOWN")
        print("\nチャットを終了します。")
