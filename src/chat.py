from dotenv import load_dotenv

load_dotenv()  # .envファイルから環境変数を読み込む

import json
import asyncio
import sounddevice as sd
from invoke.config import Config

from .tts.base import TextToSpeech
from .asr.base import SpeechToText
from .tts.voicevox import VoiceVox
from .tts.coeiroink import CoeiroInk
from .tts.aivisspeech import AivisSpeech
from .lmm.img import generate_image
from .lmm.llm import LLMs, history_to_text


async def playback_worker(queue: asyncio.Queue, asr: SpeechToText):
    """再生キューから順次オーディオデータを取り出して再生するワーカー"""
    while True:
        data, sr = await queue.get()
        if asr is not None:
            asr.pause()  # マイクをOFFにする
        sd.play(data, sr)
        await asyncio.to_thread(sd.wait)
        if asr is not None:
            asr.resume()  # 再生終了後にマイクをONにする
        queue.task_done()


async def synthesis_worker(
    synthesis_queue: asyncio.Queue,
    playback_queue: asyncio.Queue,
    engines: dict[str, TextToSpeech],
    ai_config: dict[str, Config],
):
    """合成キューから順次テキストを取り出して音声合成し、再生キューに投入するワーカー"""
    while True:
        name, text_segment = await synthesis_queue.get()
        cfg = ai_config[name]
        if cfg["engine"] is not None:
            tts = engines[cfg["engine"]]
            data, sr = await tts.synthesize_async(text_segment, **cfg["config"])
            await playback_queue.put((data, sr))
        synthesis_queue.task_done()


async def chat_start(cfg: Config):
    user_name = cfg.chat.user.name
    ai_names = {f"ai{i}_name": ai["name"] for i, ai in enumerate(cfg.chat.ai)}

    # LLMの設定
    llms = LLMs(cfg, user_name=user_name, ai_names=ai_names)
    utter_chain = llms.get_utter_chain()

    # 音声認識の設定
    asr: SpeechToText = None
    if cfg.chat.user.input == "vosk":
        from .asr.vosk_asr import VoskASR

        asr = VoskASR(**cfg.vosk)
    elif cfg.chat.user.input == "whisper":
        from .asr.whisper_asr import WhisperASR

        asr = WhisperASR(**cfg.whisper, **cfg.webrtcvad)

    # 音声合成の設定
    engines = {
        "voicevox": VoiceVox(),
        "coeiroink": CoeiroInk(),
        "aivisspeech": AivisSpeech(),
    }
    ai_config = {ai["name"]: ai["voice"] for ai in cfg.chat.ai}
    if cfg.chat.user.input == "ai":
        ai_config[user_name] = cfg.chat.user.voice

    # 再生・合成用のグローバルなキューとワーカーを起動
    playback_queue = asyncio.Queue()
    synthesis_queue = asyncio.Queue()
    asyncio.create_task(playback_worker(playback_queue, asr))
    asyncio.create_task(
        synthesis_worker(synthesis_queue, playback_queue, engines, ai_config)
    )

    print(f"Chat Start: user.input={cfg.chat.user.input}", flush=True)

    # 発話履歴をリストで管理
    history = [
        {
            "name": item["name"].format(user_name=user_name, **ai_names),
            "content": item["content"].format(user_name=user_name, **ai_names),
        }
        for item in cfg.chat.initial_message
    ]

    prev_turn = None
    turn = cfg.chat.initial_turn.format(user_name=user_name, **ai_names)

    # チャット全体をループで実行（各ターンごとにユーザー入力とテキスト生成を処理）
    print(f"{turn}: ", end="", flush=True)
    while True:
        if turn is None:
            turn = llms.get_next_speaker(history, except_names=[prev_turn])
            print(f"{turn}: ", end="", flush=True)
        # ユーザー入力取得（音声入力の場合は asr.audio_input、テキストの場合は input()）
        if turn == user_name and cfg.chat.user.input != "ai":
            if cfg.chat.user.input == "text":
                user_input = await asyncio.to_thread(input)
            else:
                user_input = await asyncio.to_thread(asr.audio_input)
                print(user_input, flush=True)

            history.append({"name": user_name, "content": user_input})

            turn = llms.get_next_speaker(history, except_names=[prev_turn])
            print(f"{turn}: ", end="", flush=True)

        text_queue = asyncio.Queue()

        async def process_text_queue():
            nonlocal history, turn, prev_turn
            answer = ""

            while True:
                chunk = await text_queue.get()
                if chunk is None:
                    break  # ストリーム終了の合図
                # 生成されたチャンクを即座に表示
                print(chunk.content, end="", flush=True)
                answer += chunk.content
                # 指定された文字が現れたタイミングで音声合成
                if answer and answer[-1] in cfg.chat.streaming_voice_output:
                    await synthesis_queue.put((turn, answer))
                    history.append({"name": turn, "content": answer})
                    answer = ""
                text_queue.task_done()

            answer = answer.strip()
            if turn and answer:
                await synthesis_queue.put((turn, answer))
                history.append({"name": turn, "content": answer})
                answer = ""

            # 上記でメッセージ追記後の改行
            print()
            prev_turn = turn
            turn = None

        # テキスト処理タスクを開始
        processing_task = asyncio.create_task(process_text_queue())

        loop = asyncio.get_running_loop()
        # 実行時にのみ必要な変数を渡す
        utter_prompt_vars = {"speaker": turn, "messages": history_to_text(history)}

        def generate_text():
            for chunk in utter_chain.stream(utter_prompt_vars):
                asyncio.run_coroutine_threadsafe(text_queue.put(chunk), loop)
            # ストリーム終了の合図として None を投入
            asyncio.run_coroutine_threadsafe(text_queue.put(None), loop)

        await asyncio.to_thread(generate_text)
        # テキスト処理タスクが完了するのを待つ
        await processing_task
