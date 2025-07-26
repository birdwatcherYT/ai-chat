import argparse
import asyncio
import sys

from dotenv import load_dotenv

from src.app_context import load_config
from src.logger import get_logger

logger = get_logger(__name__, level="INFO")

load_dotenv()

# --- 各タスクの定義 ---


def task_chat():
    """AIとのチャットを開始します"""
    from main import chat_start

    config = load_config()
    logger.info("✅ 設定を読み込みました。チャットを開始します。")
    asyncio.run(chat_start(config))


def task_server():
    """Web UIサーバーを起動します"""
    import uvicorn

    from server import app, load_config_and_init

    logger.info("✅ 設定を読み込み、Webサーバーを初期化します。")
    load_config_and_init()  # server.py内の初期化関数を呼び出し
    logger.info("🚀 Webサーバーを http://localhost:5000 で起動します。")
    uvicorn.run(app, host="localhost", port=5000)


def task_tts_list(args):
    """指定されたTTSエンジンの話者一覧を表示します"""
    engine = args.engine.lower()
    if engine == "voicevox":
        from src.tts.voicevox import VoiceVox

        VoiceVox().print_speakers()
    elif engine == "coeiroink":
        from src.tts.coeiroink import CoeiroInk

        CoeiroInk().print_speakers()
    elif engine == "aivisspeech":
        from src.tts.aivisspeech import AivisSpeech

        AivisSpeech().print_speakers()
    else:
        logger.error(f"❌ [エラー] 未知のTTSエンジン: {engine}")
        sys.exit(1)


def task_tts(args):
    """指定されたTTSエンジンで音声合成をテストします"""
    import sounddevice as sd

    config = load_config()
    engine = args.engine.lower()
    text_to_synthesize = args.text

    tts_instance = None
    tts_config = None
    if engine == "voicevox":
        from src.tts.voicevox import VoiceVox

        tts_instance = VoiceVox()
        tts_config = config.voicevox
    elif engine == "coeiroink":
        from src.tts.coeiroink import CoeiroInk

        tts_instance = CoeiroInk()
        tts_config = config.coeiroink
    elif engine == "aivisspeech":
        from src.tts.aivisspeech import AivisSpeech

        tts_instance = AivisSpeech()
        tts_config = config.aivisspeech
    else:
        logger.error(f"❌ [エラー] 未知のTTSエンジン: {engine}")
        sys.exit(1)

    logger.info(f"🎤 {engine.upper()}で音声合成をテストします: '{text_to_synthesize}'")
    # SimpleNamespaceをvars()で辞書に変換してから展開
    data, sr = tts_instance.synthesize(text_to_synthesize, **vars(tts_config))
    sd.play(data, sr)
    sd.wait()
    logger.info("✅ 音声合成テストが完了しました。")


def task_asr(args):
    """指定されたASRエンジンで音声認識をテストします"""
    config = load_config()
    engine = args.engine.lower()

    logger.info(f"👂 {engine}で音声認識を開始します。")

    asr_instance = None
    if engine == "whisper":
        from src.asr.whisper_asr import WhisperASR

        # SimpleNamespaceをvars()で辞書に変換してから展開
        asr_instance = WhisperASR(**vars(config.whisper), **vars(config.webrtcvad))
    elif engine == "vosk":
        from src.asr.vosk_asr import VoskASR

        # SimpleNamespaceをvars()で辞書に変換してから展開
        asr_instance = VoskASR(**vars(config.vosk))
    elif engine == "gemini":
        from src.asr.gemini_asr import GeminiASR

        # SimpleNamespaceをvars()で辞書に変換してから展開
        asr_instance = GeminiASR(config.gemini.model, **vars(config.webrtcvad))
    else:
        logger.error(f"❌ [エラー] 未知のASRエンジン: {engine}")
        sys.exit(1)

    if args.loop:
        logger.info("🔄 連続認識モードで実行中... (Ctrl+Cで停止)")
        while True:
            try:
                print(asr_instance.audio_input())
            except KeyboardInterrupt:
                logger.info("👋 連続認識を停止しました。")
                break
            except Exception as e:
                logger.error(f"❌ [エラー] 音声認識中にエラーが発生しました: {e}")
                break
    else:
        print(asr_instance.audio_input())
    logger.info("✅ 音声認識テストが完了しました。")


def task_fastsd_status(args):
    """FastSDのシステム情報とモデル一覧を取得して表示します"""
    import json

    from src.img.fastsd import FastSD
    from src.llm.common import LLMConfig
    from src.llm.llm import LLMs

    config = load_config()
    llmcfg = LLMConfig(config)
    llms = LLMs(llmcfg)
    common_args = {
        "llms": llms,
        "save_dir": config.chat.image.save_dir,
        "url_path": config.chat.image.url_path,
    }

    client = FastSD(**common_args, **vars(config.fastsd))

    # システム情報を取得
    print("=== システム情報 ===")
    info = client.get_system_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))

    # モデル一覧を取得
    print("\n=== 利用可能なモデル一覧 ===")
    models = client.get_available_models()
    print(json.dumps(models, indent=2, ensure_ascii=False))


def task_img_gen(args):
    """指定されたエンジンで画像を生成します"""
    from src.llm.common import LLMConfig
    from src.llm.llm import LLMs
    from src.utils import open_image

    config = load_config()
    llmcfg = LLMConfig(config)
    llms = LLMs(llmcfg)
    common_args = {
        "llms": llms,
        "save_dir": config.chat.image.save_dir,
        "url_path": config.chat.image.url_path,
    }

    engine = args.engine.lower()
    logger.info(f"🎨 {engine}で画像生成を開始します: '{args.prompt}'")

    if engine == "fastsd":
        from src.img.fastsd import FastSD

        client = FastSD(**common_args, **vars(config.fastsd))
        _, save_path = client._generate_image(prompt=args.prompt)
    elif engine == "gemini_image":
        from src.img.gemini_img import GeminiImg

        img_generator = GeminiImg(**common_args, **vars(config.gemini_image))
        _, save_path = img_generator._generate_image(prompt=args.prompt)
    elif engine == "mock":
        from src.img.base import ImageGenerator

        img_generator = ImageGenerator(**common_args)
        _, save_path = img_generator.generate_image([])
    else:
        logger.error(f"❌ [エラー] 未知の画像生成エンジン: {engine}")
        sys.exit(1)

    open_image(save_path)
    logger.info("✅ 画像生成が完了しました。")


# --- コマンドライン引数の解析とディスパッチ ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Chat プロジェクトのタスクランナー")
    subparsers = parser.add_subparsers(
        dest="task", required=True, help="実行するタスク"
    )

    # chat タスク
    parser_chat = subparsers.add_parser("chat", help="AIとのCLIチャットを開始します")
    parser_chat.set_defaults(func=lambda args: task_chat())

    # server タスク
    parser_server = subparsers.add_parser("server", help="Web UIサーバーを起動します")
    parser_server.set_defaults(func=lambda args: task_server())

    # tts-list タスク
    parser_tts_list = subparsers.add_parser(
        "tts-list", help="指定されたTTSエンジンの話者一覧を表示します"
    )
    parser_tts_list.add_argument(
        "engine",
        type=str,
        choices=["voicevox", "coeiroink", "aivisspeech"],
        help="話者一覧を表示するTTSエンジン (voicevox, coeiroink, aivisspeech)",
    )
    parser_tts_list.set_defaults(func=task_tts_list)

    # tts-test タスク
    parser_tts = subparsers.add_parser(
        "tts-test", help="指定されたTTSエンジンで音声合成をテストします"
    )
    parser_tts.add_argument(
        "engine",
        type=str,
        choices=["voicevox", "coeiroink", "aivisspeech"],
        help="テストするTTSエンジン (voicevox, coeiroink, aivisspeech)",
    )
    parser_tts.add_argument("text", type=str, help="合成するテキスト")
    parser_tts.set_defaults(func=task_tts)

    # asr-test タスク
    parser_asr = subparsers.add_parser(
        "asr-test", help="指定されたASRエンジンで音声認識をテストします"
    )
    parser_asr.add_argument(
        "engine",
        type=str,
        choices=["whisper", "vosk", "gemini"],
        help="テストするASRエンジン (whisper, vosk, gemini)",
    )
    parser_asr.add_argument("--loop", action="store_true", help="連続して認識を行う")
    parser_asr.set_defaults(func=task_asr)

    # fastsd-status タスク
    parser_fastsd_status = subparsers.add_parser(
        "fastsd-status", help="FastSDのシステム情報とモデル一覧を表示します"
    )
    parser_fastsd_status.set_defaults(func=task_fastsd_status)

    # image-gen タスク
    parser_img_gen = subparsers.add_parser("img-gen", help="AIで画像を生成します")
    parser_img_gen.add_argument(
        "engine",
        type=str,
        choices=["fastsd", "gemini_image", "mock"],
        help="使用する画像生成エンジン (fastsd, gemini_image, mock)",
    )
    parser_img_gen.add_argument("prompt", type=str, help="画像生成のためのプロンプト")
    parser_img_gen.set_defaults(func=task_img_gen)

    # 引数を解析して対応する関数を実行
    args = parser.parse_args()
    args.func(args)
