import argparse
import asyncio
import sys

import yaml
from invoke.config import Config

from src.logger import get_logger

logger = get_logger(__name__, level="INFO")


# --- 設定ファイルを安全に読み込む ---
def load_config():
    """config.yamlをUTF-8で読み込み、invoke.Configオブジェクトを返す"""
    try:
        with open("config.yaml", encoding="utf-8") as f:
            # invoke.Configを使うことで、既存コードへの影響を最小限にする
            return Config(yaml.safe_load(f))
    except FileNotFoundError:
        logger.error("❌ [エラー] 設定ファイル 'config.yaml' が見つかりません。")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ [エラー] config.yamlの読み込みまたは解析に失敗しました: {e}")
        sys.exit(1)


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
    logger.info("🚀 Webサーバーを http://0.0.0.0:8000 で起動します。")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def task_vv_list():
    """VOICEVOXの話者一覧を表示します"""
    from src.tts.voicevox import VoiceVox

    VoiceVox().print_speakers()


def task_vv_test(args):
    """VOICEVOXで音声合成をテストします"""
    import sounddevice as sd

    from src.tts.voicevox import VoiceVox

    config = load_config()
    data, sr = VoiceVox().synthesize(args.text, **config.voicevox)
    sd.play(data, sr)
    sd.wait()


# 他のテストタスクも同様に追加...
def task_ci_list():
    from src.tts.coeiroink import CoeiroInk

    CoeiroInk().print_speakers()


def task_ci_test(args):
    import sounddevice as sd

    from src.tts.coeiroink import CoeiroInk

    config = load_config()
    data, sr = CoeiroInk().synthesize(args.text, **config.coeiroink)
    sd.play(data, sr)
    sd.wait()


def task_whisper_test(args):
    from src.asr.whisper_asr import WhisperASR

    config = load_config()
    print("Whisperの読み取り開始")
    asr = WhisperASR(**config.whisper, **config.webrtcvad)
    if args.loop:
        while True:
            print(asr.audio_input())
    else:
        print(asr.audio_input())


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

    # vv-list タスク
    parser_vv_list = subparsers.add_parser(
        "vv-list", help="VOICEVOXの話者一覧を表示します"
    )
    parser_vv_list.set_defaults(func=lambda args: task_vv_list())

    # vv-test タスク
    parser_vv_test = subparsers.add_parser(
        "vv-test", help="VOICEVOXの音声合成をテストします"
    )
    parser_vv_test.add_argument("text", type=str, help="合成するテキスト")
    parser_vv_test.set_defaults(func=task_vv_test)

    # ci-list タスク
    parser_ci_list = subparsers.add_parser(
        "ci-list", help="COEIROINKの話者一覧を表示します"
    )
    parser_ci_list.set_defaults(func=lambda args: task_ci_list())

    # ci-test タスク
    parser_ci_test = subparsers.add_parser(
        "ci-test", help="COEIROINKの音声合成をテストします"
    )
    parser_ci_test.add_argument("text", type=str, help="合成するテキスト")
    parser_ci_test.set_defaults(func=task_ci_test)

    # whisper-test タスク
    parser_whisper_test = subparsers.add_parser(
        "whisper-test", help="Whisperの音声認識をテストします"
    )
    parser_whisper_test.add_argument(
        "--loop", action="store_true", help="連続して認識を行う"
    )
    parser_whisper_test.set_defaults(func=task_whisper_test)

    # 引数を解析して対応する関数を実行
    args = parser.parse_args()
    args.func(args)
