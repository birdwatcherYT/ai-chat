import json
import logging
import sys
from types import SimpleNamespace

import yaml

# 音声認識 (ASR)
from .asr.base import SpeechToText

# 画像生成
from .img.base import ImageGenerator
from .img.fastsd import FastSD
from .img.gemini_img import GeminiImg

# LLM
from .llm.common import LLMConfig
from .llm.llm import LLMs
from .logger import get_logger

# 音声合成 (TTS)
from .tts.aivisspeech import AivisSpeech
from .tts.coeiroink import CoeiroInk
from .tts.voicevox import VoiceVox
from .turn_manager import TurnManager

logger = get_logger(__name__, level=logging.INFO)


def load_config() -> SimpleNamespace:
    """config.yamlをUTF-8で読み込み、属性アクセス可能なオブジェクトを返す"""
    try:
        with open("config.yaml", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
            return json.loads(
                json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d)
            )
    except FileNotFoundError:
        logger.error("❌ [エラー] 設定ファイル 'config.yaml' が見つかりません。")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ [エラー] config.yamlの読み込みまたは解析に失敗しました: {e}")
        sys.exit(1)


class AppContext:
    """アプリケーション全体の依存関係と状態を管理するクラス"""

    def __init__(self, cfg: SimpleNamespace):
        self.cfg = cfg
        self.llmcfg = LLMConfig(self.cfg)
        self.llms = LLMs(self.cfg, self.llmcfg)
        self.turn_manager = TurnManager(cfg, self.llms)
        self.img_generator = self._init_image_generator()
        self.asr_engine = self._init_asr_engine()
        self.tts_engines = {
            "voicevox": VoiceVox(),
            "coeiroink": CoeiroInk(),
            "aivisspeech": AivisSpeech(),
        }
        self.ai_config = {ai.name: ai.voice for ai in self.cfg.chat.ai}
        if self.cfg.chat.user.input == "ai":
            self.ai_config[self.llmcfg.user_name] = self.cfg.chat.user.voice

        # 初期履歴を新しいデータ構造に変換
        self.initial_history = [
            {
                "name": self.llmcfg.format(item.name),
                "type": "text",
                "content": self.llmcfg.format(item.content),
            }
            for item in self.cfg.chat.initial_message
        ]
        self.initial_turn = self.llmcfg.format(self.cfg.chat.initial_turn)

    def _init_image_generator(self) -> ImageGenerator:
        """設定に基づいて画像生成エンジンを初期化する"""
        image_model = self.cfg.chat.image.model
        common_args = {
            "llms": self.llms,
            "save_dir": self.cfg.chat.image.save_dir,
            "url_path": self.cfg.chat.image.url_path,
        }
        if image_model == "fastsd":
            return FastSD(**common_args, **vars(self.cfg.fastsd))
        if image_model == "gemini_image":
            return GeminiImg(**common_args, **vars(self.cfg.gemini_image))
        return ImageGenerator(**common_args)

    def _init_asr_engine(self) -> SpeechToText | None:
        """設定に基づいて音声認識エンジンを初期化する"""
        user_input_mode = self.cfg.chat.user.input
        asr_engine = getattr(self.cfg.chat.user, "asr_engine", None)

        if user_input_mode != "voice" or not asr_engine or asr_engine == "browser":
            return None

        if asr_engine == "vosk":
            from .asr.vosk_asr import VoskASR

            return VoskASR(**vars(self.cfg.vosk))
        if asr_engine == "whisper":
            from .asr.whisper_asr import WhisperASR

            return WhisperASR(**vars(self.cfg.whisper), **vars(self.cfg.webrtcvad))
        if asr_engine == "gemini_asr":
            from .asr.gemini_asr import GeminiASR

            return GeminiASR(**vars(self.cfg.gemini_asr), **vars(self.cfg.webrtcvad))

        logger.warning(f"⚠️ [ASR] 未知または指定されていないASRエンジン: {asr_engine}")
        return None
