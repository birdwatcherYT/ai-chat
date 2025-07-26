import numpy as np
from faster_whisper import WhisperModel

from .vad_based_asr import VadBasedASR


class WhisperASR(VadBasedASR):
    def __init__(
        self,
        model_name: str,
        compute_type: str,
        vad_filter: bool,
        # vad設定
        sensitivity: int,
        hangover_threshold: int,
        pre_buffer_frames: int,
    ):
        """音声認識モデルを初期化

        Args:
            model_name (str): Faster-Whisperモデル名（例: "small", "turbo"）
            compute_type (str): 計算タイプ
            vad_filter (bool): VADフィルタを使用するかどうか
            sensitivity (int): VADの感度設定（0〜3、3が最も厳しい）
            hangover_threshold (int): 発話終了と判断するまでの無音フレーム数
            pre_buffer_frames (int): 発話開始前に保持しておくフレーム数
        """
        super().__init__(sensitivity, hangover_threshold, pre_buffer_frames)
        # Whisperモデルのロード
        self.model = WhisperModel(model_name, compute_type=compute_type)
        self.vad_filter = vad_filter

    def process_audio(self, audio_bytes: bytes) -> str:
        """音声データを処理して文字列を返す"""
        # 音声データをfloat32のNumPyアレイに変換
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # Whisperで音声を認識
        segments, _ = self.model.transcribe(
            audio, language="ja", vad_filter=self.vad_filter
        )
        # 認識結果をテキストとして結合
        text = " ".join([segment.text for segment in segments]).strip()
        return text
