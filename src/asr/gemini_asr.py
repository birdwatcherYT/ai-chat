import os
from google import genai
from .vad_based_asr import VadBasedASR


class GeminiASR(VadBasedASR):
    def __init__(
        self,
        model_name: str,
        # vad設定
        sensitivity: int,
        hangover_threshold: int,
        pre_buffer_frames: int,
    ):
        """音声認識モデルを初期化

        Args:
            model_name (str): Geminiモデル名
            sensitivity (int): VADの感度設定（0〜3、3が最も厳しい）
            hangover_threshold (int): 発話終了と判断するまでの無音フレーム数
            pre_buffer_frames (int): 発話開始前に保持しておくフレーム数
        """
        super().__init__(sensitivity, hangover_threshold, pre_buffer_frames)
        self.model_name = model_name
        self.client = genai.Client()

    def process_audio(self, audio_bytes: bytes) -> str:
        """音声データを処理して文字列を返す"""
        wav_path = self._save_wav(audio_bytes)

        try:
            audio_file = self.client.files.upload(file=wav_path)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    "この音声を書き起こしてください。ノイズは無視してください。",
                    audio_file,
                ],
            )
            return response.text
        finally:
            os.unlink(wav_path)
