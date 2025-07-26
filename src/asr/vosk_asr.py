import json

import vosk
import sounddevice as sd

from .base import SpeechToText


class VoskASR(SpeechToText):
    def __init__(self, model_dir: str):
        """音声認識モデルを初期化

        Args:
            model_dir (str): Voskモデルのディレクトリパス
        """
        super().__init__()
        self.model = vosk.Model(model_dir)

    def process_audio(self, audio_bytes: bytes) -> str:
        """
        音声データ(bytes)のチャンクを受け取り、テキストに変換して返す。
        サーバーサイドでの処理のために追加。
        """
        rec = vosk.KaldiRecognizer(self.model, 16000)
        
        # 受け取った音声データ全体を処理
        rec.AcceptWaveform(audio_bytes)
        
        # 最終的な認識結果を取得
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").replace(" ", "")
        
        return text


    def audio_input(self) -> str:
        """
        マイク入力から音声を認識し、テキストを返す。
        """
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            rec = vosk.KaldiRecognizer(self.model, 16000)
            while True:
                data = self.q.get()
                if not self.running:
                    continue
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").replace(" ", "")
                    if text:
                        return text