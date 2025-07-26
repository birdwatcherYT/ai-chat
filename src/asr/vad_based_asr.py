import os
import tempfile
import wave
from collections import deque

import sounddevice as sd
import webrtcvad

from .base import SpeechToText


class VadBasedASR(SpeechToText):
    def __init__(
        self,
        # vad設定
        sensitivity: int,
        hangover_threshold: int,
        pre_buffer_frames: int,
    ):
        """音声認識の基底クラスを初期化

        Args:
            sensitivity (int): VADの感度設定（0〜3、3が最も厳しい）
            hangover_threshold (int): 発話終了と判断するまでの無音フレーム数
            pre_buffer_frames (int): 発話開始前に保持しておくフレーム数
        """
        super().__init__()
        # VADの初期化
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(sensitivity)
        self.buffer = []
        self.is_speaking = False
        self.hangover_threshold = hangover_threshold
        self.sample_rate = 16000
        self.q = deque()
        self.pre_buffer = deque(maxlen=pre_buffer_frames)

    def _callback(self, indata, frames, time, status):
        """マイクからの入力コールバック関数"""
        if status:
            print(status)
        self.q.append(bytes(indata))

    def _save_wav(self, audio_data: bytes) -> str:
        """音声データを一時WAVファイルとして保存"""
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        return path

    def process_audio(self, audio_bytes: bytes) -> str:
        """音声データを処理して文字列を返す（サブクラスで実装）"""
        raise NotImplementedError

    def audio_input(self) -> str:
        """マイク入力から音声を認識し、テキストを返す"""
        silence_counter = 0

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=320,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            try:
                while True:
                    try:
                        data = self.q.popleft() if self.q else None
                        if not data:
                            continue

                        # まずは常にプリバッファに追加
                        self.pre_buffer.append(data)

                        is_speech = self.vad.is_speech(
                            data, sample_rate=self.sample_rate
                        )

                        if is_speech:
                            if not self.is_speaking:
                                self.is_speaking = True
                                # 発話開始時にプリバッファの内容をコピー
                                self.buffer.extend(self.pre_buffer)
                            self.buffer.append(data)
                            silence_counter = 0
                        elif self.is_speaking:
                            silence_counter += 1
                            self.buffer.append(data)

                            if silence_counter >= self.hangover_threshold:
                                audio_bytes = b"".join(self.buffer)
                                result = self.process_audio(audio_bytes)
                                self.buffer = []
                                self.is_speaking = False
                                silence_counter = 0
                                if result:
                                    return result

                    except IndexError:
                        continue
            except KeyboardInterrupt:
                return ""
