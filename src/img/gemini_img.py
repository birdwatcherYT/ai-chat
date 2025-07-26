import os
import time
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

from ..lmm.common import history_to_text
from ..lmm.llm import LLMs
from ..logger import get_logger
from .base import ImageGenerator

logger = get_logger(__name__, level="INFO")


class GeminiImg(ImageGenerator):
    def __init__(self, llms: LLMs, model: str):
        super().__init__(llms)
        self.client = genai.Client()
        self.model = model

    def _generate_image(
        self, prompt: str, image: Image.Image = None
    ) -> tuple[str, str] | None:
        """画像を生成し、(URL, ローカルパス) のタプルを返す"""
        logger.debug(f"🎨 [IMAGE] 画像生成プロンプト: {prompt[:100]}...")
        err = ""
        try:
            if self.client is None:
                raise ValueError("Client is not initialized in mock mode.")

            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image] if image else prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"]
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    new_image = Image.open(BytesIO(part.inline_data.data))
                    self.last_image = new_image
                    filename = f"{int(time.time())}.png"
                    save_path = os.path.join(self.save_dir, filename)
                    image_url = f"{self.url_path}/{filename}"
                    new_image.save(save_path)
                    logger.debug(f"🖼️ [IMAGE] 画像を保存しました: {save_path}")
                    return image_url, save_path
        except Exception as e:
            err = str(e)
        logger.error(f"❌ [IMAGE] 画像生成に失敗しました: {err}")
        return None, None

    def generate_image(
        self, history: list[dict[str, str]], edit: bool = False
    ) -> tuple[str, str] | None:
        """状況を判断し、画像を生成して(URL, ローカルパス)のタプルを返す"""
        situation = self.situation_chain.invoke({"messages": history_to_text(history)})
        logger.debug("-" * 20, "状況説明", "-" * 20, "\n", situation, "\n", "-" * 20)

        if edit and self.last_image:
            prompt = f"現在の画像から次の状況を表すアニメ風画像を生成してください:\n{situation}"
            return self._generate_image(prompt, self.last_image)
        else:
            prompt = f"次の状況を表すアニメ風画像を生成してください:\n{situation}"
            return self._generate_image(prompt)
