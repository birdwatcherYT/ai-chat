import base64
import io
import os
import time
from dataclasses import dataclass
from io import BytesIO

import requests
from PIL import Image

from ..llm.common import history_to_text
from ..llm.llm import LLMs
from ..logger import get_logger
from .base import ImageGenerator

logger = get_logger(__name__, level="INFO")


@dataclass
class GenerationSettings:
    """画像生成の設定パラメータ"""

    prompt: str
    negative_prompt: str = ""
    seed: int = -1
    guidance_scale: float = 1.0
    image_height: int = 512
    image_width: int = 512
    inference_steps: int = 4
    number_of_images: int = 1
    use_openvino: bool = False
    use_tiny_auto_encoder: bool = True
    use_lcm_lora: bool = True
    diffusion_task: str = "text_to_image"
    init_image: str | None = None
    strength: float | None = None


class FastSD(ImageGenerator):
    """Stable Diffusion APIクライアント"""

    def __init__(self, llms: LLMs, base_url: str):
        super().__init__(llms)
        self.server_url = base_url

    def _generate_image(self, settings: GenerationSettings) -> tuple[str, str] | None:
        """画像生成リクエストを送信"""
        url = f"{self.server_url}/api/generate"

        # dataclassを辞書に変換
        payload = {k: v for k, v in settings.__dict__.items() if v is not None}

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()

        # レスポンス情報を表示
        logger.debug(f"生成時間: {result['latency']}秒")
        logger.debug(f"生成された画像数: {len(result['images'])}")

        # base64画像をデコードして保存
        new_image = Image.open(BytesIO(base64.b64decode(result["images"][0])))
        self.last_image = new_image
        filename = f"{int(time.time())}.png"
        save_path = os.path.join(self.save_dir, filename)
        image_url = f"{self.url_path}/{filename}"
        new_image.save(save_path)
        logger.debug(f"画像を保存しました: {filename}")
        return image_url, save_path

    def get_system_info(self) -> dict:
        """システム情報を取得"""
        url = f"{self.server_url}/api/info"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def get_available_models(self) -> dict:
        """利用可能なモデル一覧を取得"""
        url = f"{self.server_url}/api/models"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def generate_image(
        self, history: list[dict[str, str]], edit: bool = False
    ) -> tuple[str, str] | None:
        """状況を判断し、画像を生成して(URL, ローカルパス)のタプルを返す"""
        situation = self.situation_chain.invoke({"messages": history_to_text(history)})
        logger.debug("-" * 20, "状況説明", "-" * 20, "\n", situation, "\n", "-" * 20)

        prompt = f"anime style, {situation}"
        if edit and self.last_image:
            buffered = io.BytesIO()
            self.last_image.save(buffered, format="JPEG")
            img_byte = buffered.getvalue()
            encoded_image = base64.b64encode(img_byte).decode("utf-8")

            settings = GenerationSettings(
                prompt=prompt,
                diffusion_task="image_to_image",
                init_image=encoded_image,
                strength=0.8,
            )
        else:
            settings = GenerationSettings(
                prompt=prompt,
                diffusion_task="text_to_image",
            )
        return self._generate_image(settings)
