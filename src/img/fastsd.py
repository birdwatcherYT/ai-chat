import base64
import io
import logging
import os
import time
from io import BytesIO
from types import SimpleNamespace
from typing import Literal

import requests
from PIL import Image

from ..llm.common import history_to_text
from ..llm.llm import LLMs
from ..logger import get_logger
from ..utils import simple_namespace_to_dict
from .base import ImageGenerator

logger = get_logger(__name__, level=logging.INFO)


class FastSD(ImageGenerator):
    """Stable Diffusion APIクライアント"""

    def __init__(
        self,
        save_dir: str,
        url_path: str,
        llms: LLMs,
        base_url: str,
        settings: SimpleNamespace,
    ):
        super().__init__(llms, save_dir, url_path)
        self.server_url = base_url
        self.base_settings = settings

    def _generate_image(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        diffusion_task: Literal["text_to_image", "image_to_image"] = "text_to_image",
        init_image: str = None,
    ) -> tuple[str, str] | None:
        """画像生成リクエストを送信"""
        url = f"{self.server_url}/api/generate"

        # DeepCopyと変換を同時に行う
        payload = simple_namespace_to_dict(self.base_settings)
        payload["prompt"] = prompt
        payload["negative_prompt"] = negative_prompt
        payload["diffusion_task"] = diffusion_task
        if diffusion_task == "text_to_image":
            del payload["strength"]
        else:
            payload["init_image"] = init_image
        payload["number_of_images"] = 1

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
            # image to image
            buffered = io.BytesIO()
            self.last_image.save(buffered, format="JPEG")
            img_byte = buffered.getvalue()
            encoded_image = base64.b64encode(img_byte).decode("utf-8")

            return self._generate_image(
                prompt=prompt,
                diffusion_task="image_to_image",
                init_image=encoded_image,
            )
        return self._generate_image(prompt=prompt)
