import logging
import os
import time

from PIL import Image

from ..llm.llm import LLMs
from ..logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class ImageGenerator:
    """画像生成のベースクラス兼モッククラス"""

    def __init__(
        self,
        llms: LLMs,
        save_dir: str,
        url_path: str,
    ):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.url_path = url_path
        self.last_image: Image.Image = None
        self.situation_chain = llms.get_situation_chain()

    def generate_image(
        self, history: list[dict[str, str]], edit: bool = False
    ) -> tuple[str, str] | None:
        """状況を判断し、画像を生成して(URL, ローカルパス)のタプルを返す"""
        logger.debug("🖼️ [MOCK-IMAGE] Returning a local mock image.")
        time.sleep(3)  # リアルな待機時間をシミュレート
        mock_filename = "mock.png"
        image_path = os.path.join(self.save_dir, mock_filename)
        image_url = f"{self.url_path}/{mock_filename}"

        # ユーザーへの案内メッセージ
        if not os.path.exists(image_path):
            logger.warning(
                f"⚠️  [MOCK-IMAGE] Mock image file not found. Please place a file named '{mock_filename}' in the '{self.save_dir}' directory."
            )

        return image_url, image_path
