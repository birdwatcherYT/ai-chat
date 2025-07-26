import os
import time
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

from ..logger import get_logger
from .common import LLMConfig, history_to_text
from .llm import LLMs

logger = get_logger(__name__, level="INFO")


class ImageGenerator:
    def __init__(self, llmcfg: LLMConfig, llms: LLMs, save_dir: str, url_path: str):
        self.llmcfg = llmcfg
        self.save_dir = save_dir
        self.url_path = url_path
        # mockがTrueでない場合のみクライアントを初期化
        if not self.llmcfg.image.mock:
            self.client = genai.Client()
        else:
            self.client = None
        self.situation_chain = llms.get_situation_chain()
        self.last_image: Image.Image = None

    def _generate_image(
        self, prompt: str, image: Image.Image = None
    ) -> tuple[str, str] | None:
        """画像を生成し、(URL, ローカルパス) のタプルを返す"""
        logger.debug(f"🎨 [IMAGE] 画像生成プロンプト: {prompt[:100]}...")
        try:
            if self.client is None:
                raise ValueError("Client is not initialized in mock mode.")

            response = self.client.models.generate_content(
                model=self.llmcfg.image.model,
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
            logger.error(f"❌ [IMAGE] 画像生成に失敗しました: {e}")
        return None, None

    def generate_image(
        self, history: list[dict[str, str]], edit: bool = False
    ) -> tuple[str, str] | None:
        """状況を判断し、画像を生成して(URL, ローカルパス)のタプルを返す"""
        if self.llmcfg.image.mock:
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

        situation = self.situation_chain.invoke({"messages": history_to_text(history)})
        logger.debug("-" * 20, "状況説明", "-" * 20, "\n", situation, "\n", "-" * 20)

        if edit and self.last_image:
            prompt = f"現在の画像から次の状況を表すアニメ風画像を生成してください:\n{situation}"
            return self._generate_image(prompt, self.last_image)
        else:
            prompt = f"次の状況を表すアニメ風画像を生成してください:\n{situation}"
            return self._generate_image(prompt)
