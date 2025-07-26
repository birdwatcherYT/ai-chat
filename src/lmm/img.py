import os
import time
from io import BytesIO

from google import genai
from google.genai import types
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image

from ..logger import get_logger
from .common import LLMConfig, history_to_text

logger = get_logger(__name__, level="INFO")


class ImageGenerator:
    def __init__(self, llmcfg: LLMConfig, save_dir: str, url_path: str):
        self.llmcfg = llmcfg
        self.save_dir = save_dir
        self.url_path = url_path
        # mockがTrueでない場合のみクライアントを初期化
        if not hasattr(self.llmcfg.image, "mock") or not self.llmcfg.image.mock:
            self.client = genai.Client()
        else:
            self.client = None
        # SimpleNamespaceをvars()で辞書に変換してから展開
        self.llm = ChatGoogleGenerativeAI(**vars(llmcfg.gemini))
        self.situation_chain = self.get_situation_chain()
        self.last_image: Image.Image = None

    def get_situation_chain(self):
        prompt = PromptTemplate.from_template(
            """**会話履歴**を元に今の状況を表す説明を出力してください。ただし、出力にキャラクター名を含めてはいけません。この出力は画像生成のためのプロンプトとして使用されます。
# キャラクター情報
{user_name}
{user_character}
{chara_prompt}
# 会話履歴```json
{messages}
```
""",
            partial_variables={
                "user_name": self.llmcfg.user_name,
                "user_character": self.llmcfg.user_character,
                "chara_prompt": self.llmcfg.chara_prompt,
            },
        )
        return prompt | self.llm | StrOutputParser()

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
        if hasattr(self.llmcfg.image, "mock") and self.llmcfg.image.mock:
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
