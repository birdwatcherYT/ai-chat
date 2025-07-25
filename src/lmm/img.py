from dotenv import load_dotenv

load_dotenv()

# https://ai.google.dev/gemini-api/docs/image-generation?hl=ja
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from invoke.config import Config

import json
from pydantic import BaseModel
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from .common import history_to_text
from .common import LLMConfig


class ImageGenerator:
    def __init__(self, llmcfg: LLMConfig):
        self.llmcfg = llmcfg
        self.client = genai.Client()
        self.llm = ChatGoogleGenerativeAI(**llmcfg.gemini)

        self.situation_chain = self.get_situation_chain()

    def get_situation_chain(self):
        """会話履歴から状況を表すプロンプトを生成"""
        prompt = PromptTemplate.from_template(
            """**会話履歴**を元に今の状況を表す説明を出力してください。ただし、出力にキャラクター名を含めてはいけません。この出力は画像生成のためのプロンプトとして使用されます。
# キャラクター情報
{user_name}
{user_character}
{chara_prompt}

# 会話履歴
```
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

    def generate_image(self, history: list[dict[str, str]]):
        situation = self.situation_chain.invoke({"messages": history_to_text(history)})
        print("-" * 20, "状況説明", "-" * 20)
        print(situation)
        print("-" * 20)
        self._generate_image(f"次の状況を表すアニメ風画像を生成してください:\n{situation}")

    def _generate_image(self, prompt: str):
        """Gemini APIを使用して画像を生成する関数"""
        response = self.client.models.generate_content(
            model=self.llmcfg.image.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        print("-" * 20, "画像生成結果", "-" * 20)
        print(len(response.candidates))
        print("-" * 20)
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO((part.inline_data.data)))
                image.save("gemini-image.png")
                image.show()
