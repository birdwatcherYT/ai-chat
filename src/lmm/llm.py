import json
import os
from typing import Literal

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .common import LLMConfig, history_to_text


class LLMs:
    def __init__(self, llmcfg: LLMConfig):
        self.llmcfg = llmcfg
        # LLMの選択と初期化
        if self.llmcfg.model_engine == "ollama":
            self.llm = ChatOllama(**llmcfg.ollama)
        elif self.llmcfg.model_engine == "gemini":
            self.llm = ChatGoogleGenerativeAI(**llmcfg.gemini)
        elif self.llmcfg.model_engine == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            self.llm = ChatOpenAI(
                model=llmcfg.openrouter.model,
                openai_api_key=api_key,
                openai_api_base=llmcfg.openrouter.base_url,
            )

        self.speaker_prompt_template = self.get_speaker_prompt_template()

    def get_speaker_prompt_template(self) -> PromptTemplate:
        """話者決定プロンプトテンプレートを取得"""
        # PromptTemplateの定義（from_templateで記述、input_variables/partial_variablesは省略）
        prompt = PromptTemplate.from_template(
            """次に発話するべき話者名を出力してください。
    # 出力候補
    ```json
    {char_names}
    ```
    # キャラクター情報
    {user_name}
    {user_character}
    {chara_prompt}
    # 会話履歴
    ```json
    {messages}
    ```
    """,
            partial_variables={
                "user_name": self.llmcfg.user_name,
                "user_character": self.llmcfg.user_character,
                "chara_prompt": self.llmcfg.chara_prompt,
                "char_names": json.dumps(self.llmcfg.char_names, ensure_ascii=False),
            },
        )
        return prompt

    def get_next_speaker(
        self, history: list[dict[str, str]], except_names: list[str] = None
    ) -> str:
        if except_names is None:
            except_names = []
        candidates = [c for c in self.llmcfg.char_names if c not in except_names]
        # Pydanticモデル定義: candidatesから動的にLiteral型スキーマを生成
        SpeakerSchema = type(
            "SpeakerSchema",
            (BaseModel,),
            {
                "__annotations__": {
                    "speaker": Literal[tuple(candidates)]
                },  # ここで型アノテーションを定義
                "speaker": ...,  # EllipsisはPydanticの必須フィールドを表す
            },
        )

        # 話者決定プロンプトと発話生成プロンプトのチェーンを作成
        speaker_chain = self.speaker_prompt_template | self.llm.with_structured_output(
            SpeakerSchema
        )
        result: SpeakerSchema = speaker_chain.invoke(
            {
                "char_names": json.dumps(candidates, ensure_ascii=False),
                "messages": history_to_text(history),
            }
        )
        return result.speaker

    def get_utter_chain(self):
        utter_prompt_template = PromptTemplate.from_template(
            """**会話履歴**に続く{speaker}の次の発言を生成してください。発話内容だけを出力してください。
    # キャラクター情報
    {user_name}
    {user_character}
    {chara_prompt}
    # 会話履歴
    ```json
    {messages}
    ```
    """,
            partial_variables={
                "user_name": self.llmcfg.user_name,
                "user_character": self.llmcfg.user_character,
                "chara_prompt": self.llmcfg.chara_prompt,
            },
        )
        return utter_prompt_template | self.llm
