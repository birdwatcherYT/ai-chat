from pydantic import BaseModel
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json


class LLMs:
    def __init__(self, cfg, user_name: str, ai_names: dict[str, str]):
        self.user_name = user_name
        self.ai_names = ai_names
        self.char_names = list(ai_names.values()) + [user_name]

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        # キャラクター設定のプロンプト生成
        self.user_prompt = cfg.chat.user.character.format(
            user_name=user_name, **ai_names
        )
        self.chara_prompt = "\n".join(
            [
                f"{ai['name']}\n{ai['character'].format(user_name=user_name, **ai_names)}"
                for ai in cfg.chat.ai
            ]
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
    ```
    {messages}
    ```
    """,
            partial_variables={
                "user_name": self.user_name,
                "user_character": self.user_prompt,
                "chara_prompt": self.chara_prompt,
                "char_names": json.dumps(self.char_names, ensure_ascii=False),
            },
        )
        return prompt

    def get_next_speaker(
        self, history: list[dict[str, str]], except_names: list[str] = []
    ) -> str:
        candidates = [c for c in self.char_names if c not in except_names]
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
        result = speaker_chain.invoke(
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
    ```
    {messages}
    ```
    """,
            partial_variables={
                "user_name": self.user_name,
                "user_character": self.user_prompt,
                "chara_prompt": self.chara_prompt,
            },
        )
        return utter_prompt_template | self.llm


def history_to_text(history):
    # TODO: json str化
    return "\n".join([f"{h['name']}: {h['content']}" for h in history])
