import json
import os
import random
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .common import LLMConfig, history_to_text


class LLMs:
    def __init__(self, llmcfg: LLMConfig):
        self.llmcfg = llmcfg
        if self.llmcfg.llm_engine == "ollama":
            self.llm = ChatOllama(**vars(llmcfg.ollama))
        elif self.llmcfg.llm_engine == "gemini":
            self.llm = ChatGoogleGenerativeAI(**vars(llmcfg.gemini))
        elif self.llmcfg.llm_engine == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            self.llm = ChatOpenAI(**vars(llmcfg.openrouter), openai_api_key=api_key)

        self.speaker_prompt_template = self.get_speaker_prompt_template()

    def get_speaker_prompt_template(self) -> PromptTemplate:
        # この関数はhistory_to_textを使うので、修正は不要
        prompt = PromptTemplate.from_template(
            """次に発話するべき話者名を出力してください。

# キャラクター情報
{user_prompt}
{chara_prompt}

# 会話履歴
```json
{messages}
```

# 出力候補
```json
{char_names}
```
""",
            partial_variables={
                "user_prompt": self.llmcfg.user_prompt,
                "chara_prompt": self.llmcfg.chara_prompt,
                "char_names": json.dumps(self.llmcfg.char_names, ensure_ascii=False),
            },
        )
        return prompt

    def get_next_speaker(
        self, history: list[dict[str, str]], except_names: list[str] = None
    ) -> str:
        # この関数もhistory_to_textを使うので、修正は不要
        if except_names is None:
            except_names = []
        candidates = [c for c in self.llmcfg.char_names if c not in except_names]
        if not candidates:
            fallback_candidates = [
                c for c in self.llmcfg.char_names if c != self.llmcfg.user_name
            ]
            if not fallback_candidates:
                return self.llmcfg.user_name
            return random.choice(fallback_candidates)

        SpeakerSchema = type(
            "SpeakerSchema",
            (BaseModel,),
            {
                "__annotations__": {"speaker": Literal[tuple(candidates)]},
                "speaker": ...,
            },
        )
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

    def get_situation_chain(self):
        # この関数もhistory_to_textを使うので、修正は不要
        prompt = PromptTemplate.from_template(
            """**会話履歴**を元に今の状況を表す説明を**英語**で出力してください。ただし、出力にキャラクター名を含めてはいけません。この出力は画像生成のためのプロンプトとして使用されます。

# キャラクター情報
{user_prompt}
{chara_prompt}

# 会話履歴
```json
{messages}
```
""",
            partial_variables={
                "user_character": self.llmcfg.user_prompt,
                "chara_prompt": self.llmcfg.chara_prompt,
            },
        )
        return prompt | self.llm | StrOutputParser()

    # get_utter_chain全体を新しいロジックに置き換え
    def get_utter_chain(self, history: list):
        """
        発言生成のためのチェーンを取得する。
        このチェーンは、渡されるhistoryを解釈し、動的にマルチモーダルプロンプトを構築する。
        """
        # 1. ベースとなる指示テキストのテンプレート
        instruction_template = """**キャラクター情報**を踏まえ、**会話履歴**に続く**{speaker}**の次の発言を生成してください。発話内容だけを出力してください。

# キャラクター情報
{user_prompt}
{chara_prompt}

# 会話履歴
"""

        # 2. メッセージ(history)を動的に構築する関数
        # 履歴をHumanMessageのcontent形式に変換
        multimodal_history = [{"type": "text", "text": instruction_template}]
        for item in history:
            if item["type"] == "text":
                escaped_content = json.dumps(item["content"], ensure_ascii=False).strip(
                    '"'
                )
                multimodal_history.append(
                    {
                        "type": "text",
                        "text": f"{item['name']}: {escaped_content}",
                    }
                )
            elif item["type"] == "image":
                multimodal_history.append(
                    {"type": "image_url", "image_url": {"url": item["content"]}}
                )
        # TODO: 構造化出力にして削除する
        multimodal_history.append({"type": "text", "text": "{speaker}: "})

        # 3. ChatPromptTemplateを関数から生成
        # NOTE: HumanMessageを使うと変数が展開されない
        prompt_template = ChatPromptTemplate.from_messages(
            [("human", multimodal_history)]
        )

        # 4. 部分変数を適用
        partial_prompt = prompt_template.partial(
            user_prompt=self.llmcfg.user_prompt,
            chara_prompt=self.llmcfg.chara_prompt,
        )

        return partial_prompt | self.llm | StrOutputParser()
