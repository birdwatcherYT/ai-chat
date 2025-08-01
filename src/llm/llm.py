import json
import os
import random
from types import SimpleNamespace
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .common import LLMConfig, history_to_text


class LLMs:
    def __init__(self, cfg: SimpleNamespace, llmcfg: LLMConfig):
        self.cfg = cfg
        self.llmcfg = llmcfg

        # 機能ごとにLLMを初期化
        self.utterance_llm = self._init_llm_from_config("utterance")
        self.turn_control_llm = self._init_llm_from_config("turn_control")
        self.situation_llm = self._init_llm_from_config("situation")

        self.speaker_prompt_template = self.get_speaker_prompt_template()

    def _init_llm_from_config(self, task_name: str):
        """設定から特定のタスク用のLLMを初期化する"""
        task_config = getattr(self.cfg.chat.llm, task_name, None)

        # タスクごとの設定がない場合は、発話生成(utterance)の設定をフォールバックとして使用
        if not task_config or not hasattr(task_config, "engine"):
            task_config = self.cfg.chat.llm.utterance

        engine = task_config.engine
        base_params = getattr(self.llmcfg, engine, None)
        if base_params is None:
            raise ValueError(f"LLM engine '{engine}' configuration not found.")

        params = vars(base_params).copy()
        # タスク固有の設定で上書き
        if hasattr(task_config, "config"):
            params.update(vars(task_config.config))

        if engine == "ollama":
            return ChatOllama(**params)
        elif engine == "gemini":
            return ChatGoogleGenerativeAI(**params)
        elif engine == "openrouter":
            params["openai_api_key"] = os.getenv("OPENROUTER_API_KEY")
            return ChatOpenAI(**params)

        raise ValueError(
            f"Unknown or unsupported LLM engine specified for task '{task_name}': {engine}"
        )

    def get_speaker_prompt_template(self) -> PromptTemplate:
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
        speaker_chain = (
            self.speaker_prompt_template
            | self.turn_control_llm.with_structured_output(SpeakerSchema)
        )
        result: SpeakerSchema = speaker_chain.invoke(
            {
                "char_names": json.dumps(candidates, ensure_ascii=False),
                "messages": history_to_text(history),
            }
        )
        return result.speaker

    def get_situation_chain(self):
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
                "user_prompt": self.llmcfg.user_prompt,
                "chara_prompt": self.llmcfg.chara_prompt,
            },
        )
        return prompt | self.situation_llm | StrOutputParser()

    def get_utter_chain(self, history: list, webcam_capture: str | None = None):
        """
        発言生成のためのチェーンを取得する。
        historyと、オプショナルなWebカメラキャプチャからプロンプトを構築する。
        """
        instruction = """**キャラクター情報**を踏まえ、**会話履歴**に続く発言を生成してください。発話内容だけを出力してください。

# キャラクター情報
{user_prompt}
{chara_prompt}

# 会話履歴
"""
        multimodal_history = [{"type": "text", "text": instruction}]

        for item in history:
            if item["type"] == "text":
                escaped_content = json.dumps(item["content"], ensure_ascii=False).strip(
                    '"'
                )
                multimodal_history.append(
                    {"type": "text", "text": f"{item['name']}: {escaped_content}"}
                )
            elif item["type"] == "image":
                multimodal_history.append(
                    {"type": "image_url", "image_url": {"url": item["content"]}}
                )

        multimodal_history.append({"type": "text", "text": "\n{speaker}: "})

        # webcam_captureが存在する場合、プロンプトに追加
        if webcam_capture:
            multimodal_history.append(
                {"type": "text", "text": "\n\n# 現在のWebカメラ映像"}
            )
            multimodal_history.append(
                {"type": "image_url", "image_url": {"url": webcam_capture}}
            )

        prompt_template = ChatPromptTemplate.from_messages(
            [("human", multimodal_history)]
        )

        partial_prompt = prompt_template.partial(
            user_prompt=self.llmcfg.user_prompt,
            chara_prompt=self.llmcfg.chara_prompt,
        )

        return partial_prompt | self.utterance_llm | StrOutputParser()
