import json
from types import SimpleNamespace


class LLMConfig:
    def __init__(self, cfg: SimpleNamespace):
        self.user_name = cfg.chat.user.name
        self.ai_names = {f"ai{i}_name": ai.name for i, ai in enumerate(cfg.chat.ai)}
        self.char_names = list(self.ai_names.values()) + [self.user_name]
        self.user_prompt = f"## {self.user_name}\n{cfg.chat.user.character.format(user_name=self.user_name, **self.ai_names)}"
        self.chara_prompt = "\n".join(
            [
                f"## {ai.name}\n{ai.character.format(user_name=self.user_name, **self.ai_names)}"
                for ai in cfg.chat.ai
            ]
        )
        self.gemini = cfg.gemini
        self.ollama = cfg.ollama
        self.openrouter = cfg.openrouter
        self.llm_engine = cfg.chat.llm_engine

    def format(self, text: str) -> str:
        return text.format(user_name=self.user_name, **self.ai_names)


# 新しいhistory形式を解釈する
def history_to_text(history: list[dict]) -> str:
    """
    新しい形式の会話履歴を、画像を含まないテキストのみのJSON文字列に変換する。
    画像は "(画像添付)" というテキストで表現される。
    """
    text_history = []
    for item in history:
        if item["type"] == "text":
            # テキストはそのまま追加
            text_history.append({"name": item["name"], "content": item["content"]})
        elif item["type"] == "image":
            # 画像はプレースホルダーに変換して追加
            text_history.append({"name": item["name"], "content": "(画像添付)"})

    return (
        "[\n"
        + ",\n".join([json.dumps(h, ensure_ascii=False) for h in text_history])
        + "\n]"
    )
