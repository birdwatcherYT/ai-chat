import json
from invoke.config import Config


class LLMConfig:
    def __init__(self, cfg: Config):
        self.user_name = cfg.chat.user.name
        self.ai_names = {f"ai{i}_name": ai["name"] for i, ai in enumerate(cfg.chat.ai)}
        self.char_names = list(self.ai_names.values()) + [self.user_name]

        # キャラクター設定のプロンプト生成
        self.user_character = cfg.chat.user.character.format(
            user_name=self.user_name, **self.ai_names
        )
        self.chara_prompt = "\n".join(
            [
                f"{ai['name']}\n{ai['character'].format(user_name=self.user_name, **self.ai_names)}"
                for ai in cfg.chat.ai
            ]
        )

        self.gemini = cfg.gemini
        self.image = cfg.image
        self.ollama = cfg.ollama
        self.openrouter = cfg.openrouter
        self.model_engine = cfg.chat.model_engine

    def format(self, text: str) -> str:
        return text.format(user_name=self.user_name, **self.ai_names)


def history_to_text(history):
    return (
        "[\n" + ",\n".join([json.dumps(h, ensure_ascii=False) for h in history]) + "\n]"
    )
