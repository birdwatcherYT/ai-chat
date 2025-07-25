from dotenv import load_dotenv

load_dotenv()

# https://ai.google.dev/gemini-api/docs/image-generation?hl=ja
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client()


def generate_image(prompt: str):
    """Gemini APIを使用して画像を生成する関数"""
    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            image.save("gemini-native-image.png")
            image.show()
