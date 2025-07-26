import logging
import os
import platform
import subprocess
from types import SimpleNamespace

from src.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def open_image(image_path: str):
    system = platform.system()
    if not os.path.exists(image_path):
        logger.warning(f"⚠️ [SYSTEM] 画像ファイルが見つかりません: {image_path}")
        return
    try:
        if system == "Windows":
            subprocess.run(["start", "", image_path], check=True, shell=True)
        elif system == "Darwin":
            subprocess.run(["open", image_path], check=True)
        else:
            subprocess.run(["xdg-open", image_path], check=True)
    except Exception as e:
        logger.error(f"❌ [SYSTEM] 画像を開けませんでした: {e}")


def simple_namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: simple_namespace_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [simple_namespace_to_dict(elem) for elem in obj]
    else:
        return obj
