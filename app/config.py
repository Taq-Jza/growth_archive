"""配置。所有可调项走环境变量，默认值保证 `python -m app` 就能跑。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_PATH = DATA_DIR / "app.db"

# ---- 模型服务（OpenAI 兼容接口）----
# 硅基流动 / 阿里百炼兼容模式 / 华为云 ModelArts Studio 都走这套
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
VISION_MODEL = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
# 图片重绘：把板书写真重排成整洁的笔记图，只作为阅读辅助
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "Qwen/Qwen-Image-Edit-2509")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x768")
# typeset = 用识别出的文字重新排版（默认，快且不会错字）；ai = 走图生图模型重绘
IMAGE_REDRAW = os.getenv("IMAGE_REDRAW", "typeset")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

# ---- 应用 ----
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-please-change")
DEMO_EMAIL = os.getenv("DEMO_EMAIL", "demo@local")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo1234")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
