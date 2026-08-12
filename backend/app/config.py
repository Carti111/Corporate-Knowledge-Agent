import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))

load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    app_name: str = os.getenv("APP_NAME", "企业知识库问答助手")
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")

    # DeepSeek API
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_api_url: str = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Embedding model
    embedding_model_dir: str = os.getenv(
        "EMBEDDING_MODEL_DIR",
        str(ROOT_DIR / "models" / "bge-large-zh-v1.5"),
    )
    reranker_model_dir: str = os.getenv(
        "RERANKER_MODEL_DIR",
        str(ROOT_DIR / "models" / "bge-reranker-large"),
    )

    # FAISS index
    faiss_index_path: str = os.getenv(
        "FAISS_INDEX_PATH",
        str(DATA_DIR / "faiss_index.bin"),
    )

    # Retrieval settings
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "10"))
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.8"))

    upload_dir: Path = UPLOAD_DIR

    model_config = {"arbitrary_types_allowed": True}


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
