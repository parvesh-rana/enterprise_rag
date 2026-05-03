"""Runtime configuration. All values come from .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    """All env-driven config. Never read os.environ directly elsewhere."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider ---
    llm_provider: Literal["nvidia", "anthropic", "ollama"] = "nvidia"
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model: str = "deepseek-ai/deepseek-v3.1"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1:8b"

    # --- Embeddings & reranker ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20
    final_top_k: int = 5

    # --- ChromaDB ---
    chroma_persist_dir: str = "data/chroma"
    chroma_collection: str = "filings_v1"

    # --- SEC EDGAR ---
    edgar_user_agent: str = Field(
        default="Enterprise RAG Demo contact@example.com",
        description="SEC requires a User-Agent identifying the requester.",
    )

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 60

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True

    # --- Paths (derived; not env-driven) ---
    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def raw_dir(self) -> Path:
        return DATA_DIR / "raw"

    @property
    def chunks_dir(self) -> Path:
        return DATA_DIR / "chunks"

    @property
    def bm25_dir(self) -> Path:
        return DATA_DIR / "bm25"

    @property
    def sample_dir(self) -> Path:
        return DATA_DIR / "sample"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Tests can call get_settings.cache_clear()."""
    return Settings()
