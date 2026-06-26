"""
Central configuration management using Pydantic Settings.
All environment variables are loaded from .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────
    groq_api_key: str
    groq_primary_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"

    # ── Embedding ────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # ── Vector DB ────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "lexi_find"

    # ── Security ─────────────────────────────────
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_key_header: str = "X-API-Key"
    # api_keys: str = "dev-secret-key-change-in-production"

    # ── Rate Limiting ────────────────────────────
    rate_limit_per_minute: int = 30
    rate_limit_per_day: int = 500

    # ── Pipeline ─────────────────────────────────
    default_pipeline_mode: str = "auto"
    max_agent_iterations: int = 5
    top_k_retrieval: int = 10
    final_n_rerank: int = 3
    hybrid_alpha: float = 0.5

    # ── Chunking ─────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Logging ──────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"


@lru_cache
def get_settings() -> Settings:
    """
    Returns cached Settings instance.
    Use as FastAPI dependency: Depends(get_settings)
    """
    return Settings()