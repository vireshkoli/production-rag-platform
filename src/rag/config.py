"""Central configuration. All values overridable via environment (prefix RAG_) or .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM (Anthropic). The API key is read by the anthropic SDK from
    # ANTHROPIC_API_KEY directly; it is not a RAG_-prefixed setting.
    generation_model: str = "claude-haiku-4-5"
    judge_model: str = "claude-haiku-4-5"

    # Storage
    qdrant_path: Path = Path("data/qdrant")
    collection: str = "wikipedia_ai_ml"
    telemetry_db: Path = Path("data/telemetry.sqlite3")

    # Local models
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60
    prefetch_k: int = 25  # candidates per retriever before fusion
    rerank_candidates: int = 20  # fused candidates fed to the cross-encoder
    top_k: int = 5  # chunks handed to the LLM

    # Agentic self-check
    groundedness_threshold: float = 0.75
    max_retrieval_rounds: int = 2  # initial + one corrective re-retrieval

    # Cost guard: hard per-query budget so a runaway loop can't burn money
    max_query_cost_usd: float = 0.05


@lru_cache
def settings() -> Settings:
    return Settings()
