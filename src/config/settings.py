"""
Centralized configuration for DeepRead.

Every tunable value in the application (model names, chunk size, limits,
timeouts) lives here instead of being scattered across modules. Values can
be overridden via environment variables or Streamlit secrets, so the same
code runs unmodified in local development and on Streamlit Community Cloud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Resolve a config value from Streamlit secrets first, then env vars.

    Streamlit secrets are only available inside a running Streamlit app
    (and raise if no secrets.toml exists), so this falls back to the
    environment for local scripts, tests, and the CLI. ``load_dotenv()``
    above populates the environment from a local .env file first (a no-op
    on Streamlit Cloud, which has no .env and uses secrets.toml instead).
    """
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    groq_api_key: str | None = field(default_factory=lambda: _get_secret("GROQ_API_KEY"))
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 500
    llm_request_timeout_seconds: int = 30

    # --- Embeddings ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Reranking (optional, off by default: extra model load + latency) ---
    enable_reranking: bool = field(default_factory=lambda: _get_secret("ENABLE_RERANKING", "false") == "true")
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Vector store ---
    chroma_persist_dir: str = "data/chroma_db"

    # --- Chunking ---
    chunk_size: int = 500  # characters, not tokens
    chunk_overlap: int = 50

    # --- Retrieval ---
    default_top_k: int = 4
    retrieval_score_threshold: float | None = None  # min similarity (0-1); None = no filter

    # --- URL ingestion limits ---
    request_timeout_seconds: int = 10
    max_url_content_bytes: int = 5_000_000  # 5 MB
    allowed_url_schemes: tuple[str, ...] = ("http", "https")

    # --- File ingestion limits ---
    max_file_size_bytes: int = 10_000_000  # 10 MB
    allowed_file_extensions: tuple[str, ...] = (".pdf", ".txt", ".md", ".docx")

    # --- Misc ---
    min_extracted_chars: int = 200  # below this, warn that a source looks empty/JS-rendered


settings = Settings()
