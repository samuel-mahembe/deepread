"""
Embedding generation.

The model is loaded once per process via ``lru_cache`` rather than
``st.cache_resource``. That keeps this module importable and testable
outside Streamlit (plain pytest, the CLI), while still giving a single
cached instance across Streamlit reruns within the same process — a
Streamlit rerun doesn't re-import already-imported modules, so the cache
persists exactly like ``st.cache_resource`` would, without coupling
business logic to the UI framework.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config.settings import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into embedding vectors."""
    if not texts:
        return []
    model = _get_model()
    return model.encode(texts, show_progress_bar=False).tolist()
