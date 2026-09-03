"""
Optional cross-encoder reranking.

Embedding similarity (bi-encoder) is fast but scores query and document
independently, which misses fine-grained relevance. A cross-encoder scores
the (query, document) pair jointly and is more accurate, at the cost of
being much slower per-document — so it's only used to re-score a small
over-fetched candidate set, not the whole collection, and is disabled by
default (`settings.enable_reranking`) to keep Streamlit Cloud's free-tier
memory/latency budget predictable. Uses sentence-transformers' CrossEncoder,
so no extra dependency beyond what embeddings already require.
"""

from __future__ import annotations

from functools import lru_cache

from src.config.settings import settings
from src.models.schemas import RetrievedChunk


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model_name)


def rerank(query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """Re-score candidates with a cross-encoder and return the top_n, best first."""
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_n]]
