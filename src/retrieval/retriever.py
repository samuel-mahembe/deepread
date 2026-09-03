"""Query-time retrieval: embed the question, search the vector store, optionally rerank/filter."""

from __future__ import annotations

from src.config.settings import settings
from src.embeddings.embedding_service import embed
from src.models.schemas import RetrievedChunk
from src.retrieval.reranker import rerank
from src.vectorstore import chroma_store


def retrieve(
    query: str,
    collection_id: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k most relevant chunks for a query.

    If reranking is enabled (see settings.enable_reranking), an initial
    over-fetch is retrieved by embedding similarity, then a cross-encoder
    reranks and truncates to top_k for better precision at the cost of
    latency.
    """
    top_k = top_k or settings.default_top_k
    threshold = settings.retrieval_score_threshold if score_threshold is None else score_threshold

    fetch_k = top_k * 3 if settings.enable_reranking else top_k
    query_embedding = embed([query])[0]
    candidates = chroma_store.query(collection_id, query_embedding, fetch_k)

    if settings.enable_reranking and candidates:
        candidates = rerank(query, candidates, top_n=top_k)
    else:
        candidates = candidates[:top_k]

    if threshold is not None:
        candidates = [c for c in candidates if c.similarity >= threshold]

    return candidates
