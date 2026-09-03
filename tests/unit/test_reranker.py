"""
Tests for src.retrieval.reranker. The cross-encoder model is always faked
here — this tests the ranking/truncation logic, not sentence-transformers.
"""

from __future__ import annotations

from src.models.schemas import RetrievedChunk, SourceType
from src.retrieval import reranker


def _candidate(text: str, distance: float = 0.2) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source="s", title="t", source_type=SourceType.URL, chunk_index=0, distance=distance
    )


def test_rerank_empty_candidates_short_circuits_without_loading_a_model(monkeypatch):
    def _fail_if_called():
        raise AssertionError("_get_reranker should not be called for an empty candidate list")

    monkeypatch.setattr(reranker, "_get_reranker", _fail_if_called)

    assert reranker.rerank("query", [], top_n=5) == []


def test_rerank_orders_candidates_by_score_descending(monkeypatch):
    candidates = [_candidate("low relevance"), _candidate("high relevance"), _candidate("medium relevance")]

    class FakeCrossEncoder:
        def predict(self, pairs):
            score_by_text = {"low relevance": 0.1, "high relevance": 0.9, "medium relevance": 0.5}
            return [score_by_text[text] for _, text in pairs]

    monkeypatch.setattr(reranker, "_get_reranker", lambda: FakeCrossEncoder())

    ranked = reranker.rerank("query", candidates, top_n=3)

    assert [c.text for c in ranked] == ["high relevance", "medium relevance", "low relevance"]


def test_rerank_truncates_to_top_n(monkeypatch):
    candidates = [_candidate(f"doc {i}") for i in range(5)]

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [1.0] * len(pairs)

    monkeypatch.setattr(reranker, "_get_reranker", lambda: FakeCrossEncoder())

    ranked = reranker.rerank("query", candidates, top_n=2)

    assert len(ranked) == 2
