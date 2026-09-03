"""
Vector store + retrieval tests. Uses a real (temp-directory) ChromaDB and a
deterministic keyword-vector embedding stand-in (see conftest.fake_embed) so
similarity ranking is predictable without loading the real embedding model.
"""

from __future__ import annotations

from src.config.settings import Settings
from src.models.schemas import Chunk, SourceType
from src.retrieval.retriever import retrieve
from src.vectorstore import chroma_store


def _make_chunks(*, source: str, title: str, texts: list[str], source_type=SourceType.URL) -> list[Chunk]:
    return [
        Chunk(text=t, source=source, title=title, source_type=source_type, chunk_index=i) for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# chroma_store — direct vector store behavior
# ---------------------------------------------------------------------------


def test_add_and_query_returns_indexed_chunk(chroma_env, deterministic_embeddings):
    chunks = _make_chunks(source="s1", title="Cats", texts=["All about cats and kittens."])
    embeddings = deterministic_embeddings.embed([c.text for c in chunks])
    chroma_store.add_chunks("session-a", chunks, embeddings)

    results = chroma_store.query("session-a", deterministic_embeddings.vector_for("cat"), top_k=5)

    assert len(results) == 1
    assert results[0].text == "All about cats and kittens."
    assert results[0].source == "s1"
    assert results[0].title == "Cats"
    assert results[0].source_type is SourceType.URL


def test_query_on_empty_collection_returns_no_results(chroma_env):
    results = chroma_store.query("never-touched-session", [1.0, 0.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_top_k_limits_number_of_results(chroma_env, deterministic_embeddings):
    chunks = _make_chunks(
        source="s1",
        title="Cats",
        texts=[f"Cat fact number {i}." for i in range(10)],
    )
    embeddings = deterministic_embeddings.embed([c.text for c in chunks])
    chroma_store.add_chunks("session-topk", chunks, embeddings)

    results = chroma_store.query("session-topk", deterministic_embeddings.vector_for("cat"), top_k=3)
    assert len(results) == 3


def test_top_k_larger_than_collection_returns_all_available(chroma_env, deterministic_embeddings):
    chunks = _make_chunks(source="s1", title="Cats", texts=["one cat", "two cats"])
    embeddings = deterministic_embeddings.embed([c.text for c in chunks])
    chroma_store.add_chunks("session-small", chunks, embeddings)

    results = chroma_store.query("session-small", deterministic_embeddings.vector_for("cat"), top_k=50)
    assert len(results) == 2


def test_upsert_overwrites_rather_than_duplicates(chroma_env, deterministic_embeddings):
    chunk = _make_chunks(source="s1", title="Cats", texts=["original text"])
    chroma_store.add_chunks("session-upsert", chunk, deterministic_embeddings.embed(["original text"]))

    updated = _make_chunks(source="s1", title="Cats", texts=["updated text"])
    chroma_store.add_chunks("session-upsert", updated, deterministic_embeddings.embed(["updated text"]))

    assert chroma_store.count("session-upsert") == 1


def test_delete_collection_removes_its_data(chroma_env, deterministic_embeddings):
    chunks = _make_chunks(source="s1", title="Cats", texts=["cat content"])
    chroma_store.add_chunks("session-del", chunks, deterministic_embeddings.embed(["cat content"]))
    assert chroma_store.count("session-del") == 1

    chroma_store.delete_collection("session-del")

    assert chroma_store.count("session-del") == 0


def test_delete_collection_is_safe_when_collection_never_existed(chroma_env):
    chroma_store.delete_collection("never-existed")  # should not raise


def test_sessions_do_not_leak_data_into_each_other(chroma_env, deterministic_embeddings):
    cat_chunks = _make_chunks(source="cats.txt", title="Cats", texts=["cat content only"])
    car_chunks = _make_chunks(source="cars.txt", title="Cars", texts=["car content only"])

    chroma_store.add_chunks("session-alice", cat_chunks, deterministic_embeddings.embed(["cat content only"]))
    chroma_store.add_chunks("session-bob", car_chunks, deterministic_embeddings.embed(["car content only"]))

    alice_results = chroma_store.query("session-alice", deterministic_embeddings.vector_for("cat"), top_k=10)
    bob_results = chroma_store.query("session-bob", deterministic_embeddings.vector_for("car"), top_k=10)

    assert {r.source for r in alice_results} == {"cats.txt"}
    assert {r.source for r in bob_results} == {"cars.txt"}


def test_clearing_one_session_does_not_affect_another(chroma_env, deterministic_embeddings):
    chroma_store.add_chunks(
        "session-alice",
        _make_chunks(source="a.txt", title="A", texts=["alice content"]),
        deterministic_embeddings.embed(["alice content"]),
    )
    chroma_store.add_chunks(
        "session-bob",
        _make_chunks(source="b.txt", title="B", texts=["bob content"]),
        deterministic_embeddings.embed(["bob content"]),
    )

    chroma_store.delete_collection("session-alice")

    assert chroma_store.count("session-alice") == 0
    assert chroma_store.count("session-bob") == 1


# ---------------------------------------------------------------------------
# retriever.retrieve() — embedding + search + threshold filtering
# ---------------------------------------------------------------------------


def test_retrieve_finds_the_topically_relevant_chunk(chroma_env, deterministic_embeddings):
    chunks = _make_chunks(
        source="mixed.txt",
        title="Mixed topics",
        texts=["A document entirely about cats.", "A document entirely about cars.", "A document about cooking."],
    )
    chroma_store.add_chunks("session-x", chunks, deterministic_embeddings.embed([c.text for c in chunks]))

    results = retrieve("Tell me about cats", "session-x", top_k=1)

    assert len(results) == 1
    assert "cats" in results[0].text


def test_retrieve_returns_empty_list_for_empty_session(chroma_env, deterministic_embeddings):
    results = retrieve("anything", "session-never-ingested", top_k=4)
    assert results == []


def test_retrieve_applies_score_threshold(chroma_env, deterministic_embeddings):
    chunks = _make_chunks(source="mixed.txt", title="Mixed", texts=["all about cats", "all about cooking"])
    chroma_store.add_chunks("session-threshold", chunks, deterministic_embeddings.embed([c.text for c in chunks]))

    # A query about "python" shares no keyword with either chunk, so similarity
    # should be low enough that a strict threshold filters everything out —
    # this must return [], not raise.
    results = retrieve("python", "session-threshold", top_k=4, score_threshold=0.9)
    assert results == []


def test_retrieve_preserves_source_and_title_metadata(chroma_env, deterministic_embeddings):
    chunks = _make_chunks(
        source="https://example.com/cooking-guide", title="Cooking Guide", texts=["A guide about how to cook rice."]
    )
    chroma_store.add_chunks("session-meta", chunks, deterministic_embeddings.embed([c.text for c in chunks]))

    results = retrieve("cook", "session-meta", top_k=1)

    assert results[0].source == "https://example.com/cooking-guide"
    assert results[0].title == "Cooking Guide"


def test_settings_default_top_k_is_used_when_not_specified(chroma_env, deterministic_embeddings, monkeypatch):
    from src.retrieval import retriever

    monkeypatch.setattr(
        retriever, "settings", Settings(default_top_k=2, enable_reranking=False, retrieval_score_threshold=None)
    )

    chunks = _make_chunks(
        source="s.txt",
        title="S",
        texts=[f"cat fact {i}" for i in range(5)],
    )
    chroma_store.add_chunks("session-default-k", chunks, deterministic_embeddings.embed([c.text for c in chunks]))

    results = retrieve("cat", "session-default-k")  # top_k omitted
    assert len(results) == 2


def test_retrieve_calls_reranker_when_enabled(chroma_env, deterministic_embeddings, monkeypatch):
    from src.retrieval import retriever

    monkeypatch.setattr(
        retriever, "settings", Settings(default_top_k=1, enable_reranking=True, retrieval_score_threshold=None)
    )

    chunks = _make_chunks(source="s", title="S", texts=["cat fact one", "cat fact two", "cat fact three"])
    chroma_store.add_chunks("session-rerank", chunks, deterministic_embeddings.embed([c.text for c in chunks]))

    rerank_calls = []

    def fake_rerank(query, candidates, top_n):
        rerank_calls.append((query, len(candidates), top_n))
        return candidates[:top_n]

    monkeypatch.setattr(retriever, "rerank", fake_rerank)

    results = retrieve("cat", "session-rerank", top_k=1)

    assert len(rerank_calls) == 1
    called_query, num_candidates, top_n = rerank_calls[0]
    assert called_query == "cat"
    assert top_n == 1
    assert num_candidates == 3  # over-fetched at top_k * 3 before reranking
    assert len(results) == 1


def test_retrieve_skips_reranker_when_disabled(chroma_env, deterministic_embeddings, monkeypatch):
    from src.retrieval import retriever

    monkeypatch.setattr(
        retriever, "settings", Settings(default_top_k=1, enable_reranking=False, retrieval_score_threshold=None)
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("rerank should not be called when enable_reranking is False")

    monkeypatch.setattr(retriever, "rerank", _fail_if_called)

    chunks = _make_chunks(source="s", title="S", texts=["cat fact"])
    chroma_store.add_chunks("session-norerank", chunks, deterministic_embeddings.embed([c.text for c in chunks]))

    results = retrieve("cat", "session-norerank", top_k=1)
    assert len(results) == 1
