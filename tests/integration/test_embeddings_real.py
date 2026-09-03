"""
Integration test for the real embedding implementation.

Every other test in the suite mocks src.embeddings.embedding_service.embed
for speed and determinism (see conftest.deterministic_embeddings). This file
exercises the real sentence-transformers model instead, so the mock's
assumptions (same input -> same vector, similar text -> higher cosine
similarity) are validated against the real thing at least once.

Runs offline against the locally cached model — no network call is made
here (only the first-ever run on a machine downloads the model, which is
the same one-time cost documented in README's "Run locally" section).
"""

from __future__ import annotations

import math

from src.embeddings.embedding_service import embed


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def test_embed_empty_list_returns_empty_list():
    assert embed([]) == []


def test_embed_returns_one_vector_per_input_text():
    vectors = embed(["cats are great", "cars need fuel", "cooking rice"])
    assert len(vectors) == 3
    assert all(len(v) == len(vectors[0]) for v in vectors)
    assert len(vectors[0]) > 0


def test_embed_is_deterministic_for_the_same_input():
    a = embed(["Retrieval-augmented generation grounds answers in real sources."])
    b = embed(["Retrieval-augmented generation grounds answers in real sources."])
    assert a == b


def test_embed_places_semantically_similar_text_closer_than_unrelated_text():
    query = embed(["What do cats eat?"])[0]
    cat_doc = embed(["Cats are obligate carnivores and mostly eat meat."])[0]
    unrelated_doc = embed(["The stock market closed higher on Tuesday."])[0]

    assert _cosine(query, cat_doc) > _cosine(query, unrelated_doc)
