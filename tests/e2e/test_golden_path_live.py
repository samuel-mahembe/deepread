"""
The real golden-path check: a live Wikipedia URL, the real embedding model,
a real temp ChromaDB, and a real call to the Groq API.

This is deliberately NOT part of the default `pytest` run — it needs
internet access and a real GROQ_API_KEY, neither of which the rest of the
suite (or CI) should depend on. Run it explicitly:

    pytest -m live

It's the same check that was run manually against the app during the
environment-consolidation and refactor work; this just makes it repeatable
instead of a one-off shell session.
"""

from __future__ import annotations

import pytest

from src.config.settings import settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not settings.groq_api_key, reason="GROQ_API_KEY not set — see .env.example"),
]


def test_real_golden_path_url_ingest_ask_and_cite(chroma_env):
    from src.rag import pipeline

    session = "live-golden-path-test"
    pipeline.clear_session(session)
    try:
        outcome = pipeline.ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation", session)
        assert outcome.success, outcome.error
        assert outcome.chunk_count > 0

        result = pipeline.ask("What is retrieval-augmented generation?", session)

        assert result.sources
        assert "retriev" in result.answer.lower()

        questions = pipeline.suggest_questions_for_session(session)
        assert questions
    finally:
        pipeline.clear_session(session)
