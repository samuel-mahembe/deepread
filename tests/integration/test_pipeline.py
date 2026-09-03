"""
Integration tests for src.rag.pipeline: the full ingest -> chunk -> index ->
retrieve -> generate -> answer path, using controlled local fixture data.

Networking (URL fetch) and the LLM are mocked; chunking, embedding-shape
wiring, and the vector store are real (see conftest for what's faked and why).
"""

from __future__ import annotations

import pytest

from src.config.settings import Settings
from src.ingestion.errors import IngestionError
from src.rag import pipeline

pytestmark = pytest.mark.usefixtures("chroma_env", "deterministic_embeddings")


def _html_response(body_text: str) -> bytes:
    return f"<html><head><title>Fixture Page</title></head><body><p>{body_text}</p></body></html>".encode()


# ---------------------------------------------------------------------------
# Golden path: URL ingestion -> ask -> grounded, cited answer
# ---------------------------------------------------------------------------


def test_full_pipeline_url_ingest_then_ask(fake_requests_get, fake_dns, fake_groq):
    from tests.conftest import FakeHTTPResponse

    body = ("This page is about chunk boundaries and retrieval quality. " * 10).strip()
    fake_requests_get.response = FakeHTTPResponse(_html_response(body))
    fake_groq.set_content("Chunk size affects retrieval quality. [Source 1]")

    outcome = pipeline.ingest_url("https://example.com/chunking", "session-golden")
    assert outcome.success
    assert outcome.chunk_count >= 1

    result = pipeline.ask("How does chunk size affect retrieval?", "session-golden")

    assert result.answer == "Chunk size affects retrieval quality. [Source 1]"
    assert len(result.sources) >= 1
    assert result.sources[0].source == "https://example.com/chunking"
    assert result.sources[0].title == "Fixture Page"

    # The retrieved text actually reached the model as context.
    sent_context = fake_groq.calls[0]["messages"][1]["content"]
    assert "chunk boundaries" in sent_context


# ---------------------------------------------------------------------------
# Golden path: file ingestion -> ask
# ---------------------------------------------------------------------------


def test_full_pipeline_file_ingest_then_ask(make_uploaded_file, fake_groq):
    fake_groq.set_content("Rice needs about twice its volume in water. [Source 1]")

    text = ("A short guide to cooking rice: use twice the water by volume. " * 8).strip()
    f = make_uploaded_file("cooking.txt", text)

    outcome = pipeline.ingest_file(f, "session-file")
    assert outcome.success
    assert outcome.chunk_count >= 1

    result = pipeline.ask("How much water do I need to cook rice?", "session-file")

    assert "twice" in result.answer
    assert result.sources[0].source == "cooking.txt"
    assert result.sources[0].source_type.value == "file"


# ---------------------------------------------------------------------------
# Multiple documents — citations stay attached to the right source
# ---------------------------------------------------------------------------


def test_multiple_documents_are_attributed_to_the_correct_source(make_uploaded_file, fake_groq):
    cat_text = ("Everything about cats: cats sleep sixteen hours a day. " * 8).strip()
    car_text = ("Everything about cars: cars need regular oil changes. " * 8).strip()

    pipeline.ingest_file(make_uploaded_file("cats.txt", cat_text), "session-multi")
    pipeline.ingest_file(make_uploaded_file("cars.txt", car_text), "session-multi")

    fake_groq.set_content("Cats sleep a lot. [Source 1]")
    result = pipeline.ask("How much do cats sleep?", "session-multi", top_k=1)

    assert result.sources[0].source == "cats.txt"
    assert "cat" in result.sources[0].text.lower()


# ---------------------------------------------------------------------------
# Empty / low-content document
# ---------------------------------------------------------------------------


def test_ingest_document_with_too_little_content_fails_without_raising(make_uploaded_file):
    f = make_uploaded_file("empty.txt", "too short")
    outcome = pipeline.ingest_file(f, "session-empty")

    assert outcome.success is False
    assert outcome.chunk_count == 0
    assert outcome.error is not None
    assert pipeline.has_content("session-empty") is False


def test_ingest_unsupported_file_type_fails_without_raising(make_uploaded_file):
    f = make_uploaded_file("archive.zip", b"PK\x03\x04not-a-real-zip")
    outcome = pipeline.ingest_file(f, "session-unsupported")

    assert outcome.success is False
    assert "Unsupported file type" in outcome.error


# ---------------------------------------------------------------------------
# No matching retrieval results
# ---------------------------------------------------------------------------


def test_ask_with_no_relevant_content_returns_fallback_answer(make_uploaded_file, fake_groq, monkeypatch):
    from src.retrieval import retriever

    text = ("All about cooking rice and pasta. " * 8).strip()
    pipeline.ingest_file(make_uploaded_file("cooking.txt", text), "session-nomatch")

    # A strict threshold means the unrelated query matches nothing.
    monkeypatch.setattr(
        retriever, "settings", Settings(default_top_k=4, retrieval_score_threshold=0.9, enable_reranking=False)
    )

    result = pipeline.ask("What is the python programming language?", "session-nomatch")

    assert result.sources == []
    assert "don't have enough information" in result.answer
    assert fake_groq.calls == []  # generate_answer short-circuits on empty chunks


def test_ask_against_never_ingested_session_is_safe(fake_groq):
    result = pipeline.ask("anything at all", "session-never-touched")
    assert result.sources == []
    assert "don't have enough information" in result.answer


# ---------------------------------------------------------------------------
# LLM failure propagates (app.py is the layer that catches it)
# ---------------------------------------------------------------------------


def test_llm_failure_propagates_out_of_pipeline_ask(make_uploaded_file, fake_groq):
    text = ("Some content about cats for retrieval. " * 8).strip()
    pipeline.ingest_file(make_uploaded_file("cats.txt", text), "session-llmfail")

    fake_groq.set_error(TimeoutError("Groq request timed out"))

    with pytest.raises(TimeoutError):
        pipeline.ask("Tell me about cats", "session-llmfail")


# ---------------------------------------------------------------------------
# Session isolation, end to end through the pipeline (not just the store)
# ---------------------------------------------------------------------------


def test_pipeline_sessions_are_isolated(make_uploaded_file, fake_groq):
    pipeline.ingest_file(make_uploaded_file("a.txt", ("Session A content about cats. " * 8)), "session-a")
    pipeline.ingest_file(make_uploaded_file("b.txt", ("Session B content about cars. " * 8)), "session-b")

    fake_groq.set_content("answer")
    result_a = pipeline.ask("cats", "session-a")
    result_b = pipeline.ask("cars", "session-b")

    assert {s.source for s in result_a.sources} == {"a.txt"}
    assert {s.source for s in result_b.sources} == {"b.txt"}

    pipeline.clear_session("session-a")
    assert pipeline.has_content("session-a") is False
    assert pipeline.has_content("session-b") is True


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------


def test_suggest_questions_for_session_uses_ingested_content(make_uploaded_file, fake_groq):
    text = ("A document all about cooking rice and pasta. " * 8).strip()
    pipeline.ingest_file(make_uploaded_file("cooking.txt", text), "session-suggest")

    fake_groq.set_content("1. How long does rice take to cook?\n2. What pasta shapes work best?")

    questions = pipeline.suggest_questions_for_session("session-suggest")

    assert questions == ["How long does rice take to cook?", "What pasta shapes work best?"]
    sent_prompt = fake_groq.calls[0]["messages"][0]["content"]
    assert "cooking" in sent_prompt.lower()


def test_suggest_questions_for_empty_session_returns_empty_list(fake_groq):
    questions = pipeline.suggest_questions_for_session("session-never-ingested")
    assert questions == []
    assert fake_groq.calls == []


# ---------------------------------------------------------------------------
# Malformed / unexpected input doesn't corrupt pipeline state
# ---------------------------------------------------------------------------


def test_ingest_url_with_invalid_scheme_returns_error_outcome_not_exception(fake_dns):
    outcome = pipeline.ingest_url("ftp://example.com/file", "session-badurl")
    assert outcome.success is False
    assert isinstance(outcome.error, str)


def test_ingest_url_targeting_private_ip_is_rejected(fake_dns):
    fake_dns.ip = "127.0.0.1"
    outcome = pipeline.ingest_url("https://looks-fine.example.com", "session-ssrf")
    assert outcome.success is False
    assert "private/internal" in outcome.error


def test_ingest_urls_processes_each_url_independently(fake_requests_get, fake_dns):
    from tests.conftest import FakeHTTPResponse

    fake_requests_get.response = FakeHTTPResponse(_html_response("Cats and cooking content. " * 20))

    outcomes = pipeline.ingest_urls(["https://example.com/a", "https://example.com/b"], "session-batch-urls")

    assert len(outcomes) == 2
    assert all(o.success for o in outcomes)
    assert {o.source for o in outcomes} == {"https://example.com/a", "https://example.com/b"}


def test_ingest_files_processes_each_file_independently(make_uploaded_file):
    files = [make_uploaded_file(f"doc{i}.txt", (f"content about doc {i}. " * 20)) for i in range(3)]

    outcomes = pipeline.ingest_files(files, "session-batch-files")

    assert len(outcomes) == 3
    assert all(o.success for o in outcomes)
    assert {o.source for o in outcomes} == {"doc0.txt", "doc1.txt", "doc2.txt"}


def test_pipeline_never_raises_bare_ingestion_errors(make_uploaded_file):
    """ingest_* catches IngestionError internally — callers get an IngestOutcome, not an exception."""
    try:
        outcome = pipeline.ingest_file(make_uploaded_file("bad.pdf", b"not a real pdf"), "session-badpdf")
    except IngestionError:
        pytest.fail("ingest_file should convert IngestionError into a failed IngestOutcome, not raise")
    assert outcome.success is False
