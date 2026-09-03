"""
Generation-layer tests. The Groq client is always faked (see conftest.fake_groq)
— no test in this file makes a real API call.
"""

from __future__ import annotations

import pytest

from src.generation import llm_service
from src.generation.prompts import build_answer_messages, format_context
from src.models.schemas import RetrievedChunk, SourceType


def _chunk(text: str, title: str = "Doc", source: str = "doc.txt", distance: float = 0.1) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source=source, title=title, source_type=SourceType.FILE, chunk_index=0, distance=distance
    )


# ---------------------------------------------------------------------------
# prompts.py — construction
# ---------------------------------------------------------------------------


def test_format_context_labels_each_source():
    chunks = [_chunk("first chunk text", title="Alpha"), _chunk("second chunk text", title="Beta")]
    context = format_context(chunks)

    assert "[Source 1: Alpha]" in context
    assert "[Source 2: Beta]" in context
    assert "first chunk text" in context
    assert "second chunk text" in context


def test_build_answer_messages_includes_question_and_context():
    chunks = [_chunk("relevant passage", title="Doc A")]
    messages = build_answer_messages("What does Doc A say?", chunks)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "relevant passage" in messages[1]["content"]
    assert "What does Doc A say?" in messages[1]["content"]


# ---------------------------------------------------------------------------
# llm_service.generate_answer
# ---------------------------------------------------------------------------


def test_generate_answer_returns_fixed_reply_and_skips_api_call_when_no_chunks(fake_groq):
    answer = llm_service.generate_answer("anything", [])
    assert "don't have enough information" in answer
    assert fake_groq.calls == []


def test_generate_answer_returns_model_content(fake_groq):
    fake_groq.set_content("The capital is Paris. [Source 1]")
    answer = llm_service.generate_answer("What is the capital?", [_chunk("France's capital is Paris.")])
    assert answer == "The capital is Paris. [Source 1]"


def test_generate_answer_sends_retrieved_context_to_the_model(fake_groq):
    fake_groq.set_content("ok")
    chunks = [_chunk("The sky appears blue due to Rayleigh scattering.", title="Physics Notes")]
    llm_service.generate_answer("Why is the sky blue?", chunks)

    sent_messages = fake_groq.calls[0]["messages"]
    user_message = sent_messages[1]["content"]
    assert "Rayleigh scattering" in user_message
    assert "Physics Notes" in user_message
    assert "Why is the sky blue?" in user_message


def test_generate_answer_handles_null_content_safely(fake_groq):
    fake_groq.set_content(None)
    answer = llm_service.generate_answer("q", [_chunk("some context")])
    assert isinstance(answer, str)
    assert answer  # non-empty fallback, not None and not ""


def test_generate_answer_propagates_api_errors(fake_groq):
    fake_groq.set_error(RuntimeError("Groq request timed out"))
    with pytest.raises(RuntimeError, match="timed out"):
        llm_service.generate_answer("q", [_chunk("some context")])


# ---------------------------------------------------------------------------
# llm_service.suggest_questions
# ---------------------------------------------------------------------------


def test_suggest_questions_returns_empty_list_for_no_chunks(fake_groq):
    assert llm_service.suggest_questions([]) == []
    assert fake_groq.calls == []


def test_suggest_questions_parses_numbered_list(fake_groq):
    fake_groq.set_content("1. What is X?\n2. How does Y work?\n3. Why does Z matter?")
    questions = llm_service.suggest_questions([_chunk("some content")], n=4)
    assert questions == ["What is X?", "How does Y work?", "Why does Z matter?"]


def test_suggest_questions_respects_n_limit(fake_groq):
    fake_groq.set_content("\n".join(f"Question {i}?" for i in range(10)))
    questions = llm_service.suggest_questions([_chunk("content")], n=2)
    assert len(questions) == 2


def test_suggest_questions_handles_null_content_safely(fake_groq):
    fake_groq.set_content(None)
    questions = llm_service.suggest_questions([_chunk("content")])
    assert questions == []


def test_suggest_questions_propagates_api_errors(fake_groq):
    fake_groq.set_error(ConnectionError("network unreachable"))
    with pytest.raises(ConnectionError):
        llm_service.suggest_questions([_chunk("content")])


# ---------------------------------------------------------------------------
# llm_service._get_client — missing configuration
# ---------------------------------------------------------------------------


def test_get_client_raises_configuration_error_without_api_key(monkeypatch):
    from src.config.settings import Settings

    monkeypatch.setattr(llm_service, "settings", Settings(groq_api_key=None))
    llm_service._get_client.cache_clear()
    try:
        with pytest.raises(llm_service.LLMConfigurationError, match="GROQ_API_KEY"):
            llm_service._get_client()
    finally:
        llm_service._get_client.cache_clear()
