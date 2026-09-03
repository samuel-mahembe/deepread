"""LLM calls (Groq). The only module that talks to the generation API."""

from __future__ import annotations

from functools import lru_cache

from groq import Groq

from src.config.settings import settings
from src.generation.prompts import build_answer_messages, build_suggestion_prompt
from src.models.schemas import RetrievedChunk


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM client can't be created (e.g. missing API key)."""


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    if not settings.groq_api_key:
        raise LLMConfigurationError("GROQ_API_KEY is not set. Add it to .env (local) or Streamlit secrets (cloud).")
    return Groq(api_key=settings.groq_api_key)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Generate a grounded answer from retrieved chunks. Empty chunks -> explicit no-context reply."""
    if not chunks:
        return "I don't have enough information in the provided sources to answer that."

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=build_answer_messages(question, chunks),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_request_timeout_seconds,
    )
    # The API can return null content (e.g. a filtered/refused completion) even
    # on a 200 response, which would otherwise violate this function's `str`
    # return type and break callers like `st.markdown(result.answer)`.
    content = response.choices[0].message.content
    return content or "The model returned an empty response. Please try again."


def suggest_questions(chunks: list[RetrievedChunk], n: int = 4) -> list[str]:
    """Generate suggested questions based on a sample of ingested content."""
    if not chunks:
        return []

    sample_text = "\n\n".join(c.text[:300] for c in chunks[:8])
    prompt = build_suggestion_prompt(sample_text, n)

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300,
        timeout=settings.llm_request_timeout_seconds,
    )

    raw = response.choices[0].message.content or ""
    questions = [q.strip() for q in raw.split("\n") if q.strip()]
    questions = [q.lstrip("0123456789.-) ").strip() for q in questions]
    return questions[:n]
