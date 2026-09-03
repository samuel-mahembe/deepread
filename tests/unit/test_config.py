"""Tests for src.config.settings: env loading, defaults, missing-config behavior."""

from __future__ import annotations

from src.config.settings import Settings


def test_groq_api_key_read_from_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-from-env")
    s = Settings()
    assert s.groq_api_key == "test-key-from-env"


def test_groq_api_key_none_when_unset(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    s = Settings()
    assert s.groq_api_key is None


def test_enable_reranking_defaults_off(monkeypatch):
    monkeypatch.delenv("ENABLE_RERANKING", raising=False)
    s = Settings()
    assert s.enable_reranking is False


def test_enable_reranking_true_when_env_says_true(monkeypatch):
    monkeypatch.setenv("ENABLE_RERANKING", "true")
    s = Settings()
    assert s.enable_reranking is True


def test_enable_reranking_false_for_any_non_true_value(monkeypatch):
    monkeypatch.setenv("ENABLE_RERANKING", "yes")
    s = Settings()
    assert s.enable_reranking is False


def test_settings_is_immutable():
    """Settings is a frozen dataclass — config shouldn't be mutable at runtime."""
    s = Settings()
    with __import__("pytest").raises(Exception):
        s.llm_model = "something-else"


def test_default_llm_and_embedding_models_are_set():
    s = Settings()
    assert s.llm_model
    assert s.embedding_model_name
    assert s.reranker_model_name


def test_default_chunking_values():
    s = Settings()
    assert s.chunk_size > 0
    assert 0 <= s.chunk_overlap < s.chunk_size


def test_default_retrieval_top_k():
    s = Settings()
    assert s.default_top_k > 0


def test_default_url_and_file_limits_are_positive():
    s = Settings()
    assert s.max_url_content_bytes > 0
    assert s.max_file_size_bytes > 0
    assert s.request_timeout_seconds > 0


def test_streamlit_secrets_take_precedence_over_env_var(monkeypatch):
    import streamlit

    monkeypatch.setattr(streamlit, "secrets", {"GROQ_API_KEY": "from-streamlit-secrets"})
    monkeypatch.setenv("GROQ_API_KEY", "from-env-should-be-ignored")

    s = Settings()
    assert s.groq_api_key == "from-streamlit-secrets"


def test_default_allowed_schemes_and_extensions():
    s = Settings()
    assert "http" in s.allowed_url_schemes
    assert "https" in s.allowed_url_schemes
    assert "ftp" not in s.allowed_url_schemes
    assert ".pdf" in s.allowed_file_extensions
    assert ".exe" not in s.allowed_file_extensions
