"""
Shared fixtures for the DeepRead test suite.

Design notes:

- No test in the default (non-``live``) run touches the network, a real
  Chroma database on disk, or the real Groq API. Vector storage uses a
  per-test temp directory; the LLM is a small fake client; embeddings for
  anything above the embedding-service layer use a deterministic
  keyword-vector stand-in (see ``fake_embed``) so retrieval ranking is
  predictable without loading the real sentence-transformers model.
- The one exception is ``tests/e2e/test_golden_path_live.py``, marked
  ``@pytest.mark.live`` and excluded by ``addopts`` in pyproject.toml.
- Nothing here reads the developer's real ``.env``. Config tests that care
  about environment variables set/unset them explicitly via ``monkeypatch``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document_text() -> str:
    return (FIXTURES_DIR / "sample_document.txt").read_text(encoding="utf-8")


@pytest.fixture
def sample_html_bytes() -> bytes:
    return (FIXTURES_DIR / "sample.html").read_bytes()


# ---------------------------------------------------------------------------
# Uploaded-file stand-in (matches src.ingestion.document_loader.UploadedFileLike)
# ---------------------------------------------------------------------------


class FakeUploadedFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@pytest.fixture
def make_uploaded_file():
    def _factory(name: str, data: bytes | str) -> FakeUploadedFile:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return FakeUploadedFile(name, data)

    return _factory


# ---------------------------------------------------------------------------
# Isolated ChromaDB (real Chroma, temp directory, cleared between tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_env(tmp_path, monkeypatch):
    """Point src.vectorstore.chroma_store at a fresh temp directory.

    Real ChromaDB, real cosine math — only the on-disk location is faked,
    so retrieval tests exercise the actual vector store, not a mock of it.
    """
    from src.config.settings import Settings
    from src.vectorstore import chroma_store

    test_settings = Settings(chroma_persist_dir=str(tmp_path / "chroma_db"))
    monkeypatch.setattr(chroma_store, "settings", test_settings)
    chroma_store._get_client.cache_clear()
    yield test_settings
    chroma_store._get_client.cache_clear()


# ---------------------------------------------------------------------------
# Deterministic fake embeddings (used everywhere except the real-embedding
# integration test)
# ---------------------------------------------------------------------------

# Small fixed vocabulary. Each dimension lights up when its keyword appears
# in the text, so cosine similarity between two texts is fully predictable
# from which keywords they share. Every vector uses a 0.05 floor (never a
# true zero vector, which cosine similarity can't compare) so unrelated
# text is still a valid, low-similarity point rather than undefined.
EMBED_VOCAB = ["cat", "car", "cook", "python", "chunk"]


def _keyword_vector(text: str) -> list[float]:
    lowered = text.lower()
    return [1.0 if word in lowered else 0.05 for word in EMBED_VOCAB]


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [_keyword_vector(t) for t in texts]


@pytest.fixture
def deterministic_embeddings(monkeypatch):
    """Patch the `embed` name in every module that imported it directly."""
    from src.rag import pipeline
    from src.retrieval import retriever

    monkeypatch.setattr(pipeline, "embed", fake_embed)
    monkeypatch.setattr(retriever, "embed", fake_embed)
    return SimpleNamespace(vocab=EMBED_VOCAB, embed=fake_embed, vector_for=_keyword_vector)


# ---------------------------------------------------------------------------
# Fake Groq client
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class FakeGroqClient:
    """Stand-in for groq.Groq — records calls, returns a controllable reply."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._content: str | None = "Fake grounded answer. [Source 1]"
        self._error: Exception | None = None

    def set_content(self, content: str | None) -> None:
        self._content = content
        self._error = None

    def set_error(self, exc: Exception) -> None:
        self._error = exc

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._content)

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


@pytest.fixture
def fake_groq(monkeypatch):
    from src.generation import llm_service

    client = FakeGroqClient()
    monkeypatch.setattr(llm_service, "_get_client", lambda: client)
    return client


# ---------------------------------------------------------------------------
# Mocked HTTP + DNS for url_loader tests (no real network)
# ---------------------------------------------------------------------------


class FakeHTTPResponse:
    def __init__(self, content: bytes, content_type: str = "text/html", status_code: int = 200) -> None:
        self._content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)

    def iter_content(self, chunk_size: int = 8192):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def close(self) -> None:
        pass


@pytest.fixture
def fake_requests_get(monkeypatch):
    from src.ingestion import url_loader

    state = SimpleNamespace(response=None, exception=None)

    def _get(url, **kwargs):
        if state.exception is not None:
            raise state.exception
        return state.response

    monkeypatch.setattr(url_loader.requests, "get", _get)
    return state


@pytest.fixture
def fake_dns(monkeypatch):
    """Default to a public IP so validate_url() passes unless a test overrides it."""
    from src.ingestion import url_loader

    state = SimpleNamespace(ip="93.184.216.34")

    def _resolve(hostname):
        return state.ip

    monkeypatch.setattr(url_loader.socket, "gethostbyname", _resolve)
    return state
