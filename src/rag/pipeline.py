"""
The RAG pipeline: the single orchestration layer between the Streamlit UI
and every lower-level module (ingestion, embeddings, vector store,
retrieval, generation).

app.py should only ever call functions in this file — it should never
import from src.ingestion, src.embeddings, src.vectorstore, src.retrieval,
or src.generation directly. That boundary is what keeps the UI thin and
the pipeline independently testable without Streamlit installed.
"""

from __future__ import annotations

from src.embeddings.embedding_service import embed
from src.generation.llm_service import generate_answer, suggest_questions
from src.ingestion import document_loader, url_loader
from src.ingestion.chunker import build_chunks
from src.ingestion.cleaner import clean_text, ensure_substantial
from src.ingestion.errors import IngestionError
from src.models.schemas import AnswerResult, IngestOutcome, SourceType
from src.retrieval.retriever import retrieve
from src.vectorstore import chroma_store


def _ingest_text(raw_title: str, raw_text: str, source: str, source_type: SourceType, collection_id: str) -> int:
    text = clean_text(raw_text)
    ensure_substantial(text, source)

    chunks = build_chunks(text, source=source, title=raw_title, source_type=source_type)
    embeddings = embed([c.text for c in chunks])
    chroma_store.add_chunks(collection_id, chunks, embeddings)
    return len(chunks)


def ingest_url(url: str, collection_id: str) -> IngestOutcome:
    try:
        title, raw_text = url_loader.load_page(url)
        n = _ingest_text(title, raw_text, source=url, source_type=SourceType.URL, collection_id=collection_id)
        return IngestOutcome(source=url, success=True, chunk_count=n)
    except IngestionError as exc:
        return IngestOutcome(source=url, success=False, error=str(exc))


def ingest_urls(urls: list[str], collection_id: str) -> list[IngestOutcome]:
    return [ingest_url(url, collection_id) for url in urls]


def ingest_file(file: document_loader.UploadedFileLike, collection_id: str) -> IngestOutcome:
    try:
        title, raw_text = document_loader.load_document(file)
        n = _ingest_text(title, raw_text, source=file.name, source_type=SourceType.FILE, collection_id=collection_id)
        return IngestOutcome(source=file.name, success=True, chunk_count=n)
    except IngestionError as exc:
        return IngestOutcome(source=file.name, success=False, error=str(exc))


def ingest_files(files: list[document_loader.UploadedFileLike], collection_id: str) -> list[IngestOutcome]:
    return [ingest_file(f, collection_id) for f in files]


def ask(question: str, collection_id: str, top_k: int | None = None) -> AnswerResult:
    chunks = retrieve(question, collection_id, top_k=top_k)
    answer = generate_answer(question, chunks)
    return AnswerResult(answer=answer, sources=chunks)


def suggest_questions_for_session(collection_id: str, n: int = 4) -> list[str]:
    sample = chroma_store.peek(collection_id, n=8)
    return suggest_questions(sample, n=n)


def clear_session(collection_id: str) -> None:
    chroma_store.delete_collection(collection_id)


def has_content(collection_id: str) -> bool:
    return chroma_store.count(collection_id) > 0
