"""
Vector storage, backed by ChromaDB.

Why Chroma for this project: it's an embedded, file-backed vector database
with no separate service to run or pay for, which matches a single-process
Streamlit deployment exactly. The trade-off is that it doesn't horizontally
scale or support concurrent writers across machines — irrelevant at this
project's scale (one small collection per user session), but the reason a
managed vector DB (Pinecone, Weaviate, pgvector) isn't used here.

This module is the only place that imports ``chromadb`` directly. Every
other layer calls the functions below, so swapping the backend later means
rewriting this one file, not the rest of the codebase.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb

from src.config.settings import settings
from src.models.schemas import Chunk, RetrievedChunk, SourceType


@lru_cache(maxsize=1)
def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def _collection_name(collection_id: str) -> str:
    return f"session_{collection_id}"


def _get_collection(collection_id: str):
    client = _get_client()
    return client.get_or_create_collection(
        name=_collection_name(collection_id),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(collection_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Embed and store chunks in the given collection. IDs are content-hash based to dedupe."""
    if not chunks:
        return
    collection = _get_collection(collection_id)

    ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "source": c.source,
            "title": c.title,
            "source_type": c.source_type.value,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    # upsert (not add) so re-ingesting the same source overwrites rather than duplicates
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def query(collection_id: str, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
    collection = _get_collection(collection_id)
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    return [
        RetrievedChunk(
            text=doc,
            source=meta["source"],
            title=meta["title"],
            source_type=SourceType(meta["source_type"]),
            chunk_index=meta["chunk_index"],
            distance=dist,
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def peek(collection_id: str, n: int) -> list[RetrievedChunk]:
    """Return up to n arbitrary chunks from the collection (used for question suggestions)."""
    collection = _get_collection(collection_id)
    results = collection.peek(limit=n)
    return [
        RetrievedChunk(
            text=doc,
            source=meta["source"],
            title=meta["title"],
            source_type=SourceType(meta["source_type"]),
            chunk_index=meta["chunk_index"],
            distance=0.0,
        )
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]


def count(collection_id: str) -> int:
    return _get_collection(collection_id).count()


def delete_collection(collection_id: str) -> None:
    client = _get_client()
    try:
        client.delete_collection(name=_collection_name(collection_id))
    except Exception:
        pass  # collection didn't exist — nothing to do
