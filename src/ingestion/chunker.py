"""Splits cleaned text into overlapping chunks ready for embedding."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.models.schemas import Chunk, SourceType


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks, trying paragraph/sentence boundaries first."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def build_chunks(text: str, source: str, title: str, source_type: SourceType) -> list[Chunk]:
    """Split text and wrap each piece as a Chunk with source metadata."""
    pieces = split_text(text)
    return [
        Chunk(text=piece, source=source, title=title, source_type=source_type, chunk_index=i)
        for i, piece in enumerate(pieces)
    ]
