"""Typed data structures shared across the ingestion, retrieval, and generation layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceType(str, Enum):
    URL = "url"
    FILE = "file"


@dataclass
class Chunk:
    """A single piece of source content, ready to be embedded."""

    text: str
    source: str  # URL or original filename
    title: str  # page title or filename, used for citations
    source_type: SourceType
    chunk_index: int = 0


@dataclass
class RetrievedChunk:
    """A chunk returned from vector search, with its similarity score."""

    text: str
    source: str
    title: str
    source_type: SourceType
    chunk_index: int
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity in [0, 1], derived from Chroma's cosine distance."""
        return max(0.0, 1 - self.distance)


@dataclass
class IngestOutcome:
    """Result of ingesting a single source (URL or file)."""

    source: str
    success: bool
    chunk_count: int = 0
    error: str | None = None


@dataclass
class AnswerResult:
    """The final answer plus the sources used to ground it."""

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
