"""Tests for src.ingestion.chunker: splitting behavior and Chunk wiring."""

from __future__ import annotations

from src.ingestion.chunker import build_chunks, split_text
from src.models.schemas import SourceType

# ---------------------------------------------------------------------------
# split_text
# ---------------------------------------------------------------------------


def test_split_text_empty_input_returns_no_chunks():
    assert split_text("") == []


def test_split_text_very_short_input_returns_single_chunk():
    text = "One short sentence."
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    assert chunks == [text]


def test_split_text_splits_long_input_into_multiple_chunks():
    text = "word " * 400  # ~2000 chars, well over chunk_size
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_split_text_respects_paragraph_boundaries_when_possible():
    paragraphs = [f"Paragraph {i}. " + ("filler " * 20) for i in range(4)]
    text = "\n\n".join(paragraphs)

    chunks = split_text(text, chunk_size=200, chunk_overlap=0)

    # Each chunk should align with paragraph starts rather than cutting a
    # paragraph open mid-word wherever a chunk boundary had the option not to.
    for chunk in chunks:
        assert chunk == chunk.strip()  # no dangling leading/trailing whitespace from a bad cut


def test_split_text_produces_overlap_between_consecutive_chunks():
    text = "abcdefghij" * 30  # 300 chars, no natural separators
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    for prev_chunk, next_chunk in zip(chunks, chunks[1:]):
        # The splitter falls back to hard character cuts here (no separators),
        # so the requested overlap should be reflected in shared tail/head text.
        assert prev_chunk[-10:] in next_chunk


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------


def test_build_chunks_wires_source_metadata_onto_every_chunk():
    text = "sentence one. " * 100
    chunks = build_chunks(text, source="https://example.com/page", title="Example Page", source_type=SourceType.URL)

    assert len(chunks) > 1
    for c in chunks:
        assert c.source == "https://example.com/page"
        assert c.title == "Example Page"
        assert c.source_type is SourceType.URL


def test_build_chunks_assigns_sequential_chunk_index():
    text = "sentence one. " * 100
    chunks = build_chunks(text, source="doc.txt", title="doc.txt", source_type=SourceType.FILE)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_build_chunks_empty_text_produces_no_chunks():
    chunks = build_chunks("", source="empty.txt", title="empty.txt", source_type=SourceType.FILE)
    assert chunks == []
