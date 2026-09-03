"""Text normalization applied to extracted content before chunking."""

from __future__ import annotations

import re

from src.config.settings import settings
from src.ingestion.errors import UnsupportedContentError


def clean_text(text: str) -> str:
    """Collapse excess whitespace/blank lines left over from HTML or PDF extraction."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ensure_substantial(text: str, source: str) -> None:
    """Raise if the cleaned text looks empty/near-empty (e.g. a JS-rendered page).

    Fetching such a page "succeeds" at the HTTP layer but produces no usable
    content, which previously failed silently and degraded answer quality
    with no user-facing warning.
    """
    if len(text) < settings.min_extracted_chars:
        raise UnsupportedContentError(
            f"'{source}' produced almost no readable text "
            f"({len(text)} chars). It may require JavaScript to render, "
            "or the page structure isn't supported."
        )
