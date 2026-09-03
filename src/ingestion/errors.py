"""Exceptions raised by the ingestion layer.

Kept as a single small module so callers (the pipeline, the Streamlit UI)
can catch one family of errors and show a clean, user-facing message
instead of leaking raw exception internals.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all recoverable ingestion failures."""


class InvalidURLError(IngestionError):
    """The URL is malformed, uses a disallowed scheme, or targets a blocked host."""


class FetchError(IngestionError):
    """The URL could not be fetched (timeout, connection error, non-2xx status)."""


class ContentTooLargeError(IngestionError):
    """The remote response (or uploaded file) exceeds the configured size limit."""


class UnsupportedContentError(IngestionError):
    """The fetched content isn't a type we know how to parse (or is effectively empty)."""


class InvalidFileError(IngestionError):
    """The uploaded file failed validation (extension, size, or content)."""
