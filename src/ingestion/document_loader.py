"""
Loads user-uploaded documents (PDF, DOCX, TXT, MD).

Accepts anything with a Streamlit ``UploadedFile``-like interface (a
``.name`` attribute and a ``.read()``/``.getvalue()`` method) so this module
has no hard dependency on Streamlit and can be unit tested with plain
``io.BytesIO`` objects.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

from src.config.settings import settings
from src.ingestion.errors import ContentTooLargeError, InvalidFileError, UnsupportedContentError


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def validate_file(file: UploadedFileLike) -> None:
    """Raise InvalidFileError if the extension or size is not allowed."""
    extension = Path(file.name).suffix.lower()
    if extension not in settings.allowed_file_extensions:
        raise InvalidFileError(
            f"Unsupported file type '{extension}'. "
            f"Allowed types: {', '.join(settings.allowed_file_extensions)}."
        )

    size = len(file.getvalue())
    if size == 0:
        raise InvalidFileError(f"'{file.name}' is empty.")
    if size > settings.max_file_size_bytes:
        raise ContentTooLargeError(
            f"'{file.name}' is {size / 1_000_000:.1f}MB, "
            f"which exceeds the {settings.max_file_size_bytes // 1_000_000}MB limit."
        )


def _extract_pdf(data: bytes, filename: str) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise UnsupportedContentError(f"Could not parse '{filename}' as a PDF: {exc}") from exc
    return "\n\n".join(pages)


def _extract_docx(data: bytes, filename: str) -> str:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises bare package/XML errors
        raise UnsupportedContentError(f"Could not parse '{filename}' as a DOCX file: {exc}") from exc
    return "\n".join(p.text for p in document.paragraphs)


def _extract_text(data: bytes, filename: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedContentError(f"'{filename}' is not valid UTF-8 text.") from exc


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_text,
    ".md": _extract_text,
}


def load_document(file: UploadedFileLike) -> tuple[str, str]:
    """Validate and extract text from an uploaded document. Returns (title, text)."""
    validate_file(file)
    extension = Path(file.name).suffix.lower()
    text = _EXTRACTORS[extension](file.getvalue(), file.name)
    return file.name, text
