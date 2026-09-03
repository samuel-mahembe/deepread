"""
Ingestion-layer tests: HTML cleaning/extraction, document text extraction,
supported/unsupported file types, empty/invalid content, and the URL
content-size limit. SSRF/URL-validation and file-validation tests live in
test_security.py.
"""

from __future__ import annotations

import io

import pytest

from src.config.settings import Settings
from src.ingestion import document_loader, url_loader
from src.ingestion.cleaner import clean_text, ensure_substantial
from src.ingestion.errors import ContentTooLargeError, InvalidFileError, UnsupportedContentError

# ---------------------------------------------------------------------------
# cleaner.py
# ---------------------------------------------------------------------------


def test_clean_text_collapses_repeated_spaces_and_tabs():
    assert clean_text("a   b\t\tc") == "a b c"


def test_clean_text_collapses_excess_blank_lines():
    assert clean_text("para one\n\n\n\n\npara two") == "para one\n\npara two"


def test_clean_text_strips_leading_and_trailing_whitespace():
    assert clean_text("  \n  hello world  \n  ") == "hello world"


def test_ensure_substantial_passes_for_long_enough_text():
    text = "x" * 500
    ensure_substantial(text, source="fixture")  # should not raise


def test_ensure_substantial_raises_for_near_empty_text():
    with pytest.raises(UnsupportedContentError):
        ensure_substantial("too short", source="fixture.txt")


def test_ensure_substantial_error_names_the_source():
    with pytest.raises(UnsupportedContentError, match="fixture.txt"):
        ensure_substantial("", source="fixture.txt")


# ---------------------------------------------------------------------------
# document_loader.py — supported types
# ---------------------------------------------------------------------------


def test_load_document_extracts_txt(make_uploaded_file):
    f = make_uploaded_file("notes.txt", "Plain text content for the test suite.")
    title, text = document_loader.load_document(f)
    assert title == "notes.txt"
    assert "Plain text content" in text


def test_load_document_extracts_md(make_uploaded_file):
    f = make_uploaded_file("notes.md", "# Heading\n\nSome **markdown** body text.")
    title, text = document_loader.load_document(f)
    assert title == "notes.md"
    assert "markdown" in text


def test_load_document_extracts_docx(make_uploaded_file):
    docx = pytest.importorskip("docx")
    buf = io.BytesIO()
    doc = docx.Document()
    doc.add_paragraph("First paragraph of a real DOCX fixture.")
    doc.add_paragraph("Second paragraph with different content.")
    doc.save(buf)

    f = make_uploaded_file("report.docx", buf.getvalue())
    title, text = document_loader.load_document(f)
    assert title == "report.docx"
    assert "First paragraph" in text
    assert "Second paragraph" in text


def test_load_document_rejects_malformed_pdf(make_uploaded_file):
    f = make_uploaded_file("broken.pdf", b"this is not a real pdf file")
    with pytest.raises(UnsupportedContentError):
        document_loader.load_document(f)


def test_load_document_parses_a_structurally_valid_pdf(make_uploaded_file):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buf)

    f = make_uploaded_file("blank.pdf", buf.getvalue())
    title, text = document_loader.load_document(f)

    # A structurally valid PDF with no text layer parses without error here;
    # ensure_substantial() further up the pipeline is what rejects it as
    # too-empty-to-be-useful, not load_document itself.
    assert title == "blank.pdf"
    assert text == ""


def test_load_document_rejects_malformed_docx(make_uploaded_file):
    pytest.importorskip("docx")
    f = make_uploaded_file("broken.docx", b"not a real docx file at all")
    with pytest.raises(UnsupportedContentError):
        document_loader.load_document(f)


def test_load_document_rejects_non_utf8_text(make_uploaded_file):
    f = make_uploaded_file("bad.txt", b"\xff\xfe\x00\x01not utf8")
    with pytest.raises(UnsupportedContentError):
        document_loader.load_document(f)


# ---------------------------------------------------------------------------
# document_loader.py — unsupported / invalid input
# ---------------------------------------------------------------------------


def test_validate_file_rejects_unsupported_extension(make_uploaded_file):
    f = make_uploaded_file("archive.zip", b"PK\x03\x04fakezipdata")
    with pytest.raises(InvalidFileError, match="Unsupported file type"):
        document_loader.validate_file(f)


def test_validate_file_rejects_empty_file(make_uploaded_file):
    f = make_uploaded_file("empty.txt", b"")
    with pytest.raises(InvalidFileError, match="empty"):
        document_loader.validate_file(f)


def test_validate_file_rejects_oversized_file(make_uploaded_file, monkeypatch):
    small_limit_settings = Settings(max_file_size_bytes=10)
    monkeypatch.setattr(document_loader, "settings", small_limit_settings)

    f = make_uploaded_file("big.txt", b"way more than ten bytes of content")
    with pytest.raises(ContentTooLargeError):
        document_loader.validate_file(f)


def test_validate_file_accepts_supported_type_within_limits(make_uploaded_file):
    f = make_uploaded_file("ok.txt", b"short but valid content")
    document_loader.validate_file(f)  # should not raise


# ---------------------------------------------------------------------------
# url_loader.py — HTML cleaning/extraction (network mocked via fixtures)
# ---------------------------------------------------------------------------


def test_load_page_strips_script_style_nav_and_footer(fake_requests_get, fake_dns, sample_html_bytes):
    from tests.conftest import FakeHTTPResponse

    fake_requests_get.response = FakeHTTPResponse(sample_html_bytes, content_type="text/html; charset=utf-8")

    title, text = url_loader.load_page("https://example.com/article")

    assert title == "Fixture Page — Chunking and Retrieval"
    assert "tracking pixel nonsense" not in text  # <script> stripped
    assert "font-family" not in text  # <style> stripped
    assert "Home" not in text  # <nav> stripped
    assert "Copyright fixture footer" not in text  # <footer> stripped
    assert "Cosine similarity is used to rank" in text  # real body content kept


def test_load_page_rejects_non_html_content_type(fake_requests_get, fake_dns):
    from tests.conftest import FakeHTTPResponse

    fake_requests_get.response = FakeHTTPResponse(b"%PDF-1.4 fake pdf bytes", content_type="application/pdf")

    with pytest.raises(url_loader.FetchError, match="unsupported content type"):
        url_loader.load_page("https://example.com/file.pdf")


def test_load_page_raises_fetch_error_on_http_error_status(fake_requests_get, fake_dns):
    from tests.conftest import FakeHTTPResponse

    fake_requests_get.response = FakeHTTPResponse(b"not found", status_code=404)

    with pytest.raises(url_loader.FetchError, match="404"):
        url_loader.load_page("https://example.com/missing")


def test_fetch_raw_wraps_timeout_as_fetch_error(fake_requests_get, fake_dns):
    import requests

    fake_requests_get.exception = requests.exceptions.Timeout("took too long")

    with pytest.raises(url_loader.FetchError, match="timed out"):
        url_loader.load_page("https://example.com/slow")


def test_fetch_raw_wraps_connection_errors_as_fetch_error(fake_requests_get, fake_dns):
    import requests

    fake_requests_get.exception = requests.exceptions.ConnectionError("connection refused")

    with pytest.raises(url_loader.FetchError, match="Could not fetch"):
        url_loader.load_page("https://example.com/unreachable")


def test_fetch_raw_enforces_content_size_limit(fake_requests_get, fake_dns, monkeypatch):
    from tests.conftest import FakeHTTPResponse

    tight_settings = Settings(max_url_content_bytes=100)
    monkeypatch.setattr(url_loader, "settings", tight_settings)

    fake_requests_get.response = FakeHTTPResponse(b"x" * 1000, content_type="text/html")

    with pytest.raises(ContentTooLargeError):
        url_loader.load_page("https://example.com/huge-page")
