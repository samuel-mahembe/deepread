"""
Security/validation tests: SSRF guards on URL ingestion, file validation,
and the deterministic prompt-injection mitigation in the system prompt.
"""

from __future__ import annotations

import pytest

from src.generation.prompts import SYSTEM_PROMPT
from src.ingestion import document_loader, url_loader
from src.ingestion.errors import InvalidFileError, InvalidURLError

# ---------------------------------------------------------------------------
# URL validation — scheme / hostname
# ---------------------------------------------------------------------------


def test_validate_url_accepts_https(fake_dns):
    url_loader.validate_url("https://example.com/docs")  # should not raise


def test_validate_url_accepts_http(fake_dns):
    url_loader.validate_url("http://example.com/docs")  # should not raise


@pytest.mark.parametrize("url", ["ftp://example.com/file", "file:///etc/passwd", "javascript:alert(1)"])
def test_validate_url_rejects_disallowed_schemes(url, fake_dns):
    with pytest.raises(InvalidURLError, match="scheme"):
        url_loader.validate_url(url)


def test_validate_url_rejects_missing_hostname(fake_dns):
    with pytest.raises(InvalidURLError, match="hostname"):
        url_loader.validate_url("https:///no-host-here")


def test_validate_url_rejects_localhost(fake_dns):
    with pytest.raises(InvalidURLError, match="local/internal"):
        url_loader.validate_url("http://localhost:8000/admin")


def test_validate_url_rejects_unresolvable_hostname(monkeypatch):
    import socket

    def _raise(hostname):
        raise socket.gaierror("name resolution failed")

    monkeypatch.setattr(url_loader.socket, "gethostbyname", _raise)

    with pytest.raises(InvalidURLError, match="resolve"):
        url_loader.validate_url("https://this-domain-does-not-exist.invalid")


# ---------------------------------------------------------------------------
# URL validation — SSRF via private/internal IP ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",  # loopback
        "10.0.0.5",  # private
        "192.168.1.1",  # private
        "172.16.0.1",  # private
        "169.254.1.1",  # link-local (cloud metadata endpoint range)
        "0.0.0.0",  # reserved
        "224.0.0.1",  # multicast
    ],
)
def test_validate_url_rejects_urls_resolving_to_internal_ips(ip, fake_dns):
    fake_dns.ip = ip
    with pytest.raises(InvalidURLError, match="private/internal"):
        url_loader.validate_url("https://looks-public-but-is-not.example.com")


def test_validate_url_accepts_url_resolving_to_public_ip(fake_dns):
    fake_dns.ip = "93.184.216.34"  # a public IP
    url_loader.validate_url("https://looks-public.example.com")  # should not raise


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------


def test_validate_file_rejects_disallowed_extension(make_uploaded_file):
    f = make_uploaded_file("payload.exe", b"MZ\x90\x00fake-executable")
    with pytest.raises(InvalidFileError):
        document_loader.validate_file(f)


def test_validate_file_rejects_path_traversal_style_name(make_uploaded_file):
    """Extension is still validated even if the filename looks like a path."""
    f = make_uploaded_file("../../etc/passwd.exe", b"data")
    with pytest.raises(InvalidFileError):
        document_loader.validate_file(f)


def test_validate_file_accepts_traversal_style_name_with_allowed_extension(make_uploaded_file):
    """validate_file only checks extension/size; it never touches the filesystem,
    so a path-traversal-looking name with an allowed extension is not itself a
    security issue here — the file is never opened by that name."""
    f = make_uploaded_file("../../etc/notes.txt", b"just some text content")
    document_loader.validate_file(f)  # should not raise


# ---------------------------------------------------------------------------
# Prompt-injection mitigation (deterministic — checks the actual prompt text)
# ---------------------------------------------------------------------------


def test_system_prompt_instructs_model_to_ignore_instructions_in_context():
    lowered = SYSTEM_PROMPT.lower()
    assert "never as instructions" in lowered or "not as instructions" in lowered


def test_system_prompt_restricts_answers_to_provided_context():
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered and "context" in lowered


def test_system_prompt_requires_citations():
    assert "cite" in SYSTEM_PROMPT.lower()
