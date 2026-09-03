"""
Fetches web pages for ingestion.

Two things this module is deliberately careful about, because it's the one
place in the app that makes outbound requests to attacker-influenced URLs:

1. SSRF: before fetching, the hostname is resolved and checked against
   private/loopback/link-local ranges so the app can't be used as a proxy
   to reach internal network services.
2. Resource exhaustion: responses are streamed and cut off once they exceed
   ``settings.max_url_content_bytes``, so a single huge or slow-drip page
   can't tie up memory/time on a shared instance.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.ingestion.errors import ContentTooLargeError, FetchError, InvalidURLError

_BLOCKED_HOSTNAMES = {"localhost"}


def validate_url(url: str) -> None:
    """Raise InvalidURLError if the URL is malformed or targets a blocked host."""
    parsed = urlparse(url.strip())

    if parsed.scheme not in settings.allowed_url_schemes:
        raise InvalidURLError(
            f"Unsupported URL scheme '{parsed.scheme or '(none)'}'. "
            f"Use one of: {', '.join(settings.allowed_url_schemes)}."
        )

    hostname = parsed.hostname
    if not hostname:
        raise InvalidURLError("URL has no hostname.")

    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise InvalidURLError("Requests to local/internal hosts are not allowed.")

    try:
        resolved_ip = socket.gethostbyname(hostname)
    except socket.gaierror as exc:
        raise InvalidURLError(f"Could not resolve host '{hostname}'.") from exc

    ip = ipaddress.ip_address(resolved_ip)
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        raise InvalidURLError("Requests to private/internal network addresses are not allowed.")


def _fetch_raw(url: str) -> tuple[str, bytes]:
    """Stream the response, enforcing the max content size. Returns (content_type, body)."""
    try:
        response = requests.get(
            url,
            timeout=settings.request_timeout_seconds,
            stream=True,
            headers={"User-Agent": "DeepRead/1.0 (+https://github.com/samuel-mahembe/deepread)"},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise FetchError(f"Request to {url} timed out after {settings.request_timeout_seconds}s.") from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise FetchError(f"{url} returned HTTP {status}.") from exc
    except requests.exceptions.RequestException as exc:
        raise FetchError(f"Could not fetch {url}: {exc.__class__.__name__}.") from exc

    content_type = response.headers.get("Content-Type", "")

    body = bytearray()
    limit = settings.max_url_content_bytes
    for piece in response.iter_content(chunk_size=8192):
        body.extend(piece)
        if len(body) > limit:
            response.close()
            raise ContentTooLargeError(
                f"{url} exceeds the {limit // 1_000_000}MB content limit."
            )

    return content_type, bytes(body)


def load_page(url: str) -> tuple[str, str]:
    """Fetch a web page and return (title, clean_text). Validates and size-caps first."""
    validate_url(url)
    content_type, body = _fetch_raw(url)

    if "html" not in content_type and content_type != "":
        raise FetchError(
            f"{url} returned unsupported content type '{content_type}'. "
            "Only HTML pages are supported for URL ingestion."
        )

    soup = BeautifulSoup(body, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text = soup.get_text(separator="\n", strip=True)

    return title, text
