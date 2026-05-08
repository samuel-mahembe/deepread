"""
    Loads documentation and splits it into chunks.
    Runs once during ingestion.
"""

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single chunk of documentation, ready to be embedded."""
    text: str          # content
    source_url: str    # original citation
    title: str         # page title for citations


def load_page(url: str) -> tuple[str, str]:
    """Fetch a documentation page and return (title, clean_text)."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Strip out junk that pollutes the text
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    
    title = soup.title.string if soup.title else url
    text = soup.get_text(separator="\n", strip=True)
    
    return title, text


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks recursively."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try these in order
    )
    return splitter.split_text(text)


def chunk_url(url: str) -> list[Chunk]:
    """End-to-end: fetch a URL and return a list of Chunk objects."""
    title, text = load_page(url)
    pieces = split_text(text)
    return [Chunk(text=piece, source_url=url, title=title) for piece in pieces]

