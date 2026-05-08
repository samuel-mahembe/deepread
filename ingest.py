"""
ingest.py — Build the knowledge base.

Run this once (or whenever you want to refresh the docs):
    python ingest.py

This is Phase 1 of RAG: load documents -> chunk -> embed -> store.
"""

from rag.chunker import chunk_url
from rag.retriever import index_chunks


# The documentation pages we want to index.
# Start small — add more after you've verified the pipeline works.
URLS = [
    "https://requests.readthedocs.io/en/latest/user/quickstart/",
    "https://requests.readthedocs.io/en/latest/user/advanced/",
]


def main():
    all_chunks = []
    
    for url in URLS:
        print(f"Fetching {url}...")
        try:
            chunks = chunk_url(url)
            print(f"  -> Got {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  -> Failed: {e}")
    
    if not all_chunks:
        print("No chunks to index. Exiting.")
        return
    
    print(f"\nIndexing {len(all_chunks)} chunks into ChromaDB...")
    index_chunks(all_chunks)
    print("Done!")


if __name__ == "__main__":
    print("Starting ingestion process...")
    main()