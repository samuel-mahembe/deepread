"""
    Embedding generation and vector search.
    Functions :
        1. Embed text using sentence-transformers (turns text into 384-dim vectors)
        2. Store and query embeddings in ChromaDB (Vector DB)
"""

import chromadb
from sentence_transformers import SentenceTransformer
from rag.chunker import Chunk

# Where ChromaDB persists its files
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "askmydocs"

# Load embedding model once when this module is imported
# all-MiniLM-L6-v2: small (90MB), fast, good quality, 384 dimensions
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _get_collection(session_id: str):
    """Get (or create) the Chroma collection for this user session."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_name = f"session_{session_id}"
    return client.get_or_create_collection(name=collection_name)


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into a list of 384-dim embedding vectors."""
    return _model.encode(texts, show_progress_bar=False).tolist()


def index_chunks(chunks: list[Chunk], session_id: str) -> None:
    collection = _get_collection(session_id)
    existing_count = collection.count()
    
    texts = [c.text for c in chunks]
    embeddings = embed(texts)
    ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]
    metadatas = [
        {"source_url": c.source_url, "title": c.title}
        for c in chunks
    ]
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

def search(query: str, session_id: str, top_k: int = 4) -> list[dict]:
    collection = _get_collection(session_id)
    query_embedding = embed([query])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    
    return [
        {"text": doc, "source_url": meta["source_url"], 
         "title": meta["title"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

def clear_session(session_id: str) -> None:
    """Delete this user's collection. Useful for 'start over'."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(name=f"session_{session_id}")
    except Exception:
        pass  # Collection didn't exist