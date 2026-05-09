"""
    Generate answers from retrieved chunks using an LLM.
    It runs every time a user asks a question,
    after retriever.py has found the relevant chunks.
"""

import os
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


# Initialize the Groq client once at module load
def _get_api_key():
    """Get Groq API key from Streamlit secrets (cloud) or env (local)."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        return os.environ["GROQ_API_KEY"]

_client = Groq(api_key=_get_api_key())

# The model — Groq offers Llama 3.3 70B free and very fast
MODEL = "llama-3.3-70b-versatile"

# This prompt template is the heart of RAG
PROMPT_TEMPLATE = """You are a helpful assistant answering questions about Python documentation.
    Use ONLY the context below to answer the question. If the answer isn't in the context, say "I don't have enough information in the docs to answer that."
    Cite which source(s) you used by referencing the title.
    Context:
    {context}
    Question: {question}
    Answer:"""


def format_context(chunks: list[dict]) -> str:
    """Combine retrieved chunks into a single context string with source labels."""
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        formatted.append(
            f"[Source {i}: {chunk['title']}]\n{chunk['text']}\n"
        )
    return "\n".join(formatted)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate an answer to the question using the retrieved chunks."""
    context = format_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # Low = more factual, less creative
        max_tokens=500,
    )
    
    return response.choices[0].message.content

def suggest_questions(chunks: list[dict], n: int = 4) -> list[str]:
    """Generate suggested questions based on ingested content."""
    # Sample some chunks to give the LLM a feel for the content
    sample_text = "\n\n".join(c["text"][:300] for c in chunks[:8])
    
    prompt = f"""Based on the following content excerpts, suggest {n} concise, useful questions a user might ask about this material.

Return ONLY the questions, one per line, no numbering, no extra text.

Content:
{sample_text}

Questions:"""
    
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300,
    )
    
    raw = response.choices[0].message.content
    questions = [q.strip() for q in raw.split("\n") if q.strip()]
    # Strip leading bullets/numbers if the LLM ignored instructions
    questions = [q.lstrip("0123456789.-) ").strip() for q in questions]
    return questions[:n]


def get_sample_chunks(session_id: str, n: int = 8) -> list[dict]:
    """Pull a sample of chunks from the session — used for question suggestions."""
    from rag.retriever import _get_collection
    collection = _get_collection(session_id)
    results = collection.peek(limit=n)
    return [
        {"text": doc, "source_url": meta["source_url"], "title": meta["title"]}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]