"""
app.py — Streamlit UI for AskMyDocs (dynamic ingestion).

Users provide their own URLs, ingestion runs, then they can ask questions.
"""

import uuid
import streamlit as st
from rag.chunker import chunk_url
from rag.retriever import index_chunks, search, clear_session
from rag.generator import generate_answer, suggest_questions, get_sample_chunks


st.set_page_config(page_title="DeepRead", page_icon="📚", layout="centered")


# Each user gets a unique session_id — used to scope their ChromaDB collection
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Track app state: "ingest" (waiting for URLs) or "ask" (ready to answer)
if "stage" not in st.session_state:
    st.session_state.stage = "ingest"

if "ingested_urls" not in st.session_state:
    st.session_state.ingested_urls = []

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "question" not in st.session_state:
    st.session_state.question = ""


# Header
st.title("📚 DeepRead")
st.markdown("Paste URLs, ask questions. Powered by RAG.")
st.markdown(
    "[GitHub](https://github.com/YOUR_USERNAME/deepread) · "
    "Built with sentence-transformers + ChromaDB + Groq"
)
st.divider()


# ---------- STAGE 1: INGEST ----------
if st.session_state.stage == "ingest":
    st.markdown("### Step 1: Add your sources")
    st.caption("Paste one URL per line. We'll fetch and index them.")
    
    urls_text = st.text_area(
        "URLs",
        height=150,
        placeholder="https://requests.readthedocs.io/en/latest/user/quickstart/\nhttps://requests.readthedocs.io/en/latest/user/advanced/",
        label_visibility="collapsed",
    )
    
    if st.button("Ingest URLs", type="primary", use_container_width=True):
        urls = [u.strip() for u in urls_text.split("\n") if u.strip()]
        
        if not urls:
            st.error("Please paste at least one URL.")
        else:
            all_chunks = []
            progress = st.progress(0.0, text="Starting...")
            
            for i, url in enumerate(urls):
                progress.progress(
                    i / len(urls),
                    text=f"Fetching {url[:60]}..."
                )
                try:
                    chunks = chunk_url(url)
                    all_chunks.extend(chunks)
                    st.session_state.ingested_urls.append(url)
                except Exception as e:
                    st.warning(f"Failed to fetch {url}: {e}")
            
            if all_chunks:
                progress.progress(0.8, text=f"Embedding {len(all_chunks)} chunks...")
                index_chunks(all_chunks, st.session_state.session_id)
                
                progress.progress(0.95, text="Generating suggested questions...")
                sample = get_sample_chunks(st.session_state.session_id)
                st.session_state.suggested_questions = suggest_questions(sample)
                
                progress.progress(1.0, text="Done!")
                st.session_state.stage = "ask"
                st.rerun()
            else:
                st.error("No content could be fetched from those URLs.")


# ---------- STAGE 2: ASK ----------
elif st.session_state.stage == "ask":
    # Show what's been ingested + option to start over
    with st.expander(f"📥 Ingested {len(st.session_state.ingested_urls)} source(s)"):
        for url in st.session_state.ingested_urls:
            st.markdown(f"- {url}")
        
        if st.button("Start over with new URLs"):
            clear_session(st.session_state.session_id)
            st.session_state.stage = "ingest"
            st.session_state.ingested_urls = []
            st.session_state.suggested_questions = []
            st.session_state.question = ""
            st.rerun()
    
    # Suggested questions
    if st.session_state.suggested_questions:
        st.markdown("**Suggested questions:**")
        cols = st.columns(2)
        for i, q in enumerate(st.session_state.suggested_questions):
            if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.question = q
                st.rerun()
    
    # Question input — Streamlit forms give you Enter-to-submit for free
    with st.form(key="ask_form", clear_on_submit=False):
        question = st.text_input(
            "Ask a question:",
            value=st.session_state.question,
            placeholder="What does the documentation say about...?",
        )
        col1, col2 = st.columns([1, 5])
        submit = col1.form_submit_button("Ask", type="primary")
        clear = col2.form_submit_button("Clear")
    
    if clear:
        st.session_state.question = ""
        st.rerun()
    
    if submit and question:
        st.session_state.question = question
        
        with st.spinner("Searching..."):
            chunks = search(question, st.session_state.session_id, top_k=4)
        
        with st.spinner("Generating answer..."):
            answer = generate_answer(question, chunks)
        
        st.markdown("### Answer")
        st.markdown(answer)
        
        st.markdown("### Sources")
        st.caption(f"Retrieved {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks, start=1):
            with st.expander(
                f"Source {i}: {chunk['title']} "
                f"(similarity: {1 - chunk['distance']:.2%})"
            ):
                st.markdown(f"**URL:** {chunk['source_url']}")
                st.text(chunk['text'])