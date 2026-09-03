"""
Streamlit UI for DeepRead.

This file only renders widgets, manages session state, and calls into
src.rag.pipeline — it contains no ingestion, retrieval, or generation logic
itself. That logic lives in src/ and is unit/integration tested independently
of Streamlit.
"""

from __future__ import annotations

import sys

try:
    # Streamlit Community Cloud's base image ships a system sqlite3 too old
    # for ChromaDB. pysqlite3-binary bundles a modern build to swap in — but
    # it only publishes Linux wheels, so this is a no-op on local Windows/Mac
    # dev, where the system sqlite3 is already new enough.
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import uuid

import streamlit as st

from src.models.schemas import SourceType
from src.rag import pipeline

st.set_page_config(page_title="DeepRead", page_icon="📚", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "stage" not in st.session_state:
    st.session_state.stage = "ingest"
if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []  # list[(label, icon)]
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "question" not in st.session_state:
    st.session_state.question = ""
if "url_inputs" not in st.session_state:
    st.session_state.url_inputs = [""]

SOURCE_ICON = {SourceType.URL: "🌐", SourceType.FILE: "📄"}


def _reset_session() -> None:
    pipeline.clear_session(st.session_state.session_id)
    st.session_state.stage = "ingest"
    st.session_state.ingested_sources = []
    st.session_state.suggested_questions = []
    st.session_state.question = ""
    st.session_state.url_inputs = [""]


st.title("📚 DeepRead")
st.markdown("Bring documents or URLs, ask questions, get grounded answers with citations.")
st.markdown(
    "[GitHub](https://github.com/samuel-mahembe/deepread) · "
    "Built with sentence-transformers + ChromaDB + Groq"
)
st.divider()

# ---------- STAGE 1: INGEST ----------
if st.session_state.stage == "ingest":
    st.markdown("### Step 1: Add your sources")
    st.caption("Add web page URLs and/or upload documents (PDF, DOCX, TXT, MD).")

    url_tab, file_tab = st.tabs(["🌐 URLs", "📄 Upload files"])

    with url_tab:
        for i in range(len(st.session_state.url_inputs)):
            col1, col2 = st.columns([10, 1])
            st.session_state.url_inputs[i] = col1.text_input(
                f"URL {i + 1}",
                value=st.session_state.url_inputs[i],
                key=f"url_input_{i}",
                placeholder="https://...",
                label_visibility="collapsed",
            )
            if len(st.session_state.url_inputs) > 1:
                if col2.button("✕", key=f"remove_{i}", help="Remove this URL"):
                    st.session_state.url_inputs.pop(i)
                    st.rerun()
            else:
                col2.write("")

        if st.button("➕ Add another URL"):
            st.session_state.url_inputs.append("")
            st.rerun()

    with file_tab:
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.caption("Max 10MB per file.")

    if st.button("Ingest sources", type="primary", use_container_width=True):
        urls = [u.strip() for u in st.session_state.url_inputs if u.strip()]
        files = uploaded_files or []

        if not urls and not files:
            st.error("Add at least one URL or upload a file.")
        else:
            outcomes = []
            total_steps = len(urls) + len(files)
            progress = st.progress(0.0, text="Starting...")
            step = 0

            for url in urls:
                progress.progress(step / total_steps, text=f"Fetching {url[:60]}...")
                outcome = pipeline.ingest_url(url, st.session_state.session_id)
                outcomes.append(outcome)
                step += 1

            for file in files:
                progress.progress(step / total_steps, text=f"Reading {file.name}...")
                outcome = pipeline.ingest_file(file, st.session_state.session_id)
                outcomes.append(outcome)
                step += 1

            succeeded = [o for o in outcomes if o.success]
            failed = [o for o in outcomes if not o.success]

            for o in failed:
                st.warning(f"Skipped **{o.source}**: {o.error}")

            if succeeded:
                progress.progress(0.9, text="Generating suggested questions...")
                st.session_state.suggested_questions = pipeline.suggest_questions_for_session(
                    st.session_state.session_id
                )
                for o in succeeded:
                    is_url = any(o.source == u for u in urls)
                    icon = SOURCE_ICON[SourceType.URL if is_url else SourceType.FILE]
                    st.session_state.ingested_sources.append((o.source, icon))

                progress.progress(1.0, text="Done!")
                st.session_state.stage = "ask"
                st.rerun()
            else:
                progress.empty()
                st.error("None of the provided sources could be ingested.")

# ---------- STAGE 2: ASK ----------
elif st.session_state.stage == "ask":
    with st.expander(f"📥 Ingested {len(st.session_state.ingested_sources)} source(s)"):
        for source, icon in st.session_state.ingested_sources:
            st.markdown(f"- {icon} {source}")
        if st.button("Start over with new sources"):
            _reset_session()
            st.rerun()

    if st.session_state.suggested_questions:
        st.markdown("**Suggested questions:**")
        cols = st.columns(2)
        for i, q in enumerate(st.session_state.suggested_questions):
            if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.question = q
                st.rerun()

    with st.form(key="ask_form", clear_on_submit=False):
        question = st.text_input(
            "Ask a question:",
            value=st.session_state.question,
            placeholder="What do these sources say about...?",
        )
        col1, col2 = st.columns([1, 5])
        submit = col1.form_submit_button("Ask", type="primary")
        clear = col2.form_submit_button("Clear")

    if clear:
        st.session_state.question = ""
        st.rerun()

    if submit and question:
        st.session_state.question = question

        with st.spinner("Searching and generating an answer..."):
            try:
                result = pipeline.ask(question, st.session_state.session_id)
            except Exception as exc:
                st.error(f"Couldn't generate an answer: {exc}")
                result = None

        if result:
            st.markdown("### Answer")
            st.markdown(result.answer)

            st.markdown("### Sources")
            if not result.sources:
                st.caption("No matching content was found for this question.")
            else:
                st.caption(f"Retrieved {len(result.sources)} chunk(s)")
                for i, chunk in enumerate(result.sources, start=1):
                    icon = SOURCE_ICON[chunk.source_type]
                    with st.expander(
                        f"Source {i}: {icon} {chunk.title} (similarity: {chunk.similarity:.0%})"
                    ):
                        st.markdown(f"**{chunk.source_type.value.upper()}:** {chunk.source}")
                        st.text(chunk.text)
