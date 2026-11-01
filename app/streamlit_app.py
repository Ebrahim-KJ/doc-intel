"""
streamlit_app.py
----------------
DocIntel interactive dashboard.

Features:
  - Upload PDFs or paste URLs for ingestion
  - Chat interface with streaming responses
  - Inline source citations with chunk previews
  - Per-stage pipeline latency breakdown
  - Live evaluation scores (faithfulness, relevance, precision)
  - Conversation history with session management
"""

import os
import sys
import time
import requests
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="DocIntel — Document Intelligence",
    page_icon="",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────

if "session_id"   not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages"     not in st.session_state:
    st.session_state.messages = []
if "last_result"  not in st.session_state:
    st.session_state.last_result = None

# ── Helpers ───────────────────────────────────────────────────

def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return None


def api_ingest_url(url: str):
    r = requests.post(f"{API_BASE}/ingest/url", json={"url": url})
    r.raise_for_status()
    return r.json()


def api_ingest_file(file_bytes: bytes, filename: str):
    r = requests.post(
        f"{API_BASE}/ingest/file",
        files={"file": (filename, file_bytes, "application/pdf")}
    )
    r.raise_for_status()
    return r.json()


def api_query(query: str, session_id: str):
    r = requests.post(
        f"{API_BASE}/query",
        json={"query": query, "session_id": session_id},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def api_evaluate(query: str, answer: str, chunks: list):
    r = requests.post(
        f"{API_BASE}/evaluate",
        json={"query": query, "answer": answer, "chunks": chunks},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def api_documents():
    r = requests.get(f"{API_BASE}/documents", timeout=5)
    r.raise_for_status()
    return r.json()


# ── Header ────────────────────────────────────────────────────

st.title("DocIntel — Document Intelligence & Research Assistant")
st.markdown(
    "Upload documents, ask questions, get cited answers. "
    "Powered by **Hybrid RAG** (FAISS + BM25 + Cross-encoder reranking) and **Claude**."
)

# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.header("Document Ingestion")

    # Health check
    health = api_health()
    if health:
        st.success(f"API connected — {health['faiss_vectors']} chunks indexed")
    else:
        st.error("API offline. Start with: `uvicorn src.api.main:app --reload`")

    st.markdown("---")

    # URL ingestion
    st.subheader("Ingest a URL")
    url_input = st.text_input("Paste a URL (article, paper, Wikipedia...)")
    if st.button("Ingest URL", disabled=not url_input):
        with st.spinner("Ingesting..."):
            try:
                result = api_ingest_url(url_input)
                st.success(f"Ingested {result.get('chunks', '?')} chunks")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # PDF upload
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded and st.button("Ingest PDF"):
        with st.spinner("Ingesting..."):
            try:
                result = api_ingest_file(uploaded.read(), uploaded.name)
                st.success(f"{uploaded.name} ingested")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Indexed documents
    st.subheader("Indexed Documents")
    try:
        docs = api_documents()
        for doc in docs.get("documents", [])[:10]:
            st.caption(f"• {doc[:50]}")
        st.caption(f"Total chunks: {docs.get('total_chunks', 0)}")
    except Exception:
        st.caption("No documents indexed yet.")

    st.markdown("---")

    # Session
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("Clear conversation"):
        st.session_state.messages  = []
        st.session_state.last_result = None
        st.rerun()

# ── Main: Chat ────────────────────────────────────────────────

col_chat, col_meta = st.columns([2, 1])

with col_chat:
    st.subheader("Research Chat")

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Query input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                try:
                    result = api_query(user_input, st.session_state.session_id)
                    answer = result.get("answer", "No answer returned.")

                    st.markdown(answer)

                    # Citations
                    citations = result.get("citations", [])
                    if citations:
                        with st.expander(f"{len(citations)} source(s) cited", expanded=False):
                            for c in citations:
                                st.markdown(f"**{c['source']}** — Page {c['page']}")
                                st.caption(c['preview'])
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_result = result

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Is it running?")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Right column: Pipeline metadata ───────────────────────────

with col_meta:
    st.subheader("Pipeline Diagnostics")

    result = st.session_state.last_result
    if result is None:
        st.info("Ask a question to see pipeline diagnostics.")
    else:
        # Latency breakdown
        timings = result.get("timings", {})
        if timings:
            st.markdown("**Stage Latencies**")
            stage_labels = {
                "query_expansion_s": "Query expansion",
                "retrieval_s":       "Hybrid retrieval",
                "rerank_s":          "Cross-encoder rerank",
                "generation_s":      "LLM generation",
                "total_s":           "Total",
            }
            for key, label in stage_labels.items():
                if key in timings:
                    val = timings[key]
                    bar = "█" * min(int(val * 5), 20)
                    st.markdown(f"`{label}` — **{val}s** {bar}")

        st.divider()

        # Retrieved chunks
        chunks = result.get("chunks", [])
        if chunks:
            st.markdown(f"**Retrieved Chunks ({len(chunks)})**")
            for i, chunk in enumerate(chunks[:5], 1):
                score = chunk.get("rerank_score", chunk.get("rrf_score", 0))
                with st.expander(f"Chunk {i} — {chunk['source']} p.{chunk['page']} (score: {score:.3f})"):
                    st.caption(chunk["text"][:300] + "...")

        st.divider()

        # Evaluation scores (on demand)
        st.markdown("**Evaluate This Response**")
        if st.button("Run evaluation metrics"):
            answer = result.get("answer", "")
            with st.spinner("Scoring faithfulness, relevance, precision..."):
                try:
                    scores = api_evaluate(user_input or "", answer, chunks)
                    metrics = {
                        "Faithfulness":      scores.get("faithfulness", 0),
                        "Answer Relevance":  scores.get("answer_relevance", 0),
                        "Context Precision": scores.get("context_precision", 0),
                        "Mean Score":        scores.get("mean_score", 0),
                    }
                    for name, score in metrics.items():
                        color = "" if score >= 0.7 else ("" if score >= 0.4 else "")
                        st.metric(name, f"{score:.2f}", delta=color)
                except Exception as e:
                    st.error(f"Evaluation error: {e}")

# ── Footer ────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "**Built by Ebrahim Kashif Jahangiri** · "
    "Cornell MSc Data Science for Public Policy · "
    "Imperial MEng · "
    "[GitHub](https://github.com/YOUR_USERNAME/doc-intel)"
)
st.caption(
    "Architecture: Multi-query expansion → Hybrid retrieval (FAISS + BM25) → "
    "RRF fusion → Cross-encoder reranking → Claude generation → Citation extraction"
)
