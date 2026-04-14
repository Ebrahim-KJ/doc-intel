"""
main.py
-------
FastAPI backend exposing the DocIntel RAG pipeline as a REST API.

Endpoints:
  POST /ingest        — Upload a PDF or submit a URL for indexing
  POST /query         — Ask a question, get a cited answer
  POST /evaluate      — Score a (query, answer, chunks) triple
  GET  /health        — Health check + index stats
  GET  /documents     — List indexed documents
  DELETE /documents   — Clear all indexed documents
"""

import os
import uuid
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.ingestion.pipeline       import ingest, load_indices
from src.generation.rag_pipeline  import query as rag_query, query_stream
from src.evaluation.evaluator     import evaluate
from src.retrieval.vector_store   import get_store
from src.retrieval.bm25_retriever import get_bm25

# ── App setup ─────────────────────────────────────────────────

app = FastAPI(
    title="DocIntel API",
    description="Document Intelligence & Research Assistant — RAG backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../../data/documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
async def startup():
    """Load persisted indices on startup."""
    load_indices()


# ── Request / Response models ──────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str
    strategy: str = "semantic"

class QueryRequest(BaseModel):
    query:      str
    session_id: str  = "default"
    stream:     bool = False

class EvaluateRequest(BaseModel):
    query:  str
    answer: str
    chunks: list


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    store = get_store()
    bm25  = get_bm25()
    return {
        "status":        "ok",
        "faiss_vectors": store.size,
        "bm25_docs":     bm25.size,
        "model":         os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
    }


@app.post("/ingest/file")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: str    = "semantic",
):
    """Upload a PDF for indexing."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Save to disk
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest in background
    background_tasks.add_task(ingest, dest, strategy)

    return {"status": "ingesting", "file": file.filename}


@app.post("/ingest/url")
def ingest_url(request: IngestURLRequest):
    """Ingest a web URL."""
    try:
        stats = ingest(request.url, strategy=request.strategy)
        return {"status": "ok", **stats}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/query")
def query_endpoint(request: QueryRequest):
    """Query the RAG pipeline. Returns answer + citations + timings."""
    if get_store().size == 0:
        raise HTTPException(400, "No documents indexed. Please ingest documents first.")

    if request.stream:
        # Server-sent events for streaming
        def generate():
            for chunk in query_stream(request.query, request.session_id):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")

    try:
        result = rag_query(request.query, session_id=request.session_id)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/evaluate")
def evaluate_endpoint(request: EvaluateRequest):
    """Score a (query, answer, chunks) triple."""
    try:
        scores = evaluate(
            query=request.query,
            answer=request.answer,
            chunks=request.chunks,
            log_mlflow=True,
        )
        return scores
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/documents")
def list_documents():
    """List all indexed document sources."""
    store = get_store()
    if store.size == 0:
        return {"documents": []}
    sources = list({m["source"] for m in store.metadata})
    return {"documents": sources, "total_chunks": store.size}


@app.delete("/documents")
def clear_documents():
    """Clear all indices (destructive)."""
    store = get_store()
    bm25  = get_bm25()
    store.index    = None
    store.metadata = []
    bm25.bm25      = None
    bm25.metadata  = []

    # Remove persisted files
    index_dir = os.path.join(os.path.dirname(__file__), "../../data/indices")
    for fname in ["faiss.index", "metadata.json", "bm25.joblib", "bm25_meta.json"]:
        path = os.path.join(index_dir, fname)
        if os.path.exists(path):
            os.remove(path)

    return {"status": "cleared"}
