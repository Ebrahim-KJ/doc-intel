# 🔍 DocIntel — Document Intelligence & Research Assistant

An advanced Retrieval-Augmented Generation (RAG) system that transforms static documents into an interactive research intelligence layer. Upload any PDF or URL, ask questions in natural language, and receive cited, faithful answers — with full explainability at every stage of the pipeline.

Built with production-grade RAG architecture: hybrid dense/sparse retrieval, cross-encoder reranking, multi-query expansion, conversation memory, citation tracking, evaluation metrics, a FastAPI backend, and a live Streamlit dashboard.

---

## 🚀 Live Demo
👉 **[Launch the App](https://your-streamlit-link-here)**

---

## 🎯 Why This Is Hard (And Why It Matters)

Basic RAG — embed documents, retrieve the closest chunks, feed to an LLM — fails in subtle ways: it misses relevant chunks that use different vocabulary (vocabulary mismatch problem), it retrieves chunks that are superficially similar but contextually wrong, and it hallucinates when retrieved context is insufficient.

DocIntel addresses each failure mode with a specific architectural solution:

| Problem | Solution |
|---|---|
| Vocabulary mismatch | BM25 sparse retrieval alongside dense embeddings |
| Poor ranking | Cross-encoder reranking on the merged candidate set |
| Single-query brittleness | Multi-query expansion (3 LLM-generated variants) |
| Hallucination | Faithfulness scoring + citation grounding |
| No memory | Sliding window conversation buffer |
| Black-box retrieval | Per-stage latency tracking + evaluation suite |

---

## 📁 Project Structure

```
doc-intel/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py      # PDF, URL, plain text loading
│   │   ├── chunking.py             # Semantic + fixed-size chunking strategies
│   │   └── pipeline.py             # Full ingestion orchestration
│   ├── retrieval/
│   │   ├── embeddings.py           # Sentence-transformer embedding model
│   │   ├── vector_store.py         # FAISS index — build, save, load, search
│   │   ├── bm25_retriever.py       # BM25 sparse keyword retrieval
│   │   ├── hybrid_retriever.py     # RRF fusion of dense + sparse results
│   │   └── reranker.py             # Cross-encoder reranking (ms-marco)
│   ├── generation/
│   │   ├── llm_client.py           # Anthropic Claude API client (streaming)
│   │   ├── prompt_templates.py     # System prompts + context assembly
│   │   ├── multi_query.py          # LLM-based query expansion
│   │   └── rag_pipeline.py         # End-to-end RAG orchestration
│   ├── memory/
│   │   └── conversation.py         # Sliding window conversation buffer
│   ├── evaluation/
│   │   └── evaluator.py            # Faithfulness, relevance, context precision
│   └── api/
│       └── main.py                 # FastAPI: /ingest /query /evaluate /health
├── app/
│   └── streamlit_app.py            # Live interactive dashboard
├── data/
│   ├── documents/                  # Uploaded source documents
│   └── indices/                    # Persisted FAISS + BM25 indices
├── notebooks/
│   └── 01_rag_exploration.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & configure environment
```bash
git clone https://github.com/YOUR_USERNAME/doc-intel.git
cd doc-intel
cp .env.example .env
# Add your Anthropic API key to .env
```

### 2. Install dependencies
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run with Docker (recommended)
```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

### 4. Or run locally
```bash
# Terminal 1 — API
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run app/streamlit_app.py
```

---

## 🏃 Usage

### Ingest a document
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@paper.pdf"

# Or by URL
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/abs/2307.09288"}'
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?", "session_id": "abc123"}'
```

### Evaluate the RAG pipeline
```bash
python src/evaluation/evaluator.py
```

### View MLflow dashboard
```bash
mlflow ui  # → http://localhost:5000
```

---

## 🧠 Technical Architecture

### Ingestion
Documents are loaded (PDF via PyMuPDF, URLs via BeautifulSoup), split using a semantic chunker that respects sentence boundaries, and indexed in two parallel stores: FAISS for dense vector search and BM25 for sparse keyword matching.

### Hybrid Retrieval
Each query is first expanded into 3 variants by the LLM, then all variants query both FAISS (top-20) and BM25 (top-20). Results are merged using **Reciprocal Rank Fusion (RRF)** — a rank aggregation method that rewards chunks appearing in the top positions of multiple lists — producing a merged candidate set of 40.

### Cross-Encoder Reranking
The merged 40 candidates are scored by a **cross-encoder** (ms-marco-MiniLM-L-6-v2) that jointly encodes the query and each chunk — far more accurate than the approximate cosine similarity used during retrieval. Only the top 5 chunks are forwarded to generation.

### Generation
The top-5 chunks, conversation history (sliding window), and system prompt are assembled into a structured prompt and sent to **Claude** via the Anthropic API with streaming enabled. Each chunk carries its source document and page number, which are extracted into inline citations in the response.

### Evaluation
Three metrics are computed automatically on each response:
- **Faithfulness**: does the answer stay within the retrieved context (no hallucination)?
- **Answer relevance**: does the answer address the actual question?
- **Context precision**: were the retrieved chunks actually useful?

All metrics and latencies are logged to MLflow.

---

## 📦 Docker
```bash
docker-compose up --build
```

Services:
- `api` — FastAPI backend on port 8000
- `app` — Streamlit dashboard on port 8501
- Shared volume for `data/` (documents + indices)

---

## 👤 Author

**Ebrahim Kashif Jahangiri**
MSc Data Science for Public Policy — Cornell University | MEng — Imperial College London
📧 Ebrahim.KJahangiri@Gmail.com
