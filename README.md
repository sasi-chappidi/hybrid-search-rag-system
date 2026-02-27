# Enterprise RAG Copilot — Hybrid Search, Evidence-Cited Q&A, FastAPI + Streamlit, CI

A production-style **Retrieval-Augmented Generation (RAG)** system that runs **fully free on a laptop** (no paid LLM APIs).  
It ingests PDFs/TXT/MD, builds a searchable index, and answers questions with **evidence citations** for traceability.

## Key Features
- **Document ingestion**: PDF / TXT / Markdown → chunking + metadata (source/page/chunk)
- **Hybrid Retrieval**: **BM25 (lexical)** + **FAISS embeddings (semantic)** with hybrid scoring
- **Grounded answers**: local, extractive answer generation with **citations** to avoid hallucinations
- **API + UI**: FastAPI backend + Streamlit demo app
- **Engineering quality**: GitHub Actions CI (ruff + black + pytest)

## Architecture (High Level)
1. **Ingest** documents from `data/raw/`  
2. **Chunk** content with overlap + attach metadata  
3. **Embed** chunks using SentenceTransformers  
4. **Index** embeddings in FAISS + store metadata in JSONL  
5. **Retrieve** using Hybrid Search (BM25 + Vector)  
6. **Answer** using evidence-based extraction + citations

---

## Quickstart (Windows)

### 1) Create environment + install deps
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip

pip install fastapi uvicorn streamlit pydantic-settings python-multipart
pip install sentence-transformers faiss-cpu rank-bm25 numpy pypdf
pip install scikit-learn
pip install pytest httpx ruff black


2) Add documents to ingest

Copy any files into:

data/raw/*.pdf

data/raw/*.txt

data/raw/*.md

Example:

data/raw/resume.pdf
data/raw/job_description.pdf
data/raw/notes.txt
3) Run the API
uvicorn app.api.main:app --reload
4) Build the index
curl -X POST "http://localhost:8000/ingest/build?reset=true"
5) Run the UI
streamlit run app/ui/streamlit_app.py
API Endpoints
Build / rebuild index

POST /ingest/build?reset=true

Ask a question

POST /chat

Example:

curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are the key requirements?\",\"top_k\":5}"

Response includes:

answer (with citations)

contexts (retrieved evidence chunks)

CI (GitHub Actions)

On every push / pull request, CI runs:

ruff (lint)

black --check (format validation)

pytest (tests)