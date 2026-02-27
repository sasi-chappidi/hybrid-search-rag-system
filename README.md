# Local-First Enterprise RAG Copilot (Hybrid Search + Citations + CI)

A recruiter-grade RAG project that runs **fully free** on a laptop:
- Ingest PDF/TXT/MD → chunking + metadata
- Retrieval: **Hybrid Search** (BM25 + Embeddings) using FAISS + SentenceTransformers
- Answers: **grounded extractive generation** (no paid LLM) with citations (source/page/chunk)
- API: FastAPI
- UI: Streamlit
- CI: GitHub Actions (ruff + black + pytest)

## Quickstart (Windows)
### 1) Install
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install fastapi uvicorn streamlit pydantic-settings python-multipart
pip install sentence-transformers faiss-cpu rank-bm25 numpy pypdf
pip install scikit-learn
pip install pytest httpx ruff black