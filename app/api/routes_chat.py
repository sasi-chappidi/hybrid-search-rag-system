from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.rag.pipeline import answer_with_rag
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import FaissStore
from app.rag.bm25 import BM25Index

router = APIRouter()

_store = None
_bm25 = None
_retriever = None


class ChatReq(BaseModel):
    question: str
    top_k: int = 5


def _lazy_load():
    global _store, _bm25, _retriever

    if _store is None:
        _store = FaissStore(settings.INDEX_DIR, dim=384)

    if len(_store.meta) == 0:
        raise HTTPException(status_code=400, detail="Index empty. Build it: POST /ingest/build")

    if _bm25 is None:
        docs = [m["text"] for m in _store.meta]
        _bm25 = BM25Index(docs)

    if _retriever is None:
        _retriever = HybridRetriever(_store, _bm25)


@router.post("")
def chat(req: ChatReq):
    _lazy_load()
    hits = _retriever.retrieve(req.question, top_k=req.top_k)
    answer = answer_with_rag(req.question, hits)
    return {"answer": answer, "contexts": hits}
