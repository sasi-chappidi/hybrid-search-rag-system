from typing import Dict, List
from app.rag.embedder import embed_texts
from app.rag.vector_store import FaissStore
from app.rag.bm25 import BM25Index

class HybridRetriever:
    def __init__(self, store: FaissStore, bm25: BM25Index, alpha: float = 0.55):
        self.store = store
        self.bm25 = bm25
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        qvec = embed_texts([query])[0]

        v_hits = self.store.search(qvec, top_k=top_k * 2)
        b_hits = self.bm25.search(query, top_k=top_k * 2)

        combined: dict[int, float] = {}

        for score, meta in v_hits:
            cid = meta["chunk_id"]
            combined[cid] = combined.get(cid, 0.0) + self.alpha * score

        for score, cid in b_hits:
            combined[cid] = combined.get(cid, 0.0) + (1.0 - self.alpha) * score

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[Dict] = []
        for cid, score in ranked:
            meta = self.store.meta[cid]
            results.append({**meta, "score": float(score)})

        return results