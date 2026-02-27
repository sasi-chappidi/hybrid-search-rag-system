from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    return text.lower().split()

class BM25Index:
    def __init__(self, docs: list[str]):
        self.tokens = [tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, top_k: int) -> list[tuple[float, int]]:
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(float(score), int(doc_id)) for doc_id, score in ranked]