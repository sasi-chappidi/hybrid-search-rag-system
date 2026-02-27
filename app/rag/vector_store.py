import os
import json
import faiss
import numpy as np
from typing import Dict, List, Tuple

class FaissStore:
    def __init__(self, index_dir: str, dim: int):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        self.index_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "meta.jsonl")
        self.dim = dim

        self.index = faiss.read_index(self.index_path) if os.path.exists(self.index_path) else faiss.IndexFlatIP(dim)

        self.meta: List[Dict] = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.meta.append(json.loads(line))

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.meta = []
        self.save()

    def add(self, vectors: np.ndarray, metas: List[Dict]) -> None:
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.meta.extend(metas)
        self.save()

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[float, Dict]]:
        q = np.array([query_vec]).astype("float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, top_k)

        results: List[Tuple[float, Dict]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((float(score), self.meta[idx]))
        return results

    def save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")