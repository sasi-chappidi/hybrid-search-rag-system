from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ")
    parts = [s.strip() for s in text.split(".") if s.strip()]
    return [p + "." for p in parts]

def answer_locally(question: str, contexts: List[Dict], max_sentences: int = 4) -> str:
    if not contexts:
        return "I don't have enough information from the documents."

    candidates = []
    meta_map = []
    for c in contexts:
        for s in _split_sentences(c["text"]):
            if len(s) < 30:
                continue
            candidates.append(s)
            meta_map.append(c)

    if len(candidates) < 5:
        return _summary_fallback(contexts)

    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform([question] + candidates)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()

    top_idx = sims.argsort()[::-1][:max_sentences]

    lines = []
    used = set()
    for idx in top_idx:
        c = meta_map[idx]
        key = (c["source"], c.get("page"), c["chunk_id"])
        if key in used:
            continue
        used.add(key)
        lines.append(
            f"- {candidates[idx]} (source={c['source']}, page={c.get('page')}, chunk={c['chunk_id']})"
        )

    return "Extracted answer (grounded):\n" + "\n".join(lines)

def _summary_fallback(contexts: List[Dict]) -> str:
    lines = []
    for c in contexts[:3]:
        snippet = c["text"][:280].strip()
        lines.append(f"- {snippet}... (source={c['source']}, page={c.get('page')}, chunk={c['chunk_id']})")
    return "Grounded summary from top evidence:\n" + "\n".join(lines)