from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Chunk:
    text: str
    meta: Dict

def chunk_text(text: str, meta: Dict, chunk_size: int, overlap: int) -> List[Chunk]:
    clean = " ".join(text.split())
    if not clean:
        return []

    if len(clean) <= chunk_size:
        return [Chunk(text=clean, meta=meta)]

    chunks: List[Chunk] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        piece = clean[start:end]
        chunks.append(Chunk(text=piece, meta=meta))
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks