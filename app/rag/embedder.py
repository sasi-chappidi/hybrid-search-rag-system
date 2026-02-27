import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = SentenceTransformer(settings.EMBED_MODEL)

def embed_texts(texts: list[str]) -> np.ndarray:
    vecs = _model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    return np.array(vecs, dtype="float32")