from fastapi import APIRouter
from pathlib import Path
from app.core.config import settings
from app.core.utils import read_pdf, read_txt_md
from app.rag.chunker import chunk_text
from app.rag.embedder import embed_texts
from app.rag.vector_store import FaissStore

router = APIRouter()

def _get_store() -> FaissStore:
    return FaissStore(settings.INDEX_DIR, dim=384)  # MiniLM dim

@router.post("/build")
def build_index(reset: bool = True):
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    store = _get_store()
    if reset:
        store.reset()

    chunk_records = []
    texts_for_embed = []
    chunk_id_start = len(store.meta)

    for pdf in raw_dir.glob("*.pdf"):
        for page_no, page_text in read_pdf(pdf):
            meta = {"source": pdf.name, "page": page_no}
            for ch in chunk_text(page_text, meta, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP):
                record = {
                    "chunk_id": chunk_id_start + len(chunk_records),
                    "source": ch.meta["source"],
                    "page": ch.meta.get("page"),
                    "text": ch.text,
                }
                chunk_records.append(record)
                texts_for_embed.append(ch.text)

    for fp in list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.md")):
        text = read_txt_md(fp)
        meta = {"source": fp.name, "page": None}
        for ch in chunk_text(text, meta, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP):
            record = {
                "chunk_id": chunk_id_start + len(chunk_records),
                "source": ch.meta["source"],
                "page": ch.meta.get("page"),
                "text": ch.text,
            }
            chunk_records.append(record)
            texts_for_embed.append(ch.text)

    if not chunk_records:
        return {"status": "no_docs_found", "message": "Put pdf/txt/md into data/raw and retry."}

    vectors = embed_texts(texts_for_embed)
    store.add(vectors, chunk_records)

    return {"status": "ok", "chunks_added": len(chunk_records), "total_chunks": len(store.meta)}