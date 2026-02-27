from app.rag.chunker import chunk_text

def test_chunk_text_basic():
    text = "hello " * 200
    chunks = chunk_text(text, {"source": "x"}, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)