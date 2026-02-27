# tests/test_api.py
import os
from fastapi.testclient import TestClient


# IMPORTANT: set test index dir BEFORE importing app (settings load at import time)
def test_chat_requires_index(tmp_path, monkeypatch):
    # Point index to a temp folder so it's always empty for this test
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))

    # Import after env var set
    from app.api.main import app

    client = TestClient(app)

    resp = client.post("/chat", json={"question": "hi", "top_k": 2})
    assert resp.status_code == 400
