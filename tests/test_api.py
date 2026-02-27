# tests/test_api.py
from fastapi.testclient import TestClient


def test_chat_requires_index(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    from app.api.main import app

    client = TestClient(app)

    resp = client.post("/chat", json={"question": "hi", "top_k": 2})
    assert resp.status_code == 400
