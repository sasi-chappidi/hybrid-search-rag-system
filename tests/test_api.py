from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_chat_requires_index():
    resp = client.post("/chat", json={"question": "hi", "top_k": 2})
    assert resp.status_code == 400