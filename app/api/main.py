from fastapi import FastAPI
from app.api.routes_ingest import router as ingest_router
from app.api.routes_chat import router as chat_router

app = FastAPI(title="Local-First RAG Copilot")

app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])