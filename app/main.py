"""FastAPI surface for the assistant.

  GET  /health   -> liveness + how many chunks are indexed
  POST /ingest   -> index a file or directory already on the server
  POST /ask      -> run the multi-agent RAG pipeline on a question
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .agents.graph import answer_question
from .config import settings
from .ingest import ingest_path
from .vectorstore import get_store

app = FastAPI(title="Multi-Agent RAG Research Assistant")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    # Land on the interactive API docs so the deployed Space opens to something
    # you can click through, not a bare 404.
    return RedirectResponse(url="/docs")


class AskRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    path: str  # file or directory path on the server


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "vector_store": settings.vector_store,
        "indexed_chunks": get_store().count(),
    }


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    try:
        added = ingest_path(req.path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    return {"ingested_chunks": added, "total_chunks": get_store().count()}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return answer_question(req.question)
