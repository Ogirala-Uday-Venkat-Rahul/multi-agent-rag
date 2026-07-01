"""FastAPI surface for the assistant.

  GET  /health   -> liveness + how many chunks are indexed
  POST /ingest   -> index a file or directory already on the server
  POST /ask      -> run the multi-agent RAG pipeline on a question
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .agents.graph import answer_question
from .config import settings
from .ingest import ingest_path
from .vectorstore import get_store

logger = logging.getLogger("uvicorn.error")
_ROOT = Path(__file__).parent.parent
_STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed the store on boot when it's empty. The free Space has ephemeral storage, so
    # the index is wiped on every restart; seeding here keeps the demo answerable with
    # no manual /ingest. Guarded so a seeding failure never blocks startup.
    try:
        if settings.seed_path and get_store().count() == 0:
            seed = _ROOT / settings.seed_path
            if seed.exists():
                added = ingest_path(seed)
                logger.info("Seeded vector store from %s (%d chunks)", seed, added)
    except Exception as exc:  # noqa: BLE001 - seeding is best-effort
        logger.warning("Startup seeding skipped: %s", exc)
    yield


app = FastAPI(title="Multi-Agent RAG Research Assistant", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    # Serve the single-page front end. It's a thin client of /ask; the interactive
    # API docs are still at /docs.
    return FileResponse(_STATIC / "index.html")


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
