"""FastAPI service: query API, live metrics, dashboard, and a minimal query UI."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import telemetry
from rag.config import settings
from rag.llm import QueryBudgetExceeded
from rag.pipeline import run_query


def _find_static_dir() -> Path:
    candidates = (
        Path(__file__).resolve().parents[2] / "static",  # editable install (src layout)
        Path.cwd() / "static",  # site-packages install run from the app root (Docker)
    )
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


STATIC_DIR = _find_static_dir()


@asynccontextmanager
async def lifespan(app: FastAPI):
    telemetry.init_db()
    if not os.environ.get("RAG_SKIP_WARMUP"):
        from rag.embeddings import warm_up

        warm_up()  # load embedder/reranker up front so the first query isn't slow
    yield


app = FastAPI(
    title="Production RAG Platform",
    description=(
        "Hybrid retrieval + cross-encoder reranking + agentic groundedness self-check, "
        "with per-query cost/latency telemetry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    pipeline: str = Field(default="full", pattern="^(baseline|full)$")


@app.post("/api/query")
def query(req: QueryRequest) -> dict:
    try:
        result = run_query(req.question, req.pipeline)
    except QueryBudgetExceeded as e:
        raise HTTPException(status_code=402, detail=str(e)) from e
    telemetry.log_query(result)
    return result.to_dict()


@app.get("/api/metrics")
def get_metrics() -> dict:
    return telemetry.metrics()


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "collection": settings().collection}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")
