"""Per-query telemetry: every query is logged to SQLite; /metrics aggregates it live."""

import json
import sqlite3
import statistics
from datetime import UTC, datetime

from rag.config import settings
from rag.pipeline import QueryResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    question TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    answer_preview TEXT,
    blocked INTEGER NOT NULL DEFAULT 0,
    abstained INTEGER NOT NULL DEFAULT 0,
    groundedness REAL,
    retrieval_rounds INTEGER,
    latency_ms REAL NOT NULL,
    timings_json TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    settings().telemetry_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings().telemetry_db)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)


def log_query(result: QueryResult) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO queries
               (ts, question, pipeline, answer_preview, blocked, abstained, groundedness,
                retrieval_rounds, latency_ms, timings_json, input_tokens, output_tokens,
                cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(timespec="seconds"),
                result.question[:500],
                result.pipeline,
                result.answer[:300],
                int(result.blocked),
                int(result.abstained),
                result.groundedness,
                result.retrieval_rounds,
                result.timings_ms.get("total", 0.0),
                json.dumps(result.timings_ms),
                result.input_tokens,
                result.output_tokens,
                result.cost_usd,
            ),
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return qs[min(98, max(0, round(pct) - 1))]


def metrics(recent: int = 25) -> dict:
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM queries ORDER BY id")]
    answered = [r for r in rows if not r["blocked"]]
    latencies = sorted(r["latency_ms"] for r in answered)
    return {
        "total_queries": len(rows),
        "blocked": sum(r["blocked"] for r in rows),
        "abstained": sum(r["abstained"] for r in rows),
        "latency_ms": {
            "p50": round(_percentile(latencies, 50), 1),
            "p95": round(_percentile(latencies, 95), 1),
        },
        "cost_usd": {
            "avg_per_query": round(
                sum(r["cost_usd"] for r in rows) / len(rows), 6
            )
            if rows
            else 0.0,
            "total": round(sum(r["cost_usd"] for r in rows), 4),
        },
        "avg_groundedness": round(
            statistics.mean(
                g for r in answered if (g := r["groundedness"]) is not None
            ),
            3,
        )
        if any(r["groundedness"] is not None for r in answered)
        else None,
        "recent": [
            {
                **{k: r[k] for k in (
                    "ts", "question", "pipeline", "blocked", "abstained", "groundedness",
                    "retrieval_rounds", "latency_ms", "input_tokens", "output_tokens",
                    "cost_usd",
                )},
                "timings": json.loads(r["timings_json"] or "{}"),
            }
            for r in rows[-recent:][::-1]
        ],
    }
