# Production RAG Platform

A production-grade retrieval-augmented generation service with **hybrid retrieval**,
**cross-encoder reranking**, an **agentic groundedness self-check** that re-retrieves on
low-confidence answers and abstains rather than hallucinates, **prompt-injection
guardrails**, and a **measured evaluation harness** proving the improvements — plus a live
per-query cost/latency dashboard.

**Live demo:** [vireshk-production-rag-platform.hf.space](https://vireshk-production-rag-platform.hf.space)
(free CPU tier — a full-pipeline query takes a few seconds) ·
**Live dashboard:** [/dashboard](https://vireshk-production-rag-platform.hf.space/dashboard) ·
**Evaluation report:** [`evals/REPORT.md`](evals/REPORT.md)

---

## Measured results

Evaluated on a committed 60-case dataset (40 answerable with gold answers, 10 unanswerable,
10 prompt-injection attacks), scored by an independent LLM judge (`claude-haiku-4-5`, pinned).
**Baseline** = naive RAG (dense-only top-k, no reranking, no self-check, no guardrails).
**Full** = hybrid + rerank + agentic self-check + guardrails. Full per-case results are
committed in [`evals/results/`](evals/results/); analysis in [`evals/REPORT.md`](evals/REPORT.md).

| Metric | Baseline | Full pipeline | Change |
|---|---|---|---|
| **Prompt-injection block rate** | 0% | **100%** | **+100 pts** |
| Hallucinated answers (faithfulness < 0.8) | 1 | **0** | −1 |
| Faithfulness (claim-level grounding) | 98.8% | **99.7%** | +0.9 pts |
| Context precision (retrieval quality) | 26.0% | **31.3%** | +5.3 pts |
| Correctness on answered questions | 95.0% | **95.8%** | +0.8 pts |
| Correct abstention on unanswerable questions | 100% | 100% | — |
| p50 / p95 latency | 2.31s / 4.56s | 5.28s / 9.52s | cost of self-check |
| Avg cost per query (real token usage × list price) | $0.0029 | $0.0054 | +$0.0025 |

Honest readout: with a well-prompted modern LLM, faithfulness starts high — the self-check's
measurable value is **eliminating the residual hallucination tail** (1 → 0 low-faithfulness
answers, verified claim-by-claim on every query) and **abstaining honestly** on the 15/40
hyper-specific questions where retrieval can't surface the exact evidence (both pipelines
answer only what they can ground — correctness on *answered* questions is ~96%). The
guardrail layer is the starkest win: 0% → 100% of injection attacks blocked. The price is
~2.3× latency and ~1.9× cost per query — measured, not estimated.

## What it does

Ask a question about AI/ML. The service:

1. **Screens the input** for prompt-injection (zero-cost regex layer, then an LLM classifier).
2. **Retrieves hybrid candidates** — dense (`bge-small-en-v1.5`) and BM25 sparse vectors,
   fused server-side in Qdrant with Reciprocal Rank Fusion.
3. **Reranks** the fused candidates with a cross-encoder (`ms-marco-MiniLM-L-6-v2`).
4. **Generates a cited answer** with Claude Haiku — every claim carries a `[n]` citation
   that resolves to a real Wikipedia source chunk.
5. **Self-checks groundedness (agentic loop):** an independent verifier call extracts the
   answer's claims and checks each against the retrieved sources. If groundedness falls
   below threshold, the system **rewrites the query, re-retrieves, and regenerates**; if
   the answer still can't be grounded, it **abstains honestly** instead of hallucinating.
6. **Logs everything** — per-stage latency, token usage, and real cost (list prices) — to
   SQLite, rendered live at `/dashboard`.

The **corpus** is 312 English Wikipedia articles on AI/ML (~7,000 chunks, ~1.3M words),
fetched by a reproducible one-command ingest. Wikipedia text is CC BY-SA; the fetched
cache is committed so builds are deterministic.

## Architecture

```
                        ┌─────────────────────────────────────────────┐
 question ──▶ guardrails ──▶ hybrid retrieval ──▶ cross-encoder ──▶ generation (cited)
   │        (regex + LLM)   Qdrant: dense+BM25       reranker             │
   │                          fused with RRF                             ▼
   │                               ▲                          groundedness verifier
   │                               │ rewritten query        (claim-level LLM check)
   │                               └────────── low confidence ◀──┤
   │                                                             │ still ungrounded
   ▼                                                             ▼
 SQLite telemetry ──▶ /dashboard (p50/p95 latency, cost/query)  abstain honestly
```

| Concern | Choice | Why |
|---|---|---|
| API | FastAPI + uvicorn | Standard production Python stack |
| Vector store | Qdrant (embedded local mode) | Real vector DB with native hybrid search + server-side RRF; zero infra on free hosting; `RAG_QDRANT_URL` switches to a Qdrant server unchanged |
| Dense embeddings | BAAI/bge-small-en-v1.5 (384-d) | Best quality/latency on CPU-only free hosting |
| Keyword retrieval | BM25 sparse vectors (fastembed) | True hybrid inside one Qdrant query, not two glued systems |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | The standard CPU-fast cross-encoder |
| LLM | Claude Haiku 4.5 (`claude-haiku-4-5`) | Strong quality at $1/$5 per M tokens; same model pinned as eval judge |
| Eval metrics | RAGAS-style LLM-judge prompts (in-repo) | Transparent and reproducible — no black-box eval dependency |
| Telemetry | SQLite + self-contained dashboard | Live per-query cost/latency without an external service |

### Design decisions & trade-offs

- **The baseline is fair.** Both pipelines share the same generation prompt (including the
  abstention instruction). The measured delta isolates exactly what hybrid retrieval,
  reranking, and the self-check loop add — not prompt differences.
- **Groundedness ≠ correctness.** The verifier checks support-by-sources (faithfulness).
  Correctness against gold answers is measured separately by the eval judge.
- **Cost guard.** Every query has a hard budget (`RAG_MAX_QUERY_COST_USD`, default $0.05);
  a runaway agentic loop cannot burn money.
- **Injection defense in depth.** Regex screen (free) → LLM classifier → retrieved text
  fenced as untrusted data in the prompt → citation enforcement on output.
- **Latency trade-off is explicit.** The self-check adds a verification call (and
  sometimes a second retrieval round). The eval reports the real p50/p95 cost of that
  choice next to the quality gains.

## Run it yourself

Prereqs: Python 3.12 via [uv](https://docs.astral.sh/uv/), an
[Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/vireshkoli/production-rag-platform
cd production-rag-platform
cp .env.example .env            # paste your ANTHROPIC_API_KEY into .env

uv sync                         # install deps (Python 3.12 venv)
uv run python -m rag.ingest     # build the Qdrant index (~5 min; corpus cache is committed)
uv run uvicorn rag.api:app --port 8000
```

Open http://localhost:8000 (query UI), http://localhost:8000/dashboard (live metrics),
http://localhost:8000/docs (OpenAPI).

```bash
curl -X POST localhost:8000/api/query -H 'Content-Type: application/json' \
  -d '{"question": "Who coined the term machine learning?", "pipeline": "full"}'
```

### Docker

```bash
docker build -t rag-platform .          # bakes models + index into the image
docker run --rm -p 7860:7860 --env-file .env rag-platform
```

### Reproduce the evaluation

```bash
uv run python -m evals.build_dataset    # optional: regenerate the 60-case dataset
uv run python -m evals.run --pipeline baseline
uv run python -m evals.run --pipeline full
uv run python -m evals.report           # renders evals/REPORT.md
```

Per-case results land in `evals/results/*.json` (committed), so every number in this
README can be regenerated and audited.

## Repository layout

```
src/rag/            the service
  ingest/           Wikipedia fetcher → heading-aware chunker → Qdrant index build
  retrieval.py      hybrid RRF search + cross-encoder rerank
  pipeline.py       baseline & full pipelines, agentic self-check loop
  guardrails.py     injection screens (regex + LLM)
  llm.py            Anthropic client, cost accounting, per-query budget guard
  telemetry.py      SQLite query log + metrics aggregation
  api.py            FastAPI app
evals/              dataset builder, LLM-judge metrics, runner, committed results
static/             query UI + live dashboard (self-contained HTML)
tests/              unit tests (LLM mocked — run in CI without secrets)
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/query` | POST | `{question, pipeline: "full"\|"baseline"}` → answer, citations, sources, groundedness, timings, cost |
| `/api/metrics` | GET | Aggregated telemetry (p50/p95 latency, cost, recent queries) |
| `/dashboard` | GET | Live dashboard UI |
| `/healthz` | GET | Health check |

## License

MIT — see [LICENSE](LICENSE). Corpus text: Wikipedia, CC BY-SA 4.0, cited per chunk.
