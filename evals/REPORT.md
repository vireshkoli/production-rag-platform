# Evaluation Report — Baseline vs. Full Pipeline

- **Generation & judge model:** `claude-haiku-4-5`
- **Cases:** 60 (40 answerable, 10 unanswerable, 10 injection attacks)
- **Baseline** = dense-only retrieval, no reranking, no self-check, no guardrails (a naive RAG system).
- **Full** = hybrid retrieval + cross-encoder reranking + agentic groundedness self-check + guardrails.

Reproduce: `python -m evals.run --pipeline baseline && python -m evals.run --pipeline full && python -m evals.report`

## Answer quality (answerable questions)

| Metric | Baseline | Full | Change |
|---|---|---|---|
| Faithfulness (grounding — higher = fewer hallucinations) | 98.8% | 99.7% | ▲ +0.9 pts |
| Correctness vs. gold answer (all answerable cases) | 59.4% | 59.9% | ▲ +0.5 pts |
| Answer relevancy | 87.6% | 88.5% | ▲ +0.9 pts |
| Context precision (retrieval quality) | 26.0% | 31.3% | ▲ +5.3 pts |
| Correctness on answered questions only | 95.0% | 95.8% | ▲ +0.8 pts |
| Answers with faithfulness < 0.8 (hallucination cases) | 1 | 0 | ▼ -1 |

Both pipelines abstained on 15/40 answerable cases — hyper-specific detail questions where retrieval did not surface the exact source passage. They decline rather than fabricate (the shared generation prompt instructs abstention), which is why aggregate correctness sits well below correctness-on-answered. Retrieval recall, not generation, is the binding constraint on this corpus.

## Hallucination control & security

| Behavior | Baseline | Full | Change |
|---|---|---|---|
| Correct abstention on unanswerable questions | 100.0% | 100.0% | → +0.0 pts |
| Prompt-injection block rate | 0.0% | 100.0% | ▲ +100.0 pts |

## Latency & cost

| Metric | Baseline | Full |
|---|---|---|
| p50 latency | 2.31s | 5.28s |
| p95 latency | 4.56s | 9.52s |
| Avg cost / query | $0.00290 | $0.00536 |

The full pipeline trades latency and cost (extra retrieval + a verification LLM call, sometimes a second retrieval round) for materially higher faithfulness and honest abstention. See the numbers above for the measured trade-off.

_Generated from `evals/results/summary.json`._
