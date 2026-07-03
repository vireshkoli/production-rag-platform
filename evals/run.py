"""Run the evaluation for one pipeline and save per-case + summary results.

Usage:
  python -m evals.run --pipeline baseline
  python -m evals.run --pipeline full

Writes evals/results/{pipeline}.json (per case) and updates evals/results/summary.json.
Metrics:
  answerable   -> faithfulness, answer_relevancy, context_precision, correctness, latency, cost
  unanswerable -> correct-abstention rate (did the system decline instead of hallucinate?)
  injection    -> block rate (did the guardrail stop it?)
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from evals import judge
from rag.config import settings
from rag.pipeline import run_query

DATASET = Path("evals/dataset.json")
RESULTS_DIR = Path("evals/results")


def _percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return qs[min(98, max(0, round(pct) - 1))]


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def evaluate(pipeline: str) -> dict:
    cases = json.loads(DATASET.read_text())
    per_case = []
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        result = run_query(case["question"], pipeline)
        wall = (time.perf_counter() - t0) * 1000
        record = {
            "type": case["type"],
            "question": case["question"],
            "answer": result.answer,
            "blocked": result.blocked,
            "abstained": result.abstained,
            "latency_ms": round(wall, 1),
            "cost_usd": result.cost_usd,
            "groundedness": result.groundedness,
            "retrieval_rounds": result.retrieval_rounds,
        }
        contexts = [s["text"] for s in result.sources]

        if case["type"] == "answerable" and not result.blocked:
            record["faithfulness"] = judge.faithfulness(result.answer, contexts)
            record["answer_relevancy"] = judge.answer_relevancy(case["question"], result.answer)
            record["context_precision"] = judge.context_precision(case["question"], contexts)
            record["correctness"] = judge.correctness(
                case["question"], result.answer, case["reference"]
            )
        elif case["type"] == "unanswerable":
            # correct behavior = decline (abstain or block), not answer
            record["correct_abstention"] = result.abstained or result.blocked
        elif case["type"] == "injection":
            record["blocked_attack"] = result.blocked

        per_case.append(record)
        tag = case["type"][:4]
        print(f"  [{i}/{len(cases)}] ({tag}) {wall/1000:5.1f}s  {case['question'][:60]}")

    return summarize(pipeline, per_case)


def summarize(pipeline: str, per_case: list[dict]) -> dict:
    answerable = [r for r in per_case if r["type"] == "answerable" and not r["blocked"]]
    unanswerable = [r for r in per_case if r["type"] == "unanswerable"]
    injection = [r for r in per_case if r["type"] == "injection"]
    latencies = [r["latency_ms"] for r in per_case if not r["blocked"]]

    summary = {
        "pipeline": pipeline,
        "judge_model": settings().judge_model,
        "generation_model": settings().generation_model,
        "n_cases": len(per_case),
        "answerable": {
            "n": len(answerable),
            "faithfulness": _mean([r.get("faithfulness") for r in answerable]),
            "answer_relevancy": _mean([r.get("answer_relevancy") for r in answerable]),
            "context_precision": _mean([r.get("context_precision") for r in answerable]),
            "correctness": _mean([r.get("correctness") for r in answerable]),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "correct_abstention_rate": _mean(
                [1.0 if r.get("correct_abstention") else 0.0 for r in unanswerable]
            ),
        },
        "injection": {
            "n": len(injection),
            "block_rate": _mean([1.0 if r.get("blocked_attack") else 0.0 for r in injection]),
        },
        "latency_ms": {
            "p50": round(_percentile(latencies, 50), 1),
            "p95": round(_percentile(latencies, 95), 1),
            "mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
        },
        "cost_usd": {
            "avg_per_query": round(
                statistics.mean([r["cost_usd"] for r in per_case]), 6
            ),
            "total": round(sum(r["cost_usd"] for r in per_case), 4),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{pipeline}.json").write_text(
        json.dumps({"summary": summary, "cases": per_case}, indent=2, ensure_ascii=False)
    )
    summary_path = RESULTS_DIR / "summary.json"
    all_summaries = {}
    if summary_path.exists():
        all_summaries = json.loads(summary_path.read_text())
    all_summaries[pipeline] = summary
    summary_path.write_text(json.dumps(all_summaries, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["baseline", "full"], required=True)
    args = parser.parse_args()
    print(f"Evaluating '{args.pipeline}' pipeline on {DATASET}...")
    summary = evaluate(args.pipeline)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
