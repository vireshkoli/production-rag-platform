"""Render evals/results/summary.json into a human-readable evals/REPORT.md table.

Usage: python -m evals.report
"""

import json
from pathlib import Path

RESULTS_DIR = Path("evals/results")
REPORT = Path("evals/REPORT.md")


def _pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def _ms(v):
    return f"{v / 1000:.2f}s"


def _delta(base, full):
    if base is None or full is None:
        return ""
    d = (full - base) * 100
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")
    return f"{arrow} {d:+.1f} pts"


def main() -> None:
    summaries = json.loads((RESULTS_DIR / "summary.json").read_text())
    b = summaries.get("baseline")
    f = summaries.get("full")
    if not b or not f:
        raise SystemExit("Need both baseline and full results. Run evals.run for each pipeline.")

    lines = [
        "# Evaluation Report — Baseline vs. Full Pipeline",
        "",
        f"- **Generation & judge model:** `{f['generation_model']}`",
        f"- **Cases:** {f['n_cases']} "
        f"({f['answerable']['n']} answerable, {f['unanswerable']['n']} unanswerable, "
        f"{f['injection']['n']} injection attacks)",
        "- **Baseline** = dense-only retrieval, no reranking, no self-check, no guardrails "
        "(a naive RAG system).",
        "- **Full** = hybrid retrieval + cross-encoder reranking + agentic groundedness "
        "self-check + guardrails.",
        "",
        "Reproduce: `python -m evals.run --pipeline baseline && "
        "python -m evals.run --pipeline full && python -m evals.report`",
        "",
        "## Answer quality (answerable questions)",
        "",
        "| Metric | Baseline | Full | Change |",
        "|---|---|---|---|",
    ]
    for key, label in [
        ("faithfulness", "Faithfulness (grounding — higher = fewer hallucinations)"),
        ("correctness", "Correctness vs. gold answer"),
        ("answer_relevancy", "Answer relevancy"),
        ("context_precision", "Context precision (retrieval quality)"),
    ]:
        bv, fv = b["answerable"][key], f["answerable"][key]
        lines.append(f"| {label} | {_pct(bv)} | {_pct(fv)} | {_delta(bv, fv)} |")

    lines += [
        "",
        "## Hallucination control & security",
        "",
        "| Behavior | Baseline | Full | Change |",
        "|---|---|---|---|",
    ]
    bv = b["unanswerable"]["correct_abstention_rate"]
    fv = f["unanswerable"]["correct_abstention_rate"]
    lines.append(
        f"| Correct abstention on unanswerable questions | {_pct(bv)} | {_pct(fv)} "
        f"| {_delta(bv, fv)} |"
    )
    bv = b["injection"]["block_rate"]
    fv = f["injection"]["block_rate"]
    lines.append(f"| Prompt-injection block rate | {_pct(bv)} | {_pct(fv)} | {_delta(bv, fv)} |")

    lines += [
        "",
        "## Latency & cost",
        "",
        "| Metric | Baseline | Full |",
        "|---|---|---|",
        f"| p50 latency | {_ms(b['latency_ms']['p50'])} | {_ms(f['latency_ms']['p50'])} |",
        f"| p95 latency | {_ms(b['latency_ms']['p95'])} | {_ms(f['latency_ms']['p95'])} |",
        f"| Avg cost / query | ${b['cost_usd']['avg_per_query']:.5f} "
        f"| ${f['cost_usd']['avg_per_query']:.5f} |",
        "",
        "The full pipeline trades latency and cost (extra retrieval + a verification LLM call, "
        "sometimes a second retrieval round) for materially higher faithfulness and honest "
        "abstention. See the numbers above for the measured trade-off.",
        "",
        "_Generated from `evals/results/summary.json`._",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
