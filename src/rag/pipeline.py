"""Answer pipelines.

baseline: dense top-k retrieval -> generate. What a naive RAG system does; kept as the
          measured control for the evaluation.
full:     injection guardrails -> hybrid retrieval + rerank -> generate -> agentic
          groundedness self-check. The self-check verifies every claim against the
          retrieved sources; on low groundedness it rewrites the query, re-retrieves,
          and regenerates once; if the answer still can't be grounded, it abstains
          instead of hallucinating.
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field

from rag.config import settings
from rag.guardrails import check_input
from rag.llm import CostTracker, complete
from rag.prompts import (
    ABSTAIN_PREFIX,
    ANSWER_SYSTEM,
    REWRITE_SYSTEM,
    VERIFY_SYSTEM,
    answer_user_prompt,
    rewrite_user_prompt,
    verify_user_prompt,
)
from rag.retrieval import RetrievedChunk, rerank, retrieve

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

BLOCKED_MESSAGE = (
    "This request was flagged by the prompt-injection guardrail and was not processed."
)
UNGROUNDED_MESSAGE = (
    f"{ABSTAIN_PREFIX} to answer this reliably: I could not verify the answer against the "
    "indexed sources, so I'm declining rather than risking an unsupported answer."
)


@dataclass
class QueryResult:
    question: str
    pipeline: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    abstained: bool = False
    blocked: bool = False
    blocked_reason: str | None = None
    groundedness: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    retrieval_rounds: int = 0
    rewritten_query: str | None = None
    timings_ms: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class _Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks: dict[str, float] = {}

    def mark(self, stage: str, since: float) -> float:
        now = time.perf_counter()
        self.marks[stage] = self.marks.get(stage, 0.0) + (now - since) * 1000
        return now

    def total(self) -> dict[str, float]:
        out = {k: round(v, 1) for k, v in self.marks.items()}
        out["total"] = round((time.perf_counter() - self.t0) * 1000, 1)
        return out


def _generate(question: str, chunks: list[RetrievedChunk], tracker: CostTracker) -> str:
    return complete(
        system=ANSWER_SYSTEM,
        user=answer_user_prompt(question, chunks),
        model=settings().generation_model,
        max_tokens=700,
        purpose="generate",
        tracker=tracker,
    )


def _parse_citations(answer: str, chunks: list[RetrievedChunk]) -> list[dict]:
    cited: list[dict] = []
    seen: set[int] = set()
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(chunks) and n not in seen:
            seen.add(n)
            c = chunks[n - 1]
            cited.append({"n": n, "title": c.title, "section": c.section, "url": c.url})
    return cited


def _verify(answer: str, chunks: list[RetrievedChunk], tracker: CostTracker):
    """Claim-level groundedness check. Returns (score 0-1, unsupported claim texts)."""
    raw = complete(
        system=VERIFY_SYSTEM,
        user=verify_user_prompt(answer, chunks),
        model=settings().judge_model,
        max_tokens=1000,
        purpose="verify",
        tracker=tracker,
    )
    match = _JSON_RE.search(raw)
    if not match:
        return 0.0, ["verifier returned unparseable output"]
    try:
        claims = json.loads(match.group(0)).get("claims", [])
    except json.JSONDecodeError:
        return 0.0, ["verifier returned invalid JSON"]
    if not claims:  # abstention — nothing to ground
        return 1.0, []
    unsupported = [
        c.get("claim", "") for c in claims if c.get("verdict", "").upper() != "SUPPORTED"
    ]
    return 1 - len(unsupported) / len(claims), unsupported


def _is_abstention(answer: str) -> bool:
    return answer.strip().startswith(ABSTAIN_PREFIX)


def _sources_payload(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {"n": i, "title": c.title, "section": c.section, "url": c.url, "text": c.text}
        for i, c in enumerate(chunks, start=1)
    ]


def run_query(question: str, pipeline: str = "full") -> QueryResult:
    cfg = settings()
    tracker = CostTracker(budget_usd=cfg.max_query_cost_usd)
    timer = _Timer()
    t = timer.t0

    # --- Guardrails (full pipeline only; baseline is the unprotected control) ---
    if pipeline == "full":
        blocked, reason = check_input(question, tracker)
        t = timer.mark("guardrail", t)
        if blocked:
            return QueryResult(
                question=question,
                pipeline=pipeline,
                answer=BLOCKED_MESSAGE,
                blocked=True,
                blocked_reason=reason,
                timings_ms=timer.total(),
                input_tokens=tracker.total_input_tokens,
                output_tokens=tracker.total_output_tokens,
                cost_usd=round(tracker.total_cost_usd, 6),
            )

    # --- Round 1: retrieve + generate ---
    chunks = retrieve(question, pipeline)
    t = timer.mark("retrieval", t)
    answer = _generate(question, chunks, tracker)
    t = timer.mark("generation", t)
    rounds = 1
    rewritten = None
    groundedness: float | None = None
    unsupported: list[str] = []

    if pipeline == "full":
        # --- Agentic self-check: verify -> (re-retrieve -> regenerate -> verify) -> abstain
        citation_ok = _is_abstention(answer) or bool(_parse_citations(answer, chunks))
        groundedness, unsupported = _verify(answer, chunks, tracker)
        if not citation_ok:
            groundedness = 0.0
            unsupported = unsupported or ["answer contains no citations"]
        t = timer.mark("verification", t)

        if groundedness < cfg.groundedness_threshold and rounds < cfg.max_retrieval_rounds:
            rewritten = complete(
                system=REWRITE_SYSTEM,
                user=rewrite_user_prompt(question, unsupported),
                model=cfg.generation_model,
                max_tokens=100,
                purpose="rewrite",
                tracker=tracker,
            ).strip()
            new_chunks = retrieve(rewritten, "full")
            merged = {c.id: c for c in chunks + new_chunks}.values()
            chunks = rerank(question, list(merged), cfg.top_k + 3)
            rounds += 1
            t = timer.mark("retrieval", t)

            answer = _generate(question, chunks, tracker)
            t = timer.mark("generation", t)
            citation_ok = _is_abstention(answer) or bool(_parse_citations(answer, chunks))
            groundedness, unsupported = _verify(answer, chunks, tracker)
            if not citation_ok:
                groundedness = 0.0
            t = timer.mark("verification", t)

        if groundedness < cfg.groundedness_threshold:
            answer = UNGROUNDED_MESSAGE  # honest abstention beats a hallucination

    return QueryResult(
        question=question,
        pipeline=pipeline,
        answer=answer,
        citations=_parse_citations(answer, chunks),
        sources=_sources_payload(chunks),
        abstained=_is_abstention(answer),
        groundedness=None if groundedness is None else round(groundedness, 3),
        unsupported_claims=unsupported,
        retrieval_rounds=rounds,
        rewritten_query=rewritten,
        timings_ms=timer.total(),
        input_tokens=tracker.total_input_tokens,
        output_tokens=tracker.total_output_tokens,
        cost_usd=round(tracker.total_cost_usd, 6),
    )
