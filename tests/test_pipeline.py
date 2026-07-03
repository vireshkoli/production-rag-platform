import json

import pytest

import rag.pipeline as pipeline
from rag.llm import CostTracker, LLMCall, QueryBudgetExceeded, cost_usd
from rag.pipeline import _is_abstention, _parse_citations, _verify
from rag.prompts import ABSTAIN_PREFIX
from rag.retrieval import RetrievedChunk


def make_chunks(n=3):
    return [
        RetrievedChunk(
            id=str(i),
            title=f"Article {i}",
            section="Intro",
            text=f"text {i}",
            url=f"https://example.org/{i}",
            score=1.0,
        )
        for i in range(1, n + 1)
    ]


def test_parse_citations_maps_and_dedupes():
    chunks = make_chunks(3)
    cited = _parse_citations("Fact one [1]. Fact two [3][1]. Bogus [9].", chunks)
    assert [c["n"] for c in cited] == [1, 3]
    assert cited[0]["title"] == "Article 1"
    assert cited[1]["url"] == "https://example.org/3"


def test_abstention_detection():
    assert _is_abstention(f"{ABSTAIN_PREFIX} to answer that.")
    assert not _is_abstention("The transformer was introduced in 2017 [1].")


def test_cost_accounting_matches_list_prices():
    # haiku: $1/M input, $5/M output
    assert cost_usd("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.00)
    assert cost_usd("claude-haiku-4-5", 0, 1_000_000) == pytest.approx(5.00)
    assert cost_usd("claude-haiku-4-5", 3000, 500) == pytest.approx(0.0055)


def test_budget_guard_raises_when_exceeded():
    tracker = CostTracker(budget_usd=0.01)
    tracker.record(LLMCall("generate", 3000, 500, 0.0055))
    with pytest.raises(QueryBudgetExceeded):
        tracker.record(LLMCall("verify", 4000, 600, 0.007))
    assert tracker.total_input_tokens == 7000
    assert tracker.total_output_tokens == 1100


def _patch_verifier(monkeypatch, payload: str):
    monkeypatch.setattr(pipeline, "complete", lambda **kwargs: payload)


def test_verify_scores_supported_fraction(monkeypatch):
    _patch_verifier(
        monkeypatch,
        json.dumps(
            {
                "claims": [
                    {"claim": "a", "verdict": "SUPPORTED"},
                    {"claim": "b", "verdict": "UNSUPPORTED"},
                    {"claim": "c", "verdict": "SUPPORTED"},
                    {"claim": "d", "verdict": "SUPPORTED"},
                ]
            }
        ),
    )
    score, unsupported = _verify("answer", make_chunks(), CostTracker(1))
    assert score == pytest.approx(0.75)
    assert unsupported == ["b"]


def test_verify_treats_abstention_as_grounded(monkeypatch):
    _patch_verifier(monkeypatch, '{"claims": []}')
    score, unsupported = _verify("answer", make_chunks(), CostTracker(1))
    assert score == 1.0
    assert unsupported == []


def test_verify_handles_garbage_output(monkeypatch):
    _patch_verifier(monkeypatch, "I cannot judge this, sorry!")
    score, unsupported = _verify("answer", make_chunks(), CostTracker(1))
    assert score == 0.0
    assert unsupported


def test_verify_handles_fenced_json(monkeypatch):
    _patch_verifier(
        monkeypatch,
        '```json\n{"claims": [{"claim": "a", "verdict": "SUPPORTED"}]}\n```',
    )
    score, _ = _verify("answer", make_chunks(), CostTracker(1))
    assert score == 1.0
