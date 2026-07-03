"""Anthropic LLM client with per-call token/cost accounting and a per-query budget guard.

Cost is computed from real usage (input/output tokens) at published list prices, so the
dashboard and eval numbers reflect actual money spent.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# USD per 1M tokens (input, output) — Anthropic list prices
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
}


class QueryBudgetExceeded(Exception):
    """Raised when a single query's accumulated LLM spend exceeds the configured cap."""


@dataclass
class LLMCall:
    purpose: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates LLM spend for one query; enforces the per-query budget."""

    budget_usd: float
    calls: list[LLMCall] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    def record(self, call: LLMCall) -> None:
        self.calls.append(call)
        if self.total_cost_usd > self.budget_usd:
            raise QueryBudgetExceeded(
                f"query spend ${self.total_cost_usd:.4f} exceeds budget ${self.budget_usd:.4f}"
            )


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = PRICES[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def _load_env_file() -> None:
    """Populate os.environ from ./.env (KEY=VALUE lines) without overriding real env vars."""
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


@lru_cache
def _client():
    import anthropic

    _load_env_file()
    return anthropic.Anthropic()


def complete(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    purpose: str = "generate",
    tracker: CostTracker | None = None,
) -> str:
    """One LLM call. Returns the text; records tokens+cost on the tracker if given."""
    response = _client().messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if tracker is not None:
        tracker.record(
            LLMCall(
                purpose=purpose,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=cost_usd(
                    model, response.usage.input_tokens, response.usage.output_tokens
                ),
            )
        )
    return "".join(block.text for block in response.content if block.type == "text")
