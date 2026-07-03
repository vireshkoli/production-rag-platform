"""Prompt-injection guardrails.

Two layers on input: a zero-cost pattern screen for blatant attacks, then an LLM
classifier for subtler ones. A third structural layer lives in prompts.py (retrieved text
fenced as untrusted data). Output layer: citation enforcement in the pipeline.
"""

import re

from rag.config import settings
from rag.llm import CostTracker, complete
from rag.prompts import INJECTION_JUDGE_SYSTEM

# Blatant override attempts. Kept deliberately narrow — the LLM classifier catches the
# subtle cases; these exist so obvious attacks cost zero tokens to reject.
_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+|any\s+)?(previous|prior|above|earlier|your)\s+(instructions|prompts|rules)",
    r"you\s+are\s+now\s+(?!able)",  # persona hijack: "you are now DAN/EvilBot/..."
    r"\bnew\s+instructions?\s*:",
    r"\bsystem\s*(prompt|message)\s*:",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|initial\s+prompt)",
    r"(print|show|output|repeat)\s+(your\s+)?(system\s+prompt|instructions)\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"pretend\s+(you('re|\s+are)\s+)?(not\s+)?an?\s+(ai|assistant|different)",
    r"api[_\s-]?key",
]
_PATTERN_RE = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def pattern_screen(text: str) -> bool:
    """True if the input matches a known injection pattern."""
    return bool(_PATTERN_RE.search(text))


def llm_screen(text: str, tracker: CostTracker | None = None) -> bool:
    """True if the LLM classifier flags the input as an injection attempt."""
    verdict = complete(
        system=INJECTION_JUDGE_SYSTEM,
        user=text,
        model=settings().generation_model,
        max_tokens=5,
        purpose="guardrail",
        tracker=tracker,
    )
    return verdict.strip().upper().startswith("INJECTION")


def check_input(text: str, tracker: CostTracker | None = None) -> tuple[bool, str | None]:
    """Returns (blocked, reason). Pattern layer first (free), then LLM layer."""
    if pattern_screen(text):
        return True, "pattern"
    if llm_screen(text, tracker):
        return True, "llm_classifier"
    return False, None
