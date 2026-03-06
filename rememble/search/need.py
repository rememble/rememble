"""Memory need analysis — deterministic classifier for recall gating."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# ── Model ────────────────────────────────────────────────────


class MemoryNeed(BaseModel):
    need_type: (
        str  # none | temporal | identity | fact_lookup | open_loop | broad_context | prospective
    )
    should_recall: bool
    confidence: float
    query_hint: str | None = None
    reasons: list[str] = Field(default_factory=list)


# ── Pattern tuples: (compiled_regex, need_type, confidence) ──

_SKIP_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"^\s*$"), 1.0),
    (
        re.compile(
            r"^(ok|okay|sure|yes|no|yep|nope|yeah|nah|k|ty|thx|thanks"
            r"|thank you|got it|cool|nice|great|good|fine|alright|right"
            r"|ack|np|roger|noted|understood|done|lgtm)[\s!.]*$",
            re.I,
        ),
        0.95,
    ),
]

_TEMPORAL_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (
        re.compile(
            r"\b(what changed|what.s new|latest|recently|last time"
            r"|yesterday|today|this week|this month"
            r"|since .+|after .+|before .+|history of|timeline)\b",
            re.I,
        ),
        0.85,
    ),
    (re.compile(r"\b(when did|how long ago|last .+ (time|session))\b", re.I), 0.8),
]

_IDENTITY_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(who am i|my (name|role|preferences?|settings?|profile))\b", re.I), 0.9),
    (re.compile(r"\b(about me|my background|my (style|convention|habit)s?)\b", re.I), 0.85),
]

_OPEN_LOOP_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(did (we|i|you) (decide|agree|finish|complete|resolve))\b", re.I), 0.85),
    (
        re.compile(
            r"\b(next steps?|follow.?up|pending|unfinished"
            r"|to.?do|action items?|where (did )?we leave off)\b",
            re.I,
        ),
        0.8,
    ),
]

_FACT_LOOKUP_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"^(who|what|when|where|which|how)\b.{10,}", re.I), 0.7),
    (
        re.compile(r"\b(tell me about|explain|describe|define|what is|what are|what was)\b", re.I),
        0.75,
    ),
]

_BROAD_CONTEXT_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (
        re.compile(
            r"\b(catch me up|recap|brief me|summarize|summary|overview|what do (you|we) know)\b",
            re.I,
        ),
        0.9,
    ),
    (re.compile(r"\b(everything (about|on|related))\b", re.I), 0.8),
]

_PROSPECTIVE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(remind me|don.t forget|remember (to|that)|note (that|this))\b", re.I), 0.85),
]

# Short query threshold
_SHORT_QUERY_LEN = 20


# ── Classifier ───────────────────────────────────────────────


def analyzeMemoryNeed(query: str) -> MemoryNeed:
    """Classify whether recall is useful. Deterministic, <1ms, no LLM."""
    stripped = query.strip()

    # Empty / whitespace-only
    if not stripped:
        return MemoryNeed(
            need_type="none", should_recall=False, confidence=1.0, reasons=["empty query"]
        )

    # Acknowledgements / trivial responses
    for pat, conf in _SKIP_PATTERNS:
        if pat.match(stripped):
            return MemoryNeed(
                need_type="none",
                should_recall=False,
                confidence=conf,
                reasons=["matches skip pattern"],
            )

    # Short queries with no meaningful content
    if len(stripped) < _SHORT_QUERY_LEN and not any(c.isupper() for c in stripped[1:]):
        # Short AND no capitalized words (likely no named entities)
        words = stripped.split()
        if len(words) <= 2:
            return MemoryNeed(
                need_type="none",
                should_recall=False,
                confidence=0.6,
                reasons=["short query, few words, no entities"],
            )

    # Temporal patterns
    for pat, conf in _TEMPORAL_PATTERNS:
        if pat.search(stripped):
            return MemoryNeed(
                need_type="temporal",
                should_recall=True,
                confidence=conf,
                query_hint=stripped,
                reasons=["matches temporal pattern"],
            )

    # Identity patterns
    for pat, conf in _IDENTITY_PATTERNS:
        if pat.search(stripped):
            return MemoryNeed(
                need_type="identity",
                should_recall=True,
                confidence=conf,
                query_hint=stripped,
                reasons=["matches identity pattern"],
            )

    # Open loop patterns
    for pat, conf in _OPEN_LOOP_PATTERNS:
        if pat.search(stripped):
            return MemoryNeed(
                need_type="open_loop",
                should_recall=True,
                confidence=conf,
                query_hint=stripped,
                reasons=["matches open loop pattern"],
            )

    # Broad context patterns
    for pat, conf in _BROAD_CONTEXT_PATTERNS:
        if pat.search(stripped):
            return MemoryNeed(
                need_type="broad_context",
                should_recall=True,
                confidence=conf,
                query_hint=stripped,
                reasons=["matches broad context pattern"],
            )

    # Prospective patterns
    for pat, conf in _PROSPECTIVE_PATTERNS:
        if pat.search(stripped):
            return MemoryNeed(
                need_type="prospective",
                should_recall=True,
                confidence=conf,
                query_hint=stripped,
                reasons=["matches prospective pattern"],
            )

    # Fact lookup patterns (checked last — most general)
    for pat, conf in _FACT_LOOKUP_PATTERNS:
        if pat.search(stripped):
            return MemoryNeed(
                need_type="fact_lookup",
                should_recall=True,
                confidence=conf,
                query_hint=stripped,
                reasons=["matches fact lookup pattern"],
            )

    # Default: recall (conservative — when in doubt, recall)
    return MemoryNeed(
        need_type="general",
        should_recall=True,
        confidence=0.5,
        query_hint=stripped,
        reasons=["no specific pattern matched, defaulting to recall"],
    )
