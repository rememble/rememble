"""ACT-R base-level activation scoring for memory retrieval."""

from __future__ import annotations

import math
import time


def computeActivation(
    access_history: list[float],
    now: float | None = None,
    decay_exponent: float = 0.5,
    b_mid: float = 0.0,
    b_scale: float = 1.5,
    min_age_seconds: float = 1.0,
) -> float:
    """ACT-R base-level learning equation, normalized to [0, 1].

    B_i(t) = ln(Σ (t - t_j)^(-d))
    activation = sigmoid((B - B_mid) / B_scale)

    Each t_j is an epoch-second timestamp of an access event.
    Power-law decay naturally handles "used a lot long ago" vs "used recently".
    """
    if not access_history:
        return 0.0

    if now is None:
        now = time.time()

    total = 0.0
    for t_j in access_history:
        age = max(now - t_j, min_age_seconds)
        total += age ** (-decay_exponent)

    if total <= 0:
        return 0.0

    B = math.log(total)
    return 1.0 / (1.0 + math.exp(-(B - b_mid) / b_scale))


def temporalScore(
    created_at_ms: int,
    accessed_at_ms: int,
    access_count: int,
    half_life_days: float = 7.0,
) -> float:
    """Legacy fallback — synthesize history from count+timestamps, delegate to ACT-R.

    Kept for backward compat with callers that don't have access_history_json.
    """
    now = time.time()
    created_s = created_at_ms / 1000.0
    accessed_s = accessed_at_ms / 1000.0
    history = synthesizeHistory(created_s, accessed_s, max(access_count, 1))
    return computeActivation(history, now=now)


def synthesizeHistory(
    created_at_s: float, accessed_at_s: float, access_count: int
) -> list[float]:
    """Generate synthetic access timestamps from count + time range.

    Spaces `access_count` events evenly from created_at to accessed_at.
    """
    if access_count <= 0:
        return []
    if access_count == 1:
        return [accessed_at_s]
    span = accessed_at_s - created_at_s
    if span <= 0:
        return [accessed_at_s] * access_count
    step = span / (access_count - 1)
    return [created_at_s + i * step for i in range(access_count)]
