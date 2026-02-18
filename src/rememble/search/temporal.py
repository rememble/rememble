"""Temporal/importance scoring based on age, frequency, and recency."""

from __future__ import annotations

import math
import time


def temporalScore(
    created_at_ms: int,
    accessed_at_ms: int,
    access_count: int,
    half_life_days: float = 7.0,
) -> float:
    """Three-component importance score (Wax-inspired).

    Components:
    - Age: exp(-age_hours / (half_life_days * 24)) — newer is better
    - Frequency: min(1.0, log(access_count + 1) / 5.0) — more accessed is better
    - Recency: exp(-hours_since_access / 24) — recently accessed is better

    Weights: 0.3 age + 0.4 frequency + 0.3 recency
    """
    now_ms = int(time.time() * 1000)

    age_hours = max(0, (now_ms - created_at_ms) / 3_600_000)
    hours_since_access = max(0, (now_ms - accessed_at_ms) / 3_600_000)

    age_component = math.exp(-age_hours / (half_life_days * 24))
    frequency_component = min(1.0, math.log(access_count + 1) / 5.0)
    recency_component = math.exp(-hours_since_access / 24)

    return 0.3 * age_component + 0.4 * frequency_component + 0.3 * recency_component
