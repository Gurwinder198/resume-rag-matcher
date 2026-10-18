"""Performance metrics: retrieval accuracy (precision@k) and latency."""

from __future__ import annotations

import time
from typing import Callable


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the k requested slots that are relevant. Dividing by k
    (not by however many were actually retrieved) means a short result list
    is penalized for its unfilled slots, per the standard precision@k
    definition."""
    if k <= 0:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / k


def measure_latency(func: Callable[[], object]) -> tuple[object, float]:
    """Run func() once, return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    return result, elapsed
