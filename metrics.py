"""Performance metrics: retrieval accuracy (precision@k) and latency."""

from __future__ import annotations

import time
from typing import Callable


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved candidate names/paths that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def measure_latency(func: Callable[[], object]) -> tuple[object, float]:
    """Run func() once, return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    return result, elapsed
