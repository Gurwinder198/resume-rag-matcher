import time

from metrics import measure_latency, precision_at_k


def test_precision_at_k_counts_relevant_hits_in_top_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=4) == 0.5


def test_precision_at_k_handles_empty_retrieved():
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_precision_at_k_divides_by_k_not_by_number_retrieved():
    # Only 2 results were retrieved, but k=5 was requested: the 3 unfilled
    # slots must count against the score, not be excluded from it.
    retrieved = ["a", "b"]
    relevant = {"a", "b"}
    assert precision_at_k(retrieved, relevant, k=5) == 0.4


def test_measure_latency_returns_result_and_nonnegative_elapsed():
    result, elapsed = measure_latency(lambda: (time.sleep(0.01), "done")[1])
    assert result == "done"
    assert elapsed >= 0.0
