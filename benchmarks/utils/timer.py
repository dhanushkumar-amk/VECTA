"""
Benchmarking timing utilities for vecta.

Provides:
- Timer: Context manager and decorator for timing operations
- compute_qps: Calculate Queries Per Second
- save_benchmark_result: Persist timing & recall metrics to JSON
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional


class Timer:
    """
    Context manager and decorator for measuring wall-clock elapsed time.

    Usage:
        with Timer("Index Build") as t:
            index.add_batch(ids, vectors)
        print(f"Time: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0  # seconds

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000.0

    @property
    def elapsed_sec(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed

    def __repr__(self) -> str:
        return f"<Timer '{self.description}': {self.elapsed_ms:.2f} ms>"


def compute_qps(total_time_sec: float, query_count: int) -> float:
    """
    Compute queries per second (QPS).

    Args:
        total_time_sec: Total search execution time in seconds.
        query_count: Number of queries executed.

    Returns:
        Queries per second (QPS).
    """
    if total_time_sec <= 0.0 or query_count <= 0:
        return 0.0
    return query_count / total_time_sec


def save_benchmark_result(
    benchmark_name: str,
    metrics: Any,
    results_dir: Optional[str] = None,
) -> str:
    """
    Save benchmark metrics to a timestamped JSON file.

    Args:
        benchmark_name: Base name for benchmark (e.g., 'flat_index').
        metrics: Dictionary or list of benchmark measurements and metadata.
        results_dir: Output directory (defaults to benchmarks/results).

    Returns:
        Path to the saved JSON file.
    """
    if results_dir is None:
        # Resolve relative to this file: ../results
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{benchmark_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    if isinstance(metrics, list):
        payload = {
            "benchmark": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "sweep": metrics,
        }
    elif isinstance(metrics, dict):
        payload = {
            "benchmark": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
    else:
        payload = {
            "benchmark": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "data": metrics,
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return filepath


if __name__ == "__main__":
    # Self-test
    with Timer("Sleep test") as t:
        time.sleep(0.05)
    assert t.elapsed >= 0.04
    qps = compute_qps(t.elapsed, 100)
    assert qps > 0
    print(f"Timer test passed: {t.elapsed_ms:.2f}ms, {qps:.1f} QPS")
