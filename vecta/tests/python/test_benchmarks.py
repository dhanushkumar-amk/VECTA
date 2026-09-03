"""
Unit tests for the benchmarking harness (Phase 8).
Verifies:
1. .fvecs and .ivecs parsing against known hand-crafted files.
2. recall_at_k calculations against known ground-truth examples.
3. Timer and QPS computation.
"""

import os
import tempfile
import numpy as np
import pytest

from benchmarks.datasets.download_sift1m import (
    read_fvecs,
    read_ivecs,
    write_fvecs,
    write_ivecs,
)
from benchmarks.utils.recall import recall_at_k
from benchmarks.utils.timer import Timer, compute_qps


def test_fvecs_parser_handcrafted_sample():
    """
    Confirm the .fvecs parser correctly reads a known small sample.
    Writes a tiny hand-crafted .fvecs file with 3 known vectors of dim 4.
    """
    known_vectors = np.array(
        [
            [1.0, 2.5, -3.0, 4.25],
            [0.0, 0.5, 1.0, 1.5],
            [10.0, -20.0, 30.0, -40.0],
        ],
        dtype=np.float32,
    )

    with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_fvecs(tmp_path, known_vectors)
        parsed = read_fvecs(tmp_path)

        assert parsed.shape == (3, 4)
        assert parsed.dtype == np.float32
        np.testing.assert_array_almost_equal(parsed, known_vectors)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_ivecs_parser_handcrafted_sample():
    """
    Confirm the .ivecs parser correctly reads known integer neighbor IDs.
    """
    known_ids = np.array(
        [
            [10, 20, 30],
            [100, 200, 300],
        ],
        dtype=np.int32,
    )

    with tempfile.NamedTemporaryFile(suffix=".ivecs", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_ivecs(tmp_path, known_ids)
        parsed = read_ivecs(tmp_path)

        assert parsed.shape == (2, 3)
        assert parsed.dtype == np.int32
        np.testing.assert_array_equal(parsed, known_ids)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_recall_at_k_handcrafted():
    """
    Confirm recall_at_k gives exact expected values.
    Hand-crafted case: predicted=[[1,2,3]], ground_truth=[[1,2,4]], k=3 -> 2/3.
    """
    pred = [[1, 2, 3]]
    gt = [[1, 2, 4]]
    recall = recall_at_k(pred, gt, k=3)
    assert abs(recall - (2.0 / 3.0)) < 1e-6

    # Perfect recall
    pred_perfect = [[10, 20, 30], [5, 6, 7]]
    gt_perfect = [[10, 20, 30], [5, 6, 7]]
    assert abs(recall_at_k(pred_perfect, gt_perfect, k=3) - 1.0) < 1e-6

    # Disjoint (zero recall)
    pred_zero = [[1, 2, 3]]
    gt_zero = [[4, 5, 6]]
    assert recall_at_k(pred_zero, gt_zero, k=3) == 0.0

    # Top-1 from Top-3
    pred_mixed = [[1, 2, 3]]
    gt_mixed = [[1, 4, 5]]
    assert abs(recall_at_k(pred_mixed, gt_mixed, k=1) - 1.0) < 1e-6


def test_timer_and_qps():
    """
    Verify Timer measures non-negative duration and compute_qps produces accurate rate.
    """
    with Timer("test") as t:
        _ = sum(i * i for i in range(10000))

    assert t.elapsed_sec >= 0.0
    assert t.elapsed_ms >= 0.0

    # 100 queries in 0.5s = 200 QPS
    qps = compute_qps(0.5, 100)
    assert abs(qps - 200.0) < 1e-6

    # Zero time / count edge cases
    assert compute_qps(0.0, 100) == 0.0
    assert compute_qps(1.0, 0) == 0.0
