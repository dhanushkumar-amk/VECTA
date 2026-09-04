"""
Unit and integration tests for FAISS comparison runner machinery (Phase 36).

Covers:
1. run_trials() warmup discard and trial count verification.
2. summarize_timings() accuracy on hand-crafted known values.
3. summarize_timings() high-variance flagging (>20% stddev/mean).
4. Full compare_flat_index() end-to-end on SIFT data.
5. 100% Recall@10 sanity check on exact search.
"""

import math
import time
import numpy as np
import pytest

from benchmarks.faiss_comparison.run_comparison import (
    run_trials,
    summarize_timings,
    compare_flat_index,
)


class TestTrialMachinery:
    """Test 1: Reusable run_trials execution logic."""

    def test_run_trials_trial_count_and_warmup(self):
        call_counts = {"warmup": 0, "measured": 0}
        warmup_limit = 2
        num_trials = 4
        queries = [1, 2, 3]

        def mock_search(q, k):
            # Track invocations
            call_counts["measured"] += 1

        trial_times = run_trials(
            mock_search,
            queries,
            k=5,
            num_trials=num_trials,
            warmup_trials=warmup_limit,
        )

        assert len(trial_times) == num_trials
        # Total calls = (warmup_limit + num_trials) * len(queries)
        # = (2 + 4) * 3 = 18
        assert call_counts["measured"] == (warmup_limit + num_trials) * len(queries)

        for t in trial_times:
            assert isinstance(t, float)
            assert t >= 0.0

    def test_run_trials_timing_accuracy(self):
        queries = ["q1", "q2"]
        sleep_per_query = 0.005  # 5ms

        def sleeping_search(q, k):
            time.sleep(sleep_per_query)

        trial_times = run_trials(
            sleeping_search,
            queries,
            k=1,
            num_trials=3,
            warmup_trials=1,
        )

        assert len(trial_times) == 3
        expected_min_per_trial = len(queries) * sleep_per_query  # 10ms
        for t in trial_times:
            assert t >= expected_min_per_trial * 0.8  # Allow slight timing tolerance


class TestStatisticalSummary:
    """Test 2 & 3: Statistics computation and high-variance detection."""

    def test_summarize_timings_hand_crafted_values(self):
        # Known times: 0.10s, 0.20s, 0.30s
        # Mean time: (0.1 + 0.2 + 0.3) / 3 = 0.20s
        # Median time: 0.20s
        # Variance (sample): ((0.1-0.2)^2 + 0 + (0.3-0.2)^2) / 2 = 0.02 / 2 = 0.01
        # Stddev: sqrt(0.01) = 0.10s
        # num_queries = 100
        # QPS values: 100/0.1 = 1000, 100/0.2 = 500, 100/0.3 = 333.333
        # Mean QPS: (1000 + 500 + 333.333) / 3 = 611.111
        trial_times = [0.10, 0.20, 0.30]
        num_queries = 100

        summary = summarize_timings(trial_times, num_queries)

        assert abs(summary["mean_time_sec"] - 0.20) < 1e-6
        assert abs(summary["median_time_sec"] - 0.20) < 1e-6
        assert abs(summary["stddev_time_sec"] - 0.10) < 1e-6
        assert abs(summary["mean_qps"] - 611.1111) < 1e-3
        assert abs(summary["median_qps"] - 500.0) < 1e-3

        # Variance ratio = 0.10 / 0.20 = 0.50 (50% > 20%) -> warning flag TRUE
        assert abs(summary["variance_ratio"] - 0.50) < 1e-6
        assert summary["high_variance_warning"] is True

    def test_summarize_timings_clean_low_variance(self):
        # Clean times with < 5% variance: 0.099, 0.100, 0.101, 0.100
        trial_times = [0.099, 0.100, 0.101, 0.100]
        num_queries = 100

        summary = summarize_timings(trial_times, num_queries)

        assert abs(summary["mean_time_sec"] - 0.100) < 1e-3
        # Stddev ~ 0.000816, variance_ratio ~ 0.008 (< 20%) -> warning flag FALSE
        assert summary["variance_ratio"] < 0.05
        assert summary["high_variance_warning"] is False

    def test_summarize_timings_empty_raises(self):
        with pytest.raises(ValueError, match="trial_times cannot be empty"):
            summarize_timings([], 100)

    def test_summarize_timings_zero_queries_raises(self):
        with pytest.raises(ValueError, match="num_queries must be positive"):
            summarize_timings([0.1], 0)


class TestFlatComparisonEndToEnd:
    """Test 4 & 5: Full compare_flat_index() execution and 100% recall sanity check."""

    def test_compare_flat_index_siftsmall(self):
        # Run comparison with 3 trials, 1 warmup
        results = compare_flat_index(
            dataset_name="siftsmall",
            k=10,
            metric="euclidean",
            threads=1,
            num_trials=3,
            warmup_trials=1,
            save_json=False,
        )

        # Confirm expected structure
        assert "vecta" in results
        assert "faiss" in results
        assert "comparison" in results

        v = results["vecta"]
        f = results["faiss"]
        comp = results["comparison"]

        assert v["index_name"] == "vecta.FlatIndex"
        assert f["index_name"] == "IndexFlatL2"

        assert len(v["raw_trial_times_sec"]) == 3
        assert len(f["raw_trial_times_sec"]) == 3
        assert v["mean_qps"] > 0
        assert f["mean_qps"] > 0
        assert comp["qps_speedup_ratio"] > 0

        # Sanity check: BOTH must report recall@10 at 100%
        assert (
            abs(v["recall_at_k"] - 1.0) < 1e-4
        ), f"Vecta recall@10 was {v['recall_at_k']}, expected 1.0 (100%)"
        assert (
            abs(f["recall_at_k"] - 1.0) < 1e-4
        ), f"FAISS recall@10 was {f['recall_at_k']}, expected 1.0 (100%)"
        assert comp["recall_discrepancy"] is False
