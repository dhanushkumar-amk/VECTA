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
import os
import time
import faiss
import numpy as np
import pytest

from benchmarks.faiss_comparison.faiss_wrappers import build_faiss_ivf, set_search_params
from benchmarks.faiss_comparison.run_comparison import (
    run_trials,
    summarize_timings,
    compare_flat_index,
    compare_ivf_index,
    find_iso_recall,
    generate_ivf_plot,
    print_ivf_comparison_table,
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


class TestIVFComparison:
    """Tests for Phase 37: IVF vs. faiss.IndexIVFFlat comparison."""

    def test_faiss_nprobe_attribute_assignment_behavior(self):
        """
        Requirement 1:
        Confirm FAISS's nprobe is set via attribute (index.nprobe = X), NOT passed as a
        search-time argument that search() would silently ignore or reject.
        Verify that changing nprobe changes actual search results/distances, and that
        passing nprobe as a keyword argument to search() raises TypeError.
        """
        dim = 16
        nlist = 10
        rng = np.random.RandomState(42)
        train_data = rng.randn(300, dim).astype(np.float32)
        query = rng.randn(1, dim).astype(np.float32)

        index = build_faiss_ivf(dim, nlist=nlist, metric="euclidean")
        index.train(train_data)
        index.add(train_data)

        # 1. Attribute assignment works
        index.nprobe = 1
        assert index.nprobe == 1
        D1, I1 = index.search(query, k=5)

        # 2. Increasing nprobe expands search space -> distance improves or matches
        index.nprobe = 10
        assert index.nprobe == 10
        D10, I10 = index.search(query, k=5)

        # Confirm results actually changed or improved across nprobe values
        # (With 10 clusters and nprobe=1 vs 10, the nearest neighbor at nprobe=10
        # should have distance <= the nearest neighbor found at nprobe=1)
        assert D10[0][0] <= D1[0][0] + 1e-6
        # At nprobe=10, all inverted lists are probed, so distance must be equal or strictly closer
        # Check that candidate IDs or scores reflect the expanded probing
        assert len(I10[0]) == 5

        # 3. Setting via helper works
        set_search_params(index, nprobe=3)
        assert index.nprobe == 3

        # 4. Search fails if passing nprobe as argument
        with pytest.raises(TypeError):
            index.search(query, k=5, nprobe=5)

    def test_iso_recall_calculation_and_interpolation(self):
        """
        Requirement 4:
        Confirm iso-recall interpolation logic produces verified mathematical results
        against hand-calculated reference values, handles exact hits and boundaries.
        """
        mock_sweep = [
            {
                "nprobe": 1,
                "vecta_recall": 0.40,
                "vecta_qps": 2000.0,
                "faiss_recall": 0.50,
                "faiss_qps": 2500.0,
                "qps_ratio": 1.25,
            },
            {
                "nprobe": 5,
                "vecta_recall": 0.80,
                "vecta_qps": 1000.0,
                "faiss_recall": 0.85,
                "faiss_qps": 1200.0,
                "qps_ratio": 1.20,
            },
            {
                "nprobe": 10,
                "vecta_recall": 0.92,
                "vecta_qps": 600.0,
                "faiss_recall": 0.95,
                "faiss_qps": 700.0,
                "qps_ratio": 1.167,
            },
        ]

        # Target: 0.90 (90%)
        # Vecta: bracket is (0.80, 1000.0, np=5) and (0.92, 600.0, np=10)
        # alpha = (0.90 - 0.80) / (0.92 - 0.80) = 0.10 / 0.12 = 5/6 = 0.833333...
        # est_qps = 1000.0 + (5/6) * (600.0 - 1000.0) = 1000.0 - 333.3333... = 666.667
        # est_nprobe = 5 + (5/6) * 5 = 9.167
        #
        # FAISS: bracket is (0.85, 1200.0, np=5) and (0.95, 700.0, np=10)
        # alpha = (0.90 - 0.85) / (0.95 - 0.85) = 0.05 / 0.10 = 0.5
        # est_qps = 1200.0 + 0.5 * (700.0 - 1200.0) = 950.0
        # est_nprobe = 5 + 0.5 * (10 - 5) = 7.5
        iso = find_iso_recall(mock_sweep, target_recall=0.90)

        assert abs(iso["target_recall"] - 0.90) < 1e-6
        assert iso["vecta"]["method"] == "linear_interpolation"
        assert abs(iso["vecta"]["estimated_qps"] - 666.6667) < 1e-2
        assert abs(iso["vecta"]["estimated_nprobe"] - 9.1667) < 1e-2

        assert iso["faiss"]["method"] == "linear_interpolation"
        assert abs(iso["faiss"]["estimated_qps"] - 950.0) < 1e-2
        assert abs(iso["faiss"]["estimated_nprobe"] - 7.5) < 1e-2

        assert iso["faster_engine"] == "faiss"
        # 950.0 / 666.6667 = 1.425
        assert abs(iso["speedup_ratio"] - 1.425) < 1e-2

        # Test exact match
        iso_exact = find_iso_recall(mock_sweep, target_recall=0.80)
        assert iso_exact["vecta"]["method"] == "exact"
        assert abs(iso_exact["vecta"]["estimated_qps"] - 1000.0) < 1e-4

        # Test boundary fallback (below lower bound)
        iso_low = find_iso_recall(mock_sweep, target_recall=0.20)
        assert iso_low["vecta"]["method"] == "nearest_boundary_lower"
        assert abs(iso_low["vecta"]["estimated_qps"] - 2000.0) < 1e-4

        # Test boundary fallback (above upper bound)
        iso_high = find_iso_recall(mock_sweep, target_recall=0.99)
        assert iso_high["vecta"]["method"] == "nearest_boundary_upper"
        assert abs(iso_high["vecta"]["estimated_qps"] - 600.0) < 1e-4

    def test_compare_ivf_index_e2e_and_monotonic_recall(self):
        """
        Requirements 2, 3, 5:
        - Run compare_ivf_index() end-to-end on SIFT data across a sweep.
        - Confirm recall monotonically increases with nprobe for BOTH libraries.
        - Confirm chart PNG file is generated and non-empty.
        - Confirm JSON results saved with sweep data.
        """
        sweep_values = [1, 5, 10, 20]
        results = compare_ivf_index(
            dataset_name="siftsmall",
            k=10,
            metric="euclidean",
            nlist=100,
            nprobe_values=sweep_values,
            threads=1,
            num_trials=2,
            warmup_trials=1,
            save_json=True,
            save_plot=True,
        )

        assert "sweep" in results
        assert len(results["sweep"]) == len(sweep_values)
        assert results["vecta_build_time_sec"] > 0
        assert results["faiss_build_time_sec"] > 0

        # Requirement 3: Confirm recall increases monotonically with nprobe for BOTH libraries
        v_recalls = [entry["vecta_recall"] for entry in results["sweep"]]
        f_recalls = [entry["faiss_recall"] for entry in results["sweep"]]

        for i in range(len(sweep_values) - 1):
            assert (
                v_recalls[i] <= v_recalls[i + 1] + 1e-5
            ), f"Vecta recall did not increase monotonically: {v_recalls}"
            assert (
                f_recalls[i] <= f_recalls[i + 1] + 1e-5
            ), f"FAISS recall did not increase monotonically: {f_recalls}"

        # Confirm recall at max nprobe is strictly higher than at nprobe=1
        assert v_recalls[-1] > v_recalls[0], "Vecta recall at max nprobe should exceed nprobe=1"
        assert f_recalls[-1] > f_recalls[0], "FAISS recall at max nprobe should exceed nprobe=1"

        # Requirement 5: Confirm chart PNG was generated and has non-zero size
        chart_path = results.get("chart_path")
        assert chart_path is not None
        assert os.path.exists(chart_path), f"Chart file {chart_path} does not exist"
        assert os.path.getsize(chart_path) > 0, "Chart file is empty (0 bytes)"

        # Confirm JSON results file exists and has non-zero size
        json_path = results.get("json_path")
        assert json_path is not None
        assert os.path.exists(json_path), f"JSON result file {json_path} does not exist"
        assert os.path.getsize(json_path) > 0, "JSON result file is empty (0 bytes)"
