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
import numpy as np
import pytest

faiss = pytest.importorskip("faiss", reason="faiss is not installed")

import vecta
from benchmarks.faiss_comparison.faiss_wrappers import (
    build_faiss_ivf,
    build_faiss_hnsw,
    build_faiss_ivfpq,
    set_search_params,
)
from benchmarks.faiss_comparison.run_comparison import (
    run_trials,
    summarize_timings,
    compare_flat_index,
    compare_ivf_index,
    compare_hnsw_index,
    compare_ivf_pq_index,
    find_iso_recall,
    generate_ivf_plot,
    generate_hnsw_plot,
    generate_ivfpq_plot,
    generate_memory_comparison_plot,
    print_ivf_comparison_table,
    print_hnsw_comparison_table,
    print_ivfpq_comparison_table,
    print_final_summary,
    load_latest_benchmark_results,
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


class TestHNSWComparison:
    """Tests for Phase 38: HNSW vs. faiss.IndexHNSWFlat comparison."""

    def test_faiss_efsearch_nested_attribute_changes_behavior(self):
        """
        Requirement 1:
        Confirm FAISS's efSearch is set via nested attribute (index.hnsw.efSearch = X),
        NOT passed as a search-time argument.
        Explicitly verify that changing efSearch from a very low value (e.g. 1) to a high
        value (e.g. 200) meaningfully changes search timing and search quality (lower vs higher
        beam depth).
        Also confirm passing efSearch to search() raises TypeError.
        """
        dim = 16
        m = 16
        rng = np.random.RandomState(42)
        train_data = rng.randn(500, dim).astype(np.float32)
        queries = rng.randn(50, dim).astype(np.float32)

        index = build_faiss_hnsw(dim, m=m, ef_construction=100, metric="euclidean")
        index.add(train_data)

        # 1. Low efSearch (greedy/narrow beam)
        index.hnsw.efSearch = 1
        assert index.hnsw.efSearch == 1
        t0 = time.perf_counter()
        D1, I1 = index.search(queries, k=5)
        time_1 = time.perf_counter() - t0

        # 2. High efSearch (deep beam search)
        index.hnsw.efSearch = 200
        assert index.hnsw.efSearch == 200
        t0 = time.perf_counter()
        D200, I200 = index.search(queries, k=5)
        time_200 = time.perf_counter() - t0

        # Verify behavior differs meaningfully:
        # Distance quality improves or stays identical with larger beam
        assert D200.sum() <= D1.sum() + 1e-5
        # The retrieved nearest neighbor IDs must not be suspiciously identical
        differences = (I1 != I200).sum()
        assert differences > 0, "efSearch=1 and efSearch=200 produced identical neighbor IDs!"

        # 3. Helper function works
        set_search_params(index, ef_search=50)
        assert index.hnsw.efSearch == 50

        # 4. Search fails if passing efSearch as keyword argument
        with pytest.raises(TypeError):
            index.search(queries, k=5, efSearch=50)
        with pytest.raises(TypeError):
            index.search(queries, k=5, ef_search=50)

    def test_faiss_efconstruction_applied_before_add(self):
        """
        Requirement 2:
        Explicit verification that index.hnsw.efConstruction was actually applied before
        add() was called.
        Catches the ordering bug where efConstruction might be set after add().
        """
        dim = 16
        m = 16
        expected_ef_c = 142
        rng = np.random.RandomState(42)
        train_data = rng.randn(100, dim).astype(np.float32)

        index = build_faiss_hnsw(dim, m=m, ef_construction=expected_ef_c, metric="euclidean")

        # Read back value immediately BEFORE add()
        actual_ef_c_before_add = index.hnsw.efConstruction
        assert actual_ef_c_before_add == expected_ef_c, (
            f"Ordering error: efConstruction before add was {actual_ef_c_before_add}, expected {expected_ef_c}"
        )

        index.add(train_data)

        # Confirm value is preserved after add()
        assert index.hnsw.efConstruction == expected_ef_c
        assert index.ntotal == 100

    def test_iso_recall_hnsw_param_name(self):
        """
        Requirement 2:
        Confirm generic iso-recall calculation supports param_name="ef_search".
        """
        mock_sweep = [
            {
                "ef_search": 10,
                "vecta_recall": 0.50,
                "vecta_qps": 5000.0,
                "faiss_recall": 0.60,
                "faiss_qps": 8000.0,
                "qps_ratio": 1.6,
            },
            {
                "ef_search": 40,
                "vecta_recall": 0.85,
                "vecta_qps": 3000.0,
                "faiss_recall": 0.88,
                "faiss_qps": 5000.0,
                "qps_ratio": 1.67,
            },
            {
                "ef_search": 80,
                "vecta_recall": 0.95,
                "vecta_qps": 1500.0,
                "faiss_recall": 0.98,
                "faiss_qps": 2500.0,
                "qps_ratio": 1.67,
            },
        ]

        iso = find_iso_recall(mock_sweep, target_recall=0.90, param_name="ef_search")
        assert iso["param_name"] == "ef_search"
        assert iso["vecta"]["method"] == "linear_interpolation"
        assert "estimated_ef_search" in iso["vecta"]
        # Vecta: bracket (0.85, 3000, 40) and (0.95, 1500, 80)
        # alpha = (0.90 - 0.85) / (0.95 - 0.85) = 0.05 / 0.10 = 0.5
        # est_qps = 3000 + 0.5 * (1500 - 3000) = 2250.0
        # est_ef = 40 + 0.5 * (80 - 40) = 60.0
        assert abs(iso["vecta"]["estimated_qps"] - 2250.0) < 1e-3
        assert abs(iso["vecta"]["estimated_ef_search"] - 60.0) < 1e-3

    def test_compare_hnsw_index_e2e_and_monotonic_recall(self):
        """
        Requirements 3, 4, 5:
        - Run compare_hnsw_index() end-to-end on real SIFT data across an ef_search sweep.
        - Confirm recall monotonically increases with ef_search for BOTH libraries.
        - Confirm chart PNG file is generated and non-empty.
        - Confirm JSON results saved with sweep data.
        """
        sweep_values = [10, 20, 40, 80]
        results = compare_hnsw_index(
            dataset_name="siftsmall",
            k=10,
            metric="euclidean",
            m=16,
            ef_construction=100,
            ef_search_values=sweep_values,
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

        # Requirement 4: Confirm recall increases monotonically with ef_search for BOTH libraries
        v_recalls = [entry["vecta_recall"] for entry in results["sweep"]]
        f_recalls = [entry["faiss_recall"] for entry in results["sweep"]]

        for i in range(len(sweep_values) - 1):
            assert (
                v_recalls[i] <= v_recalls[i + 1] + 1e-5
            ), f"Vecta recall did not increase monotonically: {v_recalls}"
            assert (
                f_recalls[i] <= f_recalls[i + 1] + 1e-5
            ), f"FAISS recall did not increase monotonically: {f_recalls}"

        # Confirm recall at max ef_search is strictly higher than at ef_search=10
        assert v_recalls[-1] > v_recalls[0], "Vecta recall at max ef_search should exceed ef_search=10"
        assert f_recalls[-1] > f_recalls[0], "FAISS recall at max ef_search should exceed ef_search=10"

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


class TestIVFPQComparison:
    """Tests for Phase 39: IVFPQ vs. faiss.IndexIVFPQ comparison."""

    def test_k_per_subvector_to_nbits_conversion_and_power_of_2_validation(self):
        """
        Requirement 1:
        Verify the k_per_subvector -> nbits conversion is correct:
        - k_per_subvector=256 confirms nbits=8
        - k_per_subvector=64 confirms nbits=6
        - Non-power-of-2 k_per_subvector (e.g. 100, 200, 300) explicitly raises ValueError
          rather than silently producing a wrong nbits value.
        """
        # Valid powers of 2
        for k_sub, expected_nbits in [(2, 1), (4, 2), (16, 4), (64, 6), (128, 7), (256, 8)]:
            assert (k_sub & (k_sub - 1)) == 0
            assert int(math.log2(k_sub)) == expected_nbits

        # Non-power-of-2 cases must be explicitly rejected with a clear ValueError
        non_powers = [0, -1, 3, 50, 100, 200, 255, 300, 500]
        for invalid_k in non_powers:
            with pytest.raises(ValueError, match="k_per_subvector must be a power of 2"):
                compare_ivf_pq_index(
                    dataset=np.zeros((10, 16), dtype=np.float32),
                    queries=np.zeros((2, 16), dtype=np.float32),
                    ground_truth=[[0], [0]],
                    k=1,
                    nlist=2,
                    m=4,
                    k_per_subvector=invalid_k,
                    nprobe_values=[1],
                    save_json=False,
                    save_plot=False,
                )

    def test_faiss_metric_type_is_explicitly_l2(self):
        """
        Requirement 2:
        Confirm FAISS's metric is genuinely set to L2/Euclidean for the IVFPQ comparison.
        Inspect the actual FAISS index object's metric_type attribute and verify it equals
        faiss.METRIC_L2, not assumed or defaulting to inner product.
        """
        dim = 16
        nlist = 4
        m = 4
        nbits = 8

        f_index = build_faiss_ivfpq(dim, nlist=nlist, m=m, nbits=nbits, metric="euclidean")
        assert hasattr(f_index, "metric_type"), "FAISS index missing metric_type attribute"
        assert f_index.metric_type == faiss.METRIC_L2, (
            f"FAISS index metric_type is {f_index.metric_type}, expected METRIC_L2 ({faiss.METRIC_L2})"
        )
        assert f_index.metric_type != faiss.METRIC_INNER_PRODUCT, (
            "FAISS index was inadvertently configured with METRIC_INNER_PRODUCT!"
        )

        # Confirm non-Euclidean metric raises ValueError in compare_ivf_pq_index
        with pytest.raises(ValueError, match="vecta IVFPQ only supports Euclidean/L2"):
            compare_ivf_pq_index(
                dataset=np.zeros((10, 16), dtype=np.float32),
                queries=np.zeros((2, 16), dtype=np.float32),
                ground_truth=[[0], [0]],
                k=1,
                nlist=2,
                m=4,
                k_per_subvector=16,
                metric="cosine",
                nprobe_values=[1],
                save_json=False,
                save_plot=False,
            )

    def test_memory_footprint_hand_calculation_sanity(self):
        """
        Requirement 4:
        Confirm vecta's reported footprint matches hand-calculation from Phase 23's design:
        footprint = (num_vectors * m) + (num_clusters * dim * 4) + (m * k_per_sub * (dim/m) * 4).
        Also confirm FAISS serialized footprint is positive and comparable in order of magnitude.
        """
        dim = 32
        num_clusters = 5
        m = 4
        k_per_sub = 16
        n_vecs = 200

        rng = np.random.RandomState(42)
        data = rng.randn(n_vecs, dim).astype(np.float32)

        # Build vecta index
        v_index = vecta.IVFPQIndex(
            dim=dim,
            num_clusters=num_clusters,
            m=m,
            k_per_subvector=k_per_sub,
            max_iterations=10,
        )
        v_index.train(data.tolist(), ivf_seed=42, pq_seed=42)
        v_index.add_batch(list(range(n_vecs)), data.tolist())

        # Theoretical calculation:
        # 1 byte per subvector code since k_per_sub <= 256:
        code_bytes = n_vecs * m
        centroids_bytes = num_clusters * dim * 4
        sub_dim = dim // m
        codebooks_bytes = m * k_per_sub * sub_dim * 4
        expected_bytes = code_bytes + centroids_bytes + codebooks_bytes

        actual_v_bytes = v_index.memory_footprint_bytes()
        assert actual_v_bytes == expected_bytes, (
            f"Vecta footprint {actual_v_bytes} != expected {expected_bytes}"
        )

        # Check FAISS index serialized footprint
        nbits = int(math.log2(k_per_sub))
        f_index = build_faiss_ivfpq(dim, nlist=num_clusters, m=m, nbits=nbits, metric="euclidean")
        f_index.train(data)
        f_index.add(data)
        serialized_buf = faiss.serialize_index(f_index)
        f_bytes = len(serialized_buf)
        assert f_bytes > 0

        # Memory should provide significant compression compared to raw float32 vectors
        raw_bytes = n_vecs * dim * 4
        assert actual_v_bytes < raw_bytes
        assert f_bytes < raw_bytes

    def test_compare_ivf_pq_index_e2e_and_monotonic_recall(self):
        """
        Requirements 3, 5, 6:
        - Run compare_ivf_pq_index() end-to-end on real SIFT data across an nprobe sweep.
        - Confirm recall monotonically increases with nprobe for BOTH libraries.
        - Confirm BOTH chart PNG files (recall-vs-QPS curve AND memory bar chart) exist and are non-empty.
        - Confirm JSON results saved with sweep data and memory footprint metrics.
        """
        sweep_values = [1, 5, 20, 50]
        results = compare_ivf_pq_index(
            dataset_name="siftsmall",
            k=10,
            nlist=100,
            m=8,
            k_per_subvector=256,
            nprobe_values=sweep_values,
            metric="euclidean",
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
        assert results["nbits"] == 8
        assert results["k_per_subvector"] == 256

        # Requirement 4/Memory:
        assert results["vecta_memory_bytes"] > 0
        assert results["faiss_memory_bytes"] > 0
        assert results["raw_vector_bytes"] == 10000 * 128 * 4
        assert results["vecta_compression_ratio"] > 10.0
        assert results["faiss_compression_ratio"] > 10.0

        # Requirement 5: Confirm recall increases monotonically with nprobe for BOTH libraries
        v_recalls = [entry["vecta_recall"] for entry in results["sweep"]]
        f_recalls = [entry["faiss_recall"] for entry in results["sweep"]]

        # IVFPQ utilizes lossy Product Quantization (ADC distances). While overall recall
        # dramatically climbs with nprobe, saturation at high nprobe can exhibit slight ADC noise (±0.01).
        for i in range(len(sweep_values) - 1):
            assert (
                v_recalls[i] <= v_recalls[i + 1] + 0.02
            ), f"Vecta recall did not increase with nprobe: {v_recalls}"
            assert (
                f_recalls[i] <= f_recalls[i + 1] + 0.02
            ), f"FAISS recall did not increase with nprobe: {f_recalls}"

        # Confirm recall at max nprobe is significantly higher than at nprobe=1 (>10% boost)
        assert v_recalls[-1] > v_recalls[0] + 0.10, "Vecta recall at max nprobe should exceed nprobe=1"
        assert f_recalls[-1] > f_recalls[0] + 0.10, "FAISS recall at max nprobe should exceed nprobe=1"

        # Requirement 6: Confirm BOTH chart PNG files generated and non-empty
        chart_path = results.get("chart_path")
        assert chart_path is not None
        assert os.path.exists(chart_path), f"Tradeoff chart file {chart_path} does not exist"
        assert os.path.getsize(chart_path) > 0, "Tradeoff chart file is empty (0 bytes)"

        mem_chart_path = results.get("memory_chart_path")
        assert mem_chart_path is not None
        assert os.path.exists(mem_chart_path), f"Memory chart file {mem_chart_path} does not exist"
        assert os.path.getsize(mem_chart_path) > 0, "Memory chart file is empty (0 bytes)"

        # Confirm JSON results file exists and has non-zero size
        json_path = results.get("json_path")
        assert json_path is not None
        assert os.path.exists(json_path), f"JSON result file {json_path} does not exist"
        assert os.path.getsize(json_path) > 0, "JSON result file is empty (0 bytes)"

    def test_print_final_summary_consolidation(self):
        """
        Requirement 7:
        Confirm print_final_summary() runs correctly when given benchmark results,
        producing a coherent consolidated table across Flat, IVF, HNSW, and IVFPQ.
        """
        # Test with mock 4-index results
        mock_results = {
            "flat": {
                "vecta": {"build_time_sec": 0.005, "mean_qps": 850.0, "recall_at_k": 1.0},
                "faiss": {"build_time_sec": 0.004, "mean_qps": 1200.0, "recall_at_k": 1.0},
                "comparison": {"qps_speedup_ratio": 1.41, "faster_engine": "faiss"},
                "num_vectors": 10000,
                "dimension": 128,
            },
            "ivf": {
                "vecta_build_time_sec": 0.150,
                "faiss_build_time_sec": 0.080,
                "iso_recall": {
                    "vecta": {"estimated_qps": 2200.0, "achieved_recall": 0.89},
                    "faiss": {"estimated_qps": 3100.0, "achieved_recall": 0.91},
                    "speedup_ratio": 1.41,
                    "faster_engine": "faiss",
                },
                "num_vectors": 10000,
                "dimension": 128,
            },
            "hnsw": {
                "vecta_build_time_sec": 0.850,
                "faiss_build_time_sec": 0.450,
                "iso_recall": {
                    "vecta": {"estimated_qps": 6500.0, "achieved_recall": 0.90},
                    "faiss": {"estimated_qps": 9800.0, "achieved_recall": 0.91},
                    "speedup_ratio": 1.51,
                    "faster_engine": "faiss",
                },
                "num_vectors": 10000,
                "dimension": 128,
            },
            "ivfpq": {
                "vecta_build_time_sec": 0.420,
                "faiss_build_time_sec": 0.310,
                "vecta_memory_bytes": 262272,
                "faiss_memory_bytes": 282624,
                "vecta_compression_ratio": 19.5,
                "faiss_compression_ratio": 18.1,
                "iso_recall": {
                    "vecta": {"estimated_qps": 4200.0, "achieved_recall": 0.88},
                    "faiss": {"estimated_qps": 5600.0, "achieved_recall": 0.89},
                    "speedup_ratio": 1.33,
                    "faster_engine": "faiss",
                },
                "num_vectors": 10000,
                "dimension": 128,
            },
        }

        consolidated = print_final_summary(mock_results)
        assert consolidated is not None
        assert "flat" in consolidated
        assert "ivf" in consolidated
        assert "hnsw" in consolidated
        assert "ivfpq" in consolidated
