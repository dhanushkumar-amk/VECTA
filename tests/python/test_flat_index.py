"""
Comprehensive tests for FlatIndex Python bindings (Phase 7).

Verifies panic safety, error handling, correctness oracle consistency,
and numpy cross-checking across all metrics.
"""

import pytest
import vecta
import numpy as np


class TestFlatIndex:
    """Test suite for vecta.FlatIndex Python bindings."""

    # 1. Creating a FlatIndex with each of the 3 metric strings succeeds
    @pytest.mark.parametrize(
        "metric",
        ["euclidean", "EUCLIDEAN", "l2", "cosine", "COSINE", "cos", "dot_product", "dot", "ip"],
    )
    def test_create_valid_metrics(self, metric):
        index = vecta.FlatIndex(dim=4, metric=metric)
        assert len(index) == 0
        assert index.is_empty()
        assert index.dim() == 4

    # 2. Creating a FlatIndex with an invalid metric string raises ValueError
    @pytest.mark.parametrize("invalid_metric", ["manhattan", "hamming", "random_string", ""])
    def test_create_invalid_metric_raises(self, invalid_metric):
        with pytest.raises(ValueError, match="unknown metric"):
            vecta.FlatIndex(dim=4, metric=invalid_metric)

    def test_create_zero_dim_raises(self):
        with pytest.raises(ValueError, match="dimension must be greater than 0"):
            vecta.FlatIndex(dim=0, metric="euclidean")

    # 3. Adding vectors one at a time, then checking len(index) matches
    def test_add_single_vectors(self):
        index = vecta.FlatIndex(dim=3, metric="euclidean")
        assert len(index) == 0
        assert index.is_empty()

        index.add(10, [1.0, 2.0, 3.0])
        assert len(index) == 1
        assert not index.is_empty()

        index.add(20, [4.0, 5.0, 6.0])
        assert len(index) == 2

        index.add(30, [7.0, 8.0, 9.0])
        assert len(index) == 3

    # 4. Adding a vector with wrong dimension raises a Python exception (NOT crash)
    def test_add_wrong_dimension_raises(self):
        index = vecta.FlatIndex(dim=3, metric="euclidean")
        with pytest.raises(ValueError, match="dimension mismatch"):
            index.add(1, [1.0, 2.0])  # dim 2 instead of 3

        with pytest.raises(ValueError, match="dimension mismatch"):
            index.add(2, [1.0, 2.0, 3.0, 4.0])  # dim 4 instead of 3

        # Confirm the index is still intact and empty
        assert len(index) == 0

    # 5. Adding a duplicate id raises a Python exception, not a crash
    def test_add_duplicate_id_raises(self):
        index = vecta.FlatIndex(dim=2, metric="euclidean")
        index.add(42, [1.0, 2.0])
        assert len(index) == 1

        with pytest.raises(ValueError, match="duplicate id 42"):
            index.add(42, [3.0, 4.0])

        assert len(index) == 1

    # 6. add_batch() with valid data works, len(index) updates correctly
    def test_add_batch_valid(self):
        index = vecta.FlatIndex(dim=2, metric="euclidean")
        ids = [100, 101, 102]
        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        index.add_batch(ids, vectors)
        assert len(index) == 3

        # Add more via add_batch
        index.add_batch([103, 104], [[2.0, 2.0], [3.0, 3.0]])
        assert len(index) == 5

    def test_add_batch_mismatched_counts_raises(self):
        index = vecta.FlatIndex(dim=2, metric="euclidean")
        with pytest.raises(ValueError, match="ids count .* != vectors count"):
            index.add_batch([1, 2], [[1.0, 2.0]])

    def test_add_batch_wrong_vector_dim_raises(self):
        index = vecta.FlatIndex(dim=2, metric="euclidean")
        with pytest.raises(ValueError, match="dimension"):
            index.add_batch([1, 2], [[1.0, 2.0], [3.0]])
        assert len(index) == 0

    def test_add_batch_duplicate_id_raises(self):
        index = vecta.FlatIndex(dim=2, metric="euclidean")
        index.add(1, [1.0, 1.0])

        # Duplicate with existing
        with pytest.raises(ValueError, match="duplicate id 1"):
            index.add_batch([1, 2], [[2.0, 2.0], [3.0, 3.0]])

        # Duplicate within batch
        with pytest.raises(ValueError, match="duplicate id 5"):
            index.add_batch([5, 5], [[2.0, 2.0], [3.0, 3.0]])

    # 7. search() returns correctly-shaped results: list of (id, score) tuples, len == k (or fewer)
    def test_search_result_shapes(self):
        index = vecta.FlatIndex(dim=2, metric="euclidean")
        index.add_batch([1, 2, 3], [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])

        # k = 2
        results = index.search([1.0, 0.0], k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)

        # k > len(index) returns all vectors
        results_all = index.search([1.0, 0.0], k=10)
        assert len(results_all) == 3

        # k = 0 returns empty list
        results_zero = index.search([1.0, 0.0], k=0)
        assert results_zero == []

        # empty index returns empty list
        empty_index = vecta.FlatIndex(dim=2, metric="euclidean")
        assert empty_index.search([1.0, 0.0], k=5) == []

    # 8. search() with wrong-dimension query raises a Python exception, not a crash
    def test_search_wrong_dimension_raises(self):
        index = vecta.FlatIndex(dim=3, metric="euclidean")
        index.add(1, [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="query dimension mismatch"):
            index.search([1.0, 2.0], k=1)

        with pytest.raises(ValueError, match="query dimension mismatch"):
            index.search([1.0, 2.0, 3.0, 4.0], k=1)

    # 9. Integration test: build an index with 5 known vectors (same hand-verified data from Phase 6)
    def test_search_hand_verified_phase6_data(self):
        # 5 vectors from Phase 6:
        # v0 (id=0): [2.0, 1.0]
        # v1 (id=1): [0.5, 0.0]
        # v2 (id=2): [1.0, 3.0]
        # v3 (id=3): [-1.0, 3.0]
        # v4 (id=4): [4.0, -3.0]
        # Query: [1.0, 1.0]
        ids = [0, 1, 2, 3, 4]
        vectors = [
            [2.0, 1.0],
            [0.5, 0.0],
            [1.0, 3.0],
            [-1.0, 3.0],
            [4.0, -3.0],
        ]
        query = [1.0, 1.0]

        # (a) Euclidean: expected top-3 IDs: [0, 1, 2]
        # d(v0)=1.0, d(v1)=sqrt(1.25)~1.1180, d(v2)=2.0
        idx_eucl = vecta.FlatIndex(dim=2, metric="euclidean")
        idx_eucl.add_batch(ids, vectors)
        res_eucl = idx_eucl.search(query, k=3)
        res_ids_eucl = [r[0] for r in res_eucl]
        assert res_ids_eucl == [0, 1, 2]
        assert pytest.approx(res_eucl[0][1], abs=1e-4) == 1.0
        assert pytest.approx(res_eucl[1][1], abs=1e-4) == 1.25**0.5
        assert pytest.approx(res_eucl[2][1], abs=1e-4) == 2.0

        # (b) Cosine: expected top-3 IDs: [0, 2, 1]
        # cos(v0)=3/sqrt(10)~0.9487, cos(v2)=2/sqrt(5)~0.8944, cos(v1)=1/sqrt(2)~0.7071
        idx_cos = vecta.FlatIndex(dim=2, metric="cosine")
        idx_cos.add_batch(ids, vectors)
        res_cos = idx_cos.search(query, k=3)
        res_ids_cos = [r[0] for r in res_cos]
        assert res_ids_cos == [0, 2, 1]
        assert pytest.approx(res_cos[0][1], abs=1e-4) == 3.0 / (10.0**0.5)
        assert pytest.approx(res_cos[1][1], abs=1e-4) == 2.0 / (5.0**0.5)
        assert pytest.approx(res_cos[2][1], abs=1e-4) == 1.0 / (2.0**0.5)

        # (c) Dot Product: expected top-3 IDs: [2, 0, 3]
        # dot(v2)=4.0, dot(v0)=3.0, dot(v3)=2.0
        idx_dp = vecta.FlatIndex(dim=2, metric="dot_product")
        idx_dp.add_batch(ids, vectors)
        res_dp = idx_dp.search(query, k=3)
        res_ids_dp = [r[0] for r in res_dp]
        assert res_ids_dp == [2, 0, 3]
        assert pytest.approx(res_dp[0][1], abs=1e-4) == 4.0
        assert pytest.approx(res_dp[1][1], abs=1e-4) == 3.0
        assert pytest.approx(res_dp[2][1], abs=1e-4) == 2.0

    # 10. Comparison test against numpy
    def test_comparison_against_numpy(self):
        np.random.seed(42)
        dim = 32
        n = 200
        k = 5

        # Generate random dataset
        dataset = np.random.randn(n, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)

        # Compute nearest neighbors using numpy L2 norm
        dists = np.linalg.norm(dataset - query, axis=1)
        expected_indices = np.argsort(dists)[:k]
        expected_scores = dists[expected_indices]

        # Compute nearest neighbors using vecta FlatIndex
        index = vecta.FlatIndex(dim=dim, metric="euclidean")
        ids = list(range(n))
        vectors = dataset.tolist()
        index.add_batch(ids, vectors)

        vecta_results = index.search(query.tolist(), k=k)

        # Compare top-k IDs and scores
        vecta_ids = [r[0] for r in vecta_results]
        vecta_scores = [r[1] for r in vecta_results]

        assert vecta_ids == expected_indices.tolist()
        for v_score, exp_score in zip(vecta_scores, expected_scores):
            assert pytest.approx(v_score, abs=1e-5) == exp_score
