"""Unit and integration tests for vecta.HnswIndex Python bindings (Phase 19)."""

import pytest
import vecta


class TestHnswIndexConstructor:
    """Test category 1: construction and parameter validation."""

    def test_create_default_parameters_success(self):
        index = vecta.HnswIndex(dim=128)
        assert len(index) == 0
        assert index.is_empty()
        assert index.dim() == 128

    @pytest.mark.parametrize("metric", ["euclidean", "EUCLIDEAN", "l2", "cosine", "dot_product", "dot", "ip"])
    def test_create_valid_metrics(self, metric):
        index = vecta.HnswIndex(dim=64, metric=metric)
        assert index.dim() == 64

    @pytest.mark.parametrize("invalid_metric", ["manhattan", "hamming", "chebyshev", "invalid"])
    def test_create_invalid_metric_raises(self, invalid_metric):
        with pytest.raises(ValueError, match="unknown metric"):
            vecta.HnswIndex(dim=64, metric=invalid_metric)

    def test_create_zero_dim_raises(self):
        with pytest.raises(ValueError, match="dimension must be greater than 0"):
            vecta.HnswIndex(dim=0)

    def test_create_invalid_m_raises(self):
        with pytest.raises(ValueError, match="m must be greater than 1"):
            vecta.HnswIndex(dim=16, m=1)

    def test_create_zero_ef_construction_raises(self):
        with pytest.raises(ValueError, match="ef_construction must be greater than 0"):
            vecta.HnswIndex(dim=16, ef_construction=0)


class TestHnswIndexOperations:
    """Test categories 2, 3, 4: add, add_batch, search, error handling."""

    def test_add_and_search_end_to_end_shapes(self):
        index = vecta.HnswIndex(dim=3, metric="euclidean")
        index.add(1, [1.0, 0.0, 0.0])
        index.add(2, [0.0, 1.0, 0.0])
        index.add(3, [0.0, 0.0, 1.0])

        assert len(index) == 3
        assert not index.is_empty()

        results = index.search([1.0, 0.1, 0.0], k=2)
        assert isinstance(results, list)
        assert len(results) == 2

        top_id, top_score = results[0]
        assert top_id == 1
        assert isinstance(top_score, float)
        assert top_score >= 0.0

    def test_duplicate_id_raises_value_error(self):
        index = vecta.HnswIndex(dim=2)
        index.add(42, [1.0, 2.0])
        with pytest.raises(ValueError, match="duplicate id 42"):
            index.add(42, [3.0, 4.0])
        assert len(index) == 1

    def test_add_batch_duplicate_id_raises_value_error(self):
        index = vecta.HnswIndex(dim=2)
        index.add(10, [1.0, 1.0])
        with pytest.raises(ValueError, match="duplicate id 10"):
            index.add_batch([20, 10], [[2.0, 2.0], [3.0, 3.0]])

    def test_add_batch_mismatched_counts_raises(self):
        index = vecta.HnswIndex(dim=2)
        with pytest.raises(ValueError, match="ids count"):
            index.add_batch([1, 2], [[1.0, 1.0]])

    def test_add_wrong_dimension_raises_value_error(self):
        index = vecta.HnswIndex(dim=3)
        with pytest.raises(ValueError, match="vector dimension mismatch: expected 3, got 2"):
            index.add(1, [1.0, 2.0])

    def test_add_batch_wrong_dimension_raises_value_error(self):
        index = vecta.HnswIndex(dim=3)
        with pytest.raises(ValueError, match="vector at index 1 has dimension 2, expected 3"):
            index.add_batch([1, 2], [[1.0, 2.0, 3.0], [4.0, 5.0]])

    def test_search_wrong_dimension_raises_value_error(self):
        index = vecta.HnswIndex(dim=3)
        index.add(1, [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="query dimension mismatch: expected 3, got 4"):
            index.search([1.0, 2.0, 3.0, 4.0], k=1)


class TestHnswReproducibility:
    """Test category 5: persistent seeded RNG guarantees reproducible graphs."""

    def test_same_seed_produces_identical_search_results(self):
        dim = 8
        seed = 98765

        # Create two distinct instances with identical seed
        idx1 = vecta.HnswIndex(dim=dim, seed=seed)
        idx2 = vecta.HnswIndex(dim=dim, seed=seed)

        vectors = [
            (i, [float(i * j) for j in range(dim)])
            for i in range(1, 30)
        ]

        # Insert one by one to verify persistent state across add() calls
        for vid, vec in vectors:
            idx1.add(vid, vec)
            idx2.add(vid, vec)

        query = [float(j * 2.5) for j in range(dim)]
        res1 = idx1.search(query, k=5)
        res2 = idx2.search(query, k=5)

        assert len(res1) == 5
        assert len(res2) == 5
        for (id1, score1), (id2, score2) in zip(res1, res2):
            assert id1 == id2
            assert abs(score1 - score2) < 1e-6


class TestHnswIntegrationVsFlatIndex:
    """Test category 6: hand-verified dataset against FlatIndex oracle."""

    def test_hand_verified_dataset_matches_flat_index(self):
        dim = 2
        flat = vecta.FlatIndex(dim=dim, metric="euclidean")
        hnsw = vecta.HnswIndex(dim=dim, metric="euclidean", m=2, ef_construction=50, ef_search=50, seed=1234)

        points = [
            (1, [1.0, 1.0]),
            (2, [1.0, 1.5]),
            (3, [1.5, 1.0]),
            (4, [9.0, 9.0]),
            (5, [9.0, 8.5]),
            (6, [8.5, 9.0]),
        ]

        for pid, pt in points:
            flat.add(pid, pt)
            hnsw.add(pid, pt)

        query = [1.1, 1.2]
        k = len(points)

        flat_res = flat.search(query, k)
        hnsw_res = hnsw.search(query, k, ef_search=50)

        assert len(hnsw_res) == len(flat_res)
        for rank in range(k):
            assert hnsw_res[rank][0] == flat_res[rank][0], f"ID mismatch at rank {rank}"
            assert abs(hnsw_res[rank][1] - flat_res[rank][1]) < 1e-5, f"Score mismatch at rank {rank}"


class TestHnswEfSearchOverride:
    """Test category 7: ef_search override parameter."""

    def test_ef_search_override_parameter(self):
        # Create an index with a large default ef_search
        index = vecta.HnswIndex(dim=2, ef_search=100)
        index.add(1, [1.0, 1.0])
        index.add(2, [2.0, 2.0])

        query = [1.1, 1.1]

        # Call with default (None) -> uses ef_search=100
        res_default = index.search(query, k=2, ef_search=None)
        # Call with explicit override -> uses ef_search=1
        res_override = index.search(query, k=2, ef_search=1)

        assert len(res_default) > 0
        assert len(res_override) > 0


class TestHnswDiagnostics:
    """Test category 8: max_layer_distribution() histogram."""

    def test_max_layer_distribution_sum_equals_total_count(self):
        dim = 4
        index = vecta.HnswIndex(dim=dim, seed=42)

        n = 100
        ids = list(range(n))
        vectors = [[float(i + j) for j in range(dim)] for i in range(n)]

        index.add_batch(ids, vectors)
        assert len(index) == n

        dist = index.max_layer_distribution()
        assert isinstance(dist, dict)
        assert len(dist) > 0

        total_nodes = sum(dist.values())
        assert total_nodes == n

        # Layer 0 must have the highest count in the pyramid
        assert 0 in dist
        assert dist[0] > (n * 0.7)
