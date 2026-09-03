"""
Unit and integration tests for IVFIndex Python bindings (Phase 14).

Validates constructor parameters, safe exception handling (no crashes across FFI),
hand-verified routing and search correctness against FlatIndex, and cluster diagnostics.
"""

import math
import pytest
import vecta


class TestIVFIndexConstructor:
    """Test category 1: Construction and parameter validation."""

    @pytest.mark.parametrize(
        "metric",
        [
            "euclidean",
            "EUCLIDEAN",
            "l2",
            "cosine",
            "COSINE",
            "cos",
            "dot_product",
            "dot",
            "ip",
        ],
    )
    def test_create_valid_metrics(self, metric: str):
        index = vecta.IVFIndex(dim=128, num_clusters=8, metric=metric)
        assert index.dim() == 128
        assert index.num_clusters() == 8
        assert len(index) == 0
        assert index.is_empty()
        assert not index.is_trained()

    @pytest.mark.parametrize("invalid_metric", ["manhattan", "hamming", "random_string", ""])
    def test_create_invalid_metric_raises(self, invalid_metric: str):
        with pytest.raises(ValueError, match="unknown metric"):
            vecta.IVFIndex(dim=128, num_clusters=8, metric=invalid_metric)

    def test_create_zero_dim_raises(self):
        with pytest.raises(ValueError, match="dimension must be greater than 0"):
            vecta.IVFIndex(dim=0, num_clusters=8, metric="euclidean")

    def test_create_zero_clusters_raises(self):
        with pytest.raises(ValueError, match="num_clusters must be greater than 0"):
            vecta.IVFIndex(dim=128, num_clusters=0, metric="euclidean")


class TestIVFTrainedSafety:
    """Test category 2: Calling operations prior to train() safely raises ValueError."""

    def test_add_before_train_raises(self):
        index = vecta.IVFIndex(dim=2, num_clusters=2, metric="euclidean")
        with pytest.raises(ValueError, match="IVFIndex must be trained before adding vectors"):
            index.add(1, [1.0, 2.0])

    def test_add_batch_before_train_raises(self):
        index = vecta.IVFIndex(dim=2, num_clusters=2, metric="euclidean")
        with pytest.raises(ValueError, match="IVFIndex must be trained before adding vectors"):
            index.add_batch([1], [[1.0, 2.0]])

    def test_search_before_train_raises(self):
        index = vecta.IVFIndex(dim=2, num_clusters=2, metric="euclidean")
        with pytest.raises(ValueError, match="IVFIndex must be trained before searching"):
            index.search([1.0, 2.0], k=2, nprobe=1)

    def test_nprobe_coverage_before_train_raises(self):
        index = vecta.IVFIndex(dim=2, num_clusters=2, metric="euclidean")
        with pytest.raises(ValueError, match="IVFIndex must be trained before querying coverage"):
            index.nprobe_coverage([1.0, 2.0], nprobe=1)


class TestIVFEndToEnd:
    """Test category 3: train -> add -> search end-to-end lifecycle."""

    def test_train_add_search_lifecycle(self):
        index = vecta.IVFIndex(dim=2, num_clusters=2, metric="euclidean")

        train_data = [
            [1.0, 1.0],
            [1.0, 2.0],
            [9.0, 9.0],
            [9.0, 8.0],
        ]
        index.train(train_data, k=2, max_iterations=50, seed=42)
        assert index.is_trained()

        # Training data is NOT automatically added
        assert len(index) == 0

        # Add single vectors
        index.add(10, [1.0, 1.0])
        index.add(20, [9.0, 9.0])
        assert len(index) == 2

        # Search
        results = index.search([1.0, 1.0], k=2, nprobe=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)

        # Nearest to [1, 1] is ID 10 with distance 0.0
        assert results[0][0] == 10
        assert abs(results[0][1] - 0.0) < 1e-5


class TestIVFDimensionSafety:
    """Test category 4: Dimension safety across all entrypoints (no panics)."""

    @pytest.fixture
    def trained_index(self):
        index = vecta.IVFIndex(dim=3, num_clusters=2, metric="euclidean")
        train_data = [
            [1.0, 2.0, 3.0],
            [9.0, 8.0, 7.0],
        ]
        index.train(train_data, k=2, max_iterations=10, seed=42)
        return index

    def test_train_wrong_dimension_raises(self):
        index = vecta.IVFIndex(dim=3, num_clusters=2, metric="euclidean")
        with pytest.raises(ValueError, match="training vector at index 1 has dimension 2, expected 3"):
            index.train([[1.0, 2.0, 3.0], [4.0, 5.0]], k=2, max_iterations=10, seed=42)

    def test_train_mismatched_k_raises(self):
        index = vecta.IVFIndex(dim=3, num_clusters=2, metric="euclidean")
        with pytest.raises(ValueError, match="k .* must equal index num_clusters"):
            index.train([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]], k=5, max_iterations=10, seed=42)

    def test_add_wrong_dimension_raises(self, trained_index):
        with pytest.raises(ValueError, match="vector dimension mismatch: expected 3, got 2"):
            trained_index.add(1, [1.0, 2.0])

    def test_add_batch_wrong_dimension_raises(self, trained_index):
        with pytest.raises(ValueError, match="vector at index 1 has dimension 2, expected 3"):
            trained_index.add_batch([1, 2], [[1.0, 2.0, 3.0], [4.0, 5.0]])

    def test_search_wrong_dimension_raises(self, trained_index):
        with pytest.raises(ValueError, match="query dimension mismatch: expected 3, got 2"):
            trained_index.search([1.0, 2.0], k=5, nprobe=1)

    def test_nprobe_coverage_wrong_dimension_raises(self, trained_index):
        with pytest.raises(ValueError, match="query dimension mismatch: expected 3, got 2"):
            trained_index.nprobe_coverage([1.0, 2.0], nprobe=1)


class TestIVFIntegrationVsFlatIndex:
    """Test category 5: Integration / correctness check against FlatIndex ground truth."""

    def test_hand_verified_dataset_matches_flat_index(self):
        points = [
            (1, [1.0, 1.0]),
            (2, [1.0, 2.0]),
            (3, [1.5, 1.5]),
            (4, [9.0, 9.0]),
            (5, [9.0, 8.0]),
            (6, [8.5, 8.5]),
        ]

        ids = [p[0] for p in points]
        vectors = [p[1] for p in points]

        # 1. Populate FlatIndex (Oracle)
        flat = vecta.FlatIndex(dim=2, metric="euclidean")
        flat.add_batch(ids, vectors)

        # 2. Populate IVFIndex
        ivf = vecta.IVFIndex(dim=2, num_clusters=2, metric="euclidean")
        ivf.train(vectors, k=2, max_iterations=50, seed=42)
        ivf.add_batch(ids, vectors)

        query = [1.1, 1.2]
        k = 3

        flat_res = flat.search(query, k=k)
        # With nprobe=2 (all clusters), IVF search is exhaustive and matches FlatIndex
        ivf_res = ivf.search(query, k=k, nprobe=2)

        assert len(ivf_res) == len(flat_res)
        for i in range(k):
            flat_id, flat_dist = flat_res[i]
            ivf_id, ivf_dist = ivf_res[i]
            assert ivf_id == flat_id, f"Rank {i} ID mismatch: IVF={ivf_id} vs Flat={flat_id}"
            assert math.isclose(ivf_dist, flat_dist, rel_tol=1e-5), (
                f"Rank {i} score mismatch: IVF={ivf_dist} vs Flat={flat_dist}"
            )


class TestIVFClusterDiagnostics:
    """Test category 6: cluster_sizes() and nprobe_coverage() diagnostics."""

    def test_cluster_sizes_sum_and_coverage(self):
        dim = 4
        num_clusters = 4
        n = 40

        import random

        rng = random.Random(123)
        data = [[rng.uniform(-5.0, 5.0) for _ in range(dim)] for _ in range(n)]
        ids = list(range(1, n + 1))

        index = vecta.IVFIndex(dim=dim, num_clusters=num_clusters, metric="euclidean")
        index.train(data, k=num_clusters, max_iterations=30, seed=99)
        index.add_batch(ids, data)

        sizes = index.cluster_sizes()
        assert len(sizes) == num_clusters
        assert sum(sizes) == n

        # Check coverage progression
        query = data[0]
        cov1 = index.nprobe_coverage(query, nprobe=1)
        cov2 = index.nprobe_coverage(query, nprobe=2)
        cov4 = index.nprobe_coverage(query, nprobe=4)

        assert cov1 <= cov2 <= cov4
        assert cov4 == n
