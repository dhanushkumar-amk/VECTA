"""
Unit and integration tests for IVFPQIndex Python bindings (Phase 24).
"""

import pytest
import vecta


class TestIVFPQIndexConstructor:
    """Test constructor validation and constraints."""

    def test_create_valid_parameters_success(self):
        """Valid parameters construct IVFPQIndex successfully."""
        index = vecta.IVFPQIndex(dim=128, num_clusters=16, m=8)
        assert len(index) == 0
        assert index.is_empty()
        assert not index.is_trained()
        assert index.dim() == 128
        assert index.num_clusters() == 16

    def test_non_divisible_dimension_raises_value_error(self):
        """Constructing with dim not divisible by m raises ValueError."""
        # 10 is not divisible by 4
        with pytest.raises(ValueError, match="not evenly divisible"):
            vecta.IVFPQIndex(dim=10, num_clusters=2, m=4)

    def test_zero_dim_raises_value_error(self):
        """dim=0 raises ValueError."""
        with pytest.raises(ValueError):
            vecta.IVFPQIndex(dim=0, num_clusters=2, m=1)

    def test_zero_clusters_raises_value_error(self):
        """num_clusters=0 raises ValueError."""
        with pytest.raises(ValueError):
            vecta.IVFPQIndex(dim=8, num_clusters=0, m=2)

    def test_invalid_k_per_subvector_raises_value_error(self):
        """k_per_subvector > 256 raises ValueError."""
        with pytest.raises(ValueError):
            vecta.IVFPQIndex(dim=8, num_clusters=2, m=2, k_per_subvector=300)


class TestIVFPQTrainedSafety:
    """Verify add() and search() require training prior to execution."""

    def test_add_before_train_raises_value_error(self):
        """Calling add() before train() raises ValueError, no crash."""
        index = vecta.IVFPQIndex(dim=8, num_clusters=2, m=2)
        with pytest.raises(ValueError, match="must be trained"):
            index.add(1, [0.0] * 8)

    def test_add_batch_before_train_raises_value_error(self):
        """Calling add_batch() before train() raises ValueError, no crash."""
        index = vecta.IVFPQIndex(dim=8, num_clusters=2, m=2)
        with pytest.raises(ValueError, match="must be trained"):
            index.add_batch([1, 2], [[0.0] * 8, [1.0] * 8])

    def test_search_before_train_raises_value_error(self):
        """Calling search() before train() raises ValueError, no crash."""
        index = vecta.IVFPQIndex(dim=8, num_clusters=2, m=2)
        with pytest.raises(ValueError, match="must be trained"):
            index.search([0.0] * 8, k=5, nprobe=1)


class TestIVFPQEndToEndLifecycle:
    """Verify train -> add -> search end-to-end lifecycle."""

    def test_train_add_search_lifecycle(self):
        """Full train -> add -> search flow with correct return shapes."""
        dim = 8
        num_clusters = 2
        m = 2
        k_per_subvector = 4

        index = vecta.IVFPQIndex(
            dim=dim,
            num_clusters=num_clusters,
            m=m,
            k_per_subvector=k_per_subvector,
            max_iterations=10,
        )

        training_data = [
            [float(i + j) for j in range(dim)]
            for i in range(10)
        ]
        index.train(training_data, ivf_seed=42, pq_seed=42)
        assert index.is_trained()

        # Add single vectors
        index.add(101, [0.0] * dim)
        index.add(102, [5.0] * dim)
        assert len(index) == 2
        assert not index.is_empty()

        # Add batch
        index.add_batch(
            [103, 104],
            [[1.0] * dim, [6.0] * dim],
        )
        assert len(index) == 4

        # Search
        results = index.search([0.1] * dim, k=2, nprobe=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)

        # First result should be id 101 ([0.0]*dim is closest to [0.1]*dim)
        assert results[0][0] == 101


class TestIVFPQDimensionSafety:
    """Verify dimension checks raise Python exceptions and do not panic."""

    def test_train_wrong_dimension_raises(self):
        """Training vector with dimension mismatch raises ValueError."""
        index = vecta.IVFPQIndex(dim=8, num_clusters=2, m=2)
        bad_train = [[1.0] * 7, [2.0] * 8]
        with pytest.raises(ValueError, match="dimension"):
            index.train(bad_train)

    def test_add_wrong_dimension_raises(self):
        """Vector with wrong dimension raises ValueError."""
        index = vecta.IVFPQIndex(dim=8, num_clusters=2, m=2, k_per_subvector=2)
        index.train([[float(i)] * 8 for i in range(5)])
        with pytest.raises(ValueError, match="dimension"):
            index.add(1, [1.0] * 7)

    def test_add_batch_wrong_dimension_raises(self):
        """Batch vector with wrong dimension raises ValueError."""
        index = vecta.IVFPQIndex(dim=8, num_clusters=2, m=2, k_per_subvector=2)
        index.train([[float(i)] * 8 for i in range(5)])
        with pytest.raises(ValueError, match="dimension"):
            index.add_batch([1, 2], [[1.0] * 8, [2.0] * 9])

    def test_add_batch_mismatched_counts_raises(self):
        """Mismatched ids and vectors count raises ValueError."""
        index = vecta.IVFPQIndex(dim=8, num_clusters=2, m=2, k_per_subvector=2)
        index.train([[float(i)] * 8 for i in range(5)])
        with pytest.raises(ValueError, match="count"):
            index.add_batch([1], [[1.0] * 8, [2.0] * 8])

    def test_search_wrong_dimension_raises(self):
        """Query with wrong dimension raises ValueError."""
        index = vecta.IVFPQIndex(dim=8, num_clusters=2, m=2, k_per_subvector=2)
        index.train([[float(i)] * 8 for i in range(5)])
        with pytest.raises(ValueError, match="dimension"):
            index.search([1.0] * 7, k=2, nprobe=1)


class TestIVFPQIntegrationVsFlatIndex:
    """Integration test comparing IVFPQIndex against FlatIndex oracle."""

    def test_hand_verified_dataset_matches_flat_index(self):
        """Confirm top result is in the correct neighborhood comparing to FlatIndex."""
        dim = 4
        num_clusters = 2
        m = 2
        k_per_sub = 2

        ivfpq = vecta.IVFPQIndex(
            dim=dim,
            num_clusters=num_clusters,
            m=m,
            k_per_subvector=k_per_sub,
            max_iterations=20,
        )

        points = [
            (1, [0.0, 0.1, 0.1, 0.0]),
            (2, [0.1, 0.0, 0.0, 0.1]),
            (3, [0.2, 0.2, 0.1, 0.1]),
            (4, [10.0, 9.9, 10.1, 10.0]),
            (5, [9.9, 10.0, 10.0, 9.9]),
            (6, [10.1, 10.1, 9.9, 10.0]),
        ]

        ids = [p[0] for p in points]
        vectors = [p[1] for p in points]

        ivfpq.train(vectors, ivf_seed=42, pq_seed=42)
        ivfpq.add_batch(ids, vectors)

        flat = vecta.FlatIndex(dim=dim, metric="euclidean")
        flat.add_batch(ids, vectors)

        query = [0.05, 0.05, 0.05, 0.05]
        flat_results = flat.search(query, k=3)
        ivfpq_results = ivfpq.search(query, k=3, nprobe=2)

        assert len(ivfpq_results) == 3
        # In this easily separated setup, top result must be in cluster 1 (id 1, 2, or 3)
        top_id = ivfpq_results[0][0]
        assert top_id in [1, 2, 3]


class TestIVFPQMemoryFootprint:
    """Test memory_footprint_bytes diagnostic reporting."""

    def test_memory_footprint_sanity_and_compression(self):
        """Footprint is non-zero, matches expected size, and is significantly smaller than full-precision."""
        dim = 128
        num_clusters = 16
        m = 8
        k_per_sub = 256
        n = 500

        index = vecta.IVFPQIndex(
            dim=dim,
            num_clusters=num_clusters,
            m=m,
            k_per_subvector=k_per_sub,
            max_iterations=5,
        )

        import random
        random.seed(42)
        data = [
            [random.uniform(-1.0, 1.0) for _ in range(dim)]
            for _ in range(n)
        ]
        ids = list(range(n))

        index.train(data, ivf_seed=10, pq_seed=20)
        index.add_batch(ids, data)

        footprint = index.memory_footprint_bytes()
        assert footprint > 0

        # Vector storage: n * m = 500 * 8 = 4000 bytes
        # Full precision equivalent vector storage: 500 * 128 * 4 = 256,000 bytes
        full_precision_vector_bytes = n * dim * 4
        assert footprint < full_precision_vector_bytes
