"""
Smoke tests for the vecta Python extension module.

This file grows with each phase — add new assertions here as new
Python-exposed functions and classes are added in src/python.rs.
The CI workflow (ci.yml) runs this via pytest on every push/PR.
"""

import vecta


class TestPhase1:
    """Phase 1: basic module import and placeholder function."""

    def test_import(self):
        """Module should be importable without errors."""
        assert vecta is not None

    def test_hello_vecta(self):
        """hello_vecta() returns the expected initialization string."""
        result = vecta.hello_vecta()
        assert result == "vecta engine initialized"

    def test_hello_vecta_type(self):
        """hello_vecta() returns a Python str."""
        assert isinstance(vecta.hello_vecta(), str)


class TestPhase7:
    """Phase 7: FlatIndex Python bindings smoke tests."""

    def test_flat_index_smoke(self):
        """Basic creation, add, len, and search smoke test."""
        index = vecta.FlatIndex(dim=2, metric="euclidean")
        assert len(index) == 0
        assert index.is_empty()

        index.add(1, [1.0, 2.0])
        index.add(2, [3.0, 4.0])
        assert len(index) == 2
        assert not index.is_empty()

        results = index.search([1.0, 2.0], k=2)
        assert len(results) == 2
        # Nearest vector to [1, 2] is id 1 with distance 0.0
        assert results[0][0] == 1
        assert abs(results[0][1] - 0.0) < 1e-5


class TestPhase14:
    """Phase 14: IVFIndex Python bindings smoke tests."""

    def test_ivf_index_smoke(self):
        """Basic creation, train, add, len, cluster_sizes, and search smoke test."""
        index = vecta.IVFIndex(dim=2, num_clusters=2, metric="euclidean")
        assert len(index) == 0
        assert index.is_empty()
        assert not index.is_trained()

        # Train with 4 points
        train_data = [
            [1.0, 1.0],
            [1.0, 2.0],
            [9.0, 9.0],
            [9.0, 8.0],
        ]
        index.train(train_data, k=2, max_iterations=20, seed=42)
        assert index.is_trained()

        # Add vectors
        index.add(1, [1.0, 1.0])
        index.add(2, [9.0, 9.0])
        assert len(index) == 2
        assert not index.is_empty()

        sizes = index.cluster_sizes()
        assert len(sizes) == 2
        assert sum(sizes) == 2

        # Search with nprobe=2
        results = index.search([1.0, 1.0], k=2, nprobe=2)
        assert len(results) == 2
        assert results[0][0] == 1
        assert abs(results[0][1] - 0.0) < 1e-5


class TestPhase19:
    """Phase 19: HnswIndex Python bindings smoke tests."""

    def test_hnsw_index_smoke(self):
        """Basic creation, add, add_batch, len, max_layer_distribution, and search smoke test."""
        index = vecta.HnswIndex(dim=2, metric="euclidean")
        assert len(index) == 0
        assert index.is_empty()

        index.add(1, [1.0, 1.0])
        index.add_batch([2, 3], [[1.0, 2.0], [9.0, 9.0]])
        assert len(index) == 3
        assert not index.is_empty()

        dist = index.max_layer_distribution()
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 3

        results = index.search([1.0, 1.0], k=2)
        assert len(results) == 2
        assert results[0][0] == 1
        assert abs(results[0][1] - 0.0) < 1e-5


class TestPhase24:
    """Phase 24: IVFPQIndex Python bindings smoke tests."""

    def test_ivf_pq_index_smoke(self):
        """Basic creation, train, add, len, memory_footprint, and search smoke test."""
        index = vecta.IVFPQIndex(dim=4, num_clusters=2, m=2, k_per_subvector=2)
        assert len(index) == 0
        assert index.is_empty()
        assert not index.is_trained()

        # Train
        train_data = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
            [9.0, 9.0, 9.0, 9.0],
            [9.0, 8.0, 9.0, 8.0],
        ]
        index.train(train_data, ivf_seed=42, pq_seed=42)
        assert index.is_trained()

        # Add vectors
        index.add(1, [1.0, 1.0, 1.0, 1.0])
        index.add(2, [9.0, 9.0, 9.0, 9.0])
        assert len(index) == 2
        assert not index.is_empty()

        # Check footprint
        assert index.memory_footprint_bytes() > 0

        # Search
        results = index.search([1.0, 1.0, 1.0, 1.0], k=2, nprobe=2)
        assert len(results) == 2
        assert results[0][0] == 1



