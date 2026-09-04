"""
Tests for ShardedFlatIndex Python bindings (Phase 34).

Covers:
1. Constructor validation (valid params succeed, invalid metric/dim/num_shards raise ValueError).
2. add() then search() round-trip correctness (fan-out and merge locate vectors correctly).
3. Cross-check: FlatIndex vs ShardedFlatIndex with identical data return IDENTICAL results.
4. Equivalence between search(parallel=False) and search(parallel=True).
5. shard_sizes() returns distribution summing to total length and reasonably balanced.
6. Dimension validation and error handling (wrong dimension, duplicate IDs, batch mismatches).
7. Thread-safe concurrent access across Python threads with GIL release.
"""

import math
import random
import threading
import pytest
import vecta


class TestShardedFlatIndexConstructor:
    """Test 1: Constructor validation."""

    def test_create_valid_parameters(self):
        index = vecta.ShardedFlatIndex(dim=128, num_shards=4, metric="euclidean")
        assert len(index) == 0
        assert index.is_empty()
        assert index.dim() == 128
        assert index.num_shards() == 4
        assert index.metric() == "euclidean"
        assert index.shard_sizes() == [0, 0, 0, 0]

    def test_create_valid_metrics(self):
        for metric in ["euclidean", "l2", "cosine", "cos", "dot_product", "dot", "ip"]:
            idx = vecta.ShardedFlatIndex(dim=8, num_shards=2, metric=metric)
            assert idx.dim() == 8
            assert idx.num_shards() == 2

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="unknown metric"):
            vecta.ShardedFlatIndex(dim=128, num_shards=4, metric="manhattan")

    def test_zero_dim_raises(self):
        with pytest.raises(ValueError, match="dimension must be greater than 0"):
            vecta.ShardedFlatIndex(dim=0, num_shards=4, metric="euclidean")

    def test_zero_shards_raises(self):
        with pytest.raises(ValueError, match="num_shards must be greater than 0"):
            vecta.ShardedFlatIndex(dim=128, num_shards=0, metric="euclidean")


class TestShardedFlatIndexRoundTrip:
    """Test 2: add() then search() round-trip correctness."""

    def test_add_and_search_round_trip(self):
        dim = 4
        num_shards = 4
        index = vecta.ShardedFlatIndex(dim=dim, num_shards=num_shards, metric="euclidean")

        vectors = [
            (10, [1.0, 0.0, 0.0, 0.0]),
            (20, [0.0, 1.0, 0.0, 0.0]),
            (30, [0.0, 0.0, 1.0, 0.0]),
            (40, [0.0, 0.0, 0.0, 1.0]),
            (50, [0.5, 0.5, 0.0, 0.0]),
            (60, [0.0, 0.5, 0.5, 0.0]),
        ]

        for vid, vec in vectors:
            index.add(vid, vec)

        assert len(index) == len(vectors)
        assert not index.is_empty()

        # Query each known vector, verify top result is the exact vector with score ~ 0.0
        for vid, vec in vectors:
            results = index.search(vec, k=1)
            assert len(results) == 1
            top_id, top_score = results[0]
            assert top_id == vid
            assert abs(top_score - 0.0) < 1e-5


class TestShardedVsFlatIndexCrossCheck:
    """Test 3: Correctness cross-check between plain FlatIndex and ShardedFlatIndex."""

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "dot_product"])
    def test_sharded_vs_flat_identical_results(self, metric):
        dim = 8
        num_shards = 4
        n_vectors = 150
        random.seed(42)

        flat = vecta.FlatIndex(dim=dim, metric=metric)
        sharded = vecta.ShardedFlatIndex(dim=dim, num_shards=num_shards, metric=metric)

        vectors = []
        ids = list(range(1, n_vectors + 1))
        for _ in range(n_vectors):
            vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            vectors.append(vec)

        flat.add_batch(ids, vectors)
        sharded.add_batch(ids, vectors)

        assert len(flat) == len(sharded) == n_vectors

        # Test multiple random query vectors
        for _ in range(5):
            query = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            for k in [1, 5, 10, 20]:
                flat_res = flat.search(query, k=k)
                sharded_res = sharded.search(query, k=k)

                assert len(flat_res) == len(sharded_res) == k

                for rank, ((f_id, f_score), (s_id, s_score)) in enumerate(
                    zip(flat_res, sharded_res)
                ):
                    assert (
                        f_id == s_id
                    ), f"ID mismatch at rank {rank} for metric {metric}, k={k}: flat={f_id}, sharded={s_id}"
                    assert (
                        abs(f_score - s_score) < 1e-5
                    ), f"Score mismatch at rank {rank} for metric {metric}, k={k}: flat={f_score}, sharded={s_score}"


class TestParallelSearchEquivalence:
    """Test 4: search(parallel=True) and search(parallel=False) return IDENTICAL results."""

    def test_parallel_matches_sequential(self):
        dim = 16
        num_shards = 4
        n_vectors = 300
        random.seed(1234)

        index = vecta.ShardedFlatIndex(dim=dim, num_shards=num_shards, metric="euclidean")

        ids = list(range(1, n_vectors + 1))
        vectors = [[random.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(n_vectors)]
        index.add_batch(ids, vectors)

        for _ in range(5):
            query = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            for k in [1, 10, 25]:
                seq_res = index.search(query, k=k, parallel=False)
                par_res = index.search(query, k=k, parallel=True)

                assert len(seq_res) == len(par_res) == k

                for rank, ((s_id, s_score), (p_id, p_score)) in enumerate(
                    zip(seq_res, par_res)
                ):
                    assert (
                        s_id == p_id
                    ), f"ID mismatch at rank {rank}: seq={s_id}, par={p_id}"
                    assert (
                        abs(s_score - p_score) < 1e-5
                    ), f"Score mismatch at rank {rank}: seq={s_score}, par={p_score}"


class TestShardSizesDistribution:
    """Test 5: shard_sizes() returns a list summing to the total inserted count, reasonably evenly distributed."""

    def test_shard_sizes_integrity_and_distribution(self):
        dim = 8
        num_shards = 8
        n_vectors = 800
        random.seed(777)

        index = vecta.ShardedFlatIndex(dim=dim, num_shards=num_shards, metric="euclidean")

        ids = list(range(1, n_vectors + 1))
        vectors = [[random.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(n_vectors)]
        index.add_batch(ids, vectors)

        assert len(index) == n_vectors
        sizes = index.shard_sizes()

        assert len(sizes) == num_shards
        assert sum(sizes) == n_vectors

        # Expected mean per shard = 100
        # Assert each shard receives between 50 and 175 (reasonably balanced)
        for s, count in enumerate(sizes):
            assert count > 0, f"Shard {s} was unexpectedly empty"
            assert (
                40 <= count <= 180
            ), f"Shard {s} has {count} items, expected reasonably close to 100"


class TestDimensionAndInputValidation:
    """Test 6: Dimension mismatches, duplicate IDs, and input errors raise ValueError without crashing."""

    def test_add_wrong_dimension_raises(self):
        index = vecta.ShardedFlatIndex(dim=4, num_shards=2, metric="euclidean")
        with pytest.raises(ValueError, match="vector dimension mismatch"):
            index.add(1, [1.0, 2.0])

    def test_add_duplicate_id_raises(self):
        index = vecta.ShardedFlatIndex(dim=4, num_shards=2, metric="euclidean")
        index.add(1, [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="duplicate id 1"):
            index.add(1, [0.0, 1.0, 0.0, 0.0])

    def test_add_batch_mismatched_lengths_raises(self):
        index = vecta.ShardedFlatIndex(dim=4, num_shards=2, metric="euclidean")
        with pytest.raises(ValueError, match="ids count"):
            index.add_batch([1, 2], [[1.0, 0.0, 0.0, 0.0]])

    def test_add_batch_wrong_dimension_raises(self):
        index = vecta.ShardedFlatIndex(dim=4, num_shards=2, metric="euclidean")
        with pytest.raises(ValueError, match="vector at index 0 has dimension"):
            index.add_batch([1], [[1.0, 0.0]])

    def test_search_wrong_dimension_raises(self):
        index = vecta.ShardedFlatIndex(dim=4, num_shards=2, metric="euclidean")
        index.add(1, [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="query dimension mismatch"):
            index.search([1.0, 2.0], k=1)


class TestConcurrentAccess:
    """Test 7: Threaded concurrent access with Python GIL release."""

    def test_concurrent_searches_and_adds(self):
        dim = 8
        num_shards = 4
        index = vecta.ShardedFlatIndex(dim=dim, num_shards=num_shards, metric="euclidean")

        # Pre-populate with some data
        initial_ids = list(range(1, 101))
        initial_vecs = [[float(i)] * dim for i in initial_ids]
        index.add_batch(initial_ids, initial_vecs)

        errors = []

        def reader_task():
            try:
                for _ in range(50):
                    q = [float(random.randint(1, 100))] * dim
                    res = index.search(q, k=5, parallel=True)
                    assert len(res) == 5
            except Exception as e:
                errors.append(e)

        def writer_task(thread_id):
            try:
                base = 1000 + thread_id * 100
                for i in range(25):
                    vid = base + i
                    vec = [float(vid)] * dim
                    index.add(vid, vec)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(4):
            threads.append(threading.Thread(target=reader_task))
        for t_id in range(2):
            threads.append(threading.Thread(target=writer_task, args=(t_id,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        assert len(index) == 100 + (2 * 25)
