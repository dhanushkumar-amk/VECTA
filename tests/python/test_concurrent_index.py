import random
import threading
import time
import pytest
import vecta


class TestConcurrentFlatIndexBasic:
    """Test 1: Single-threaded sanity check matching FlatIndex behavior."""

    def test_single_threaded_basic_correctness(self):
        dim = 4
        metric = "euclidean"

        plain = vecta.FlatIndex(dim, metric)
        concurrent = vecta.ConcurrentFlatIndex(dim, metric)

        vectors = [
            (1, [1.0, 0.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0, 0.0]),
            (3, [0.0, 0.0, 1.0, 0.0]),
            (4, [0.5, 0.5, 0.0, 0.0]),
            (5, [0.1, 0.9, 0.0, 0.0]),
        ]

        for vid, vec in vectors:
            plain.add(vid, vec)
            concurrent.add(vid, vec)

        assert len(plain) == len(concurrent) == 5
        assert plain.dim() == concurrent.dim() == 4
        assert not concurrent.is_empty()

        query = [0.2, 0.8, 0.0, 0.0]
        plain_res = plain.search(query, k=3)
        conc_res = concurrent.search(query, k=3)

        assert len(plain_res) == len(conc_res) == 3
        for i in range(3):
            assert plain_res[i][0] == conc_res[i][0]
            assert abs(plain_res[i][1] - conc_res[i][1]) < 1e-6


class TestConcurrentFlatIndexGilRelease:
    """Test 2: Genuine GIL release verification comparing concurrent vs sequential wall-clock time."""

    def test_gil_release_concurrent_search_speedup(self):
        dim = 64
        num_vectors = 8000
        num_threads = 4
        queries_per_thread = 20
        total_queries = num_threads * queries_per_thread

        index = vecta.ConcurrentFlatIndex(dim, "euclidean")

        # Populate index with deterministic pseudorandom data
        rng = random.Random(42)
        ids = list(range(num_vectors))
        vectors = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(num_vectors)]
        index.add_batch(ids, vectors)
        assert len(index) == num_vectors

        # Generate deterministic test queries
        q_rng = random.Random(9999)
        queries = [[q_rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(total_queries)]

        # 1. Sequential search timing on main thread
        t_seq_start = time.perf_counter()
        for q in queries:
            index.search(q, k=10)
        seq_duration = time.perf_counter() - t_seq_start

        # 2. Concurrent search timing using Python threading.Thread
        threads = []
        results_by_thread = [[] for _ in range(num_threads)]

        def worker(thread_idx: int):
            start_idx = thread_idx * queries_per_thread
            end_idx = start_idx + queries_per_thread
            for i in range(start_idx, end_idx):
                res = index.search(queries[i], k=10)
                results_by_thread[thread_idx].append(res)

        for tid in range(num_threads):
            t = threading.Thread(target=worker, args=(tid,))
            threads.append(t)

        t_conc_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        conc_duration = time.perf_counter() - t_conc_start

        speedup = seq_duration / conc_duration

        print("\n" + "=" * 80)
        print(f"Phase 32 Test 2: Python GIL-Release Read Benchmark (N={num_vectors}, Dim={dim}, Queries={total_queries})")
        print("=" * 80)
        print(f"  Sequential (Single-Thread):  {seq_duration * 1000:.2f} ms")
        print(f"  Concurrent ({num_threads} Python Threads): {conc_duration * 1000:.2f} ms")
        print(f"  GIL-Release Speedup:         {speedup:.2f}x faster")
        print("=" * 80)

        # Confirm all threads completed all queries
        for res_list in results_by_thread:
            assert len(res_list) == queries_per_thread

        # Empirically verify speedup > 1.0x (proving GIL was released for parallel execution)
        assert speedup > 1.0, f"Expected speedup > 1.0x due to GIL release, got {speedup:.2f}x"


class TestConcurrentFlatIndexWrites:
    """Test 3, 4: Concurrent writes and mixed reads/writes from Python threads."""

    def test_concurrent_writes_from_python_threads(self):
        """Test 3: Multiple Python threads calling add() simultaneously with distinct IDs."""
        dim = 4
        num_threads = 8
        inserts_per_thread = 50
        total_inserts = num_threads * inserts_per_thread

        index = vecta.ConcurrentFlatIndex(dim, "euclidean")
        threads = []

        def worker(thread_idx: int):
            base_id = thread_idx * 1000
            for i in range(inserts_per_thread):
                vid = base_id + i
                vec = [float(thread_idx) * 10.0, float(i), 0.0, 0.0]
                index.add(vid, vec)

        for tid in range(num_threads):
            t = threading.Thread(target=worker, args=(tid,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Confirm total count matches sum of thread writes
        assert len(index) == total_inserts

        # Spot check that vectors are intact and searchable
        for tid in range(num_threads):
            target_id = tid * 1000
            query = [float(tid) * 10.0, 0.0, 0.0, 0.0]
            top1 = index.search(query, k=1)
            assert top1[0][0] == target_id
            assert abs(top1[0][1]) < 1e-6

    def test_mixed_concurrent_reads_and_writes(self):
        """Test 4: Mixed concurrent reads and writes from Python threads."""
        dim = 4
        index = vecta.ConcurrentFlatIndex(dim, "euclidean")

        # Initial seeding
        for i in range(1, 11):
            index.add(i, [float(i), 0.0, 0.0, 0.0])

        stop_event = threading.Event()
        read_counts = [0, 0, 0, 0]

        def reader_worker(reader_id: int):
            query = [1.0, 0.0, 0.0, 0.0]
            while not stop_event.is_set():
                res = index.search(query, k=3)
                assert len(res) > 0
                read_counts[reader_id] += 1

        num_writers = 4
        writes_per_writer = 40

        def writer_worker(writer_id: int):
            base_id = 100 + (writer_id * 1000)
            for i in range(writes_per_writer):
                vid = base_id + i
                vec = [float(writer_id) * 10.0 + 5.0, float(i), 1.0, 0.0]
                index.add(vid, vec)

        reader_threads = [
            threading.Thread(target=reader_worker, args=(i,)) for i in range(4)
        ]
        writer_threads = [
            threading.Thread(target=writer_worker, args=(i,)) for i in range(num_writers)
        ]

        # Start readers then writers
        for t in reader_threads:
            t.start()
        for t in writer_threads:
            t.start()

        # Wait for writers to complete
        for t in writer_threads:
            t.join()

        # Stop and join readers
        stop_event.set()
        for t in reader_threads:
            t.join()

        expected_total = 10 + (num_writers * writes_per_writer)
        assert len(index) == expected_total
        assert sum(read_counts) > 0


class TestConcurrentFlatIndexValidation:
    """Test 5: Error handling and dimension validation before allow_threads."""

    def test_dimension_validation_before_allow_threads(self):
        dim = 4
        index = vecta.ConcurrentFlatIndex(dim, "euclidean")

        # Dimension mismatch on add raises ValueError
        with pytest.raises(ValueError, match="vector dimension mismatch"):
            index.add(1, [1.0, 2.0])  # dim 2 instead of 4

        # Dimension mismatch on search raises ValueError
        with pytest.raises(ValueError, match="query dimension mismatch"):
            index.search([1.0, 2.0], k=5)

        # Dimension mismatch in add_batch
        with pytest.raises(ValueError, match="dimension"):
            index.add_batch([1], [[1.0, 2.0]])

        # Mismatched count in add_batch
        with pytest.raises(ValueError, match="ids count"):
            index.add_batch([1, 2], [[1.0, 0.0, 0.0, 0.0]])

        # Duplicate ID raises ValueError
        index.add(10, [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="duplicate id"):
            index.add(10, [2.0, 0.0, 0.0, 0.0])
