//! Concurrent access and read/write locking for [`FlatIndex`].
//!
//! # Architecture & Concurrency Model
//!
//! By default, vecta index structures are single-threaded and assume sequential access.
//! In production vector databases, query workloads are heavily asymmetric: **searches (reads)
//! occur orders of magnitude more frequently than inserts (writes)**.
//!
//! [`ConcurrentFlatIndex`] wraps [`FlatIndex`] inside an `std::sync::RwLock`:
//! - **Multiple Concurrent Readers**: Arbitrary reader threads can acquire shared read locks
//!   simultaneously and execute [`ConcurrentFlatIndex::search`] in parallel without blocking each other.
//! - **Exclusive Writers**: Ingestion operations ([`ConcurrentFlatIndex::add`] and
//!   [`ConcurrentFlatIndex::add_batch`]) acquire an exclusive write lock. A writer will wait until
//!   all active read guards have completed, and new searches will wait until the write guard completes.
//!
//! # Usage Pattern
//! Wrap [`ConcurrentFlatIndex`] in an `Arc`:
//! ```rust
//! use std::sync::Arc;
//! use vecta::core::concurrent_index::ConcurrentFlatIndex;
//! use vecta::core::flat_index::Metric;
//!
//! let index = Arc::new(ConcurrentFlatIndex::new(128, Metric::Euclidean));
//!
//! // Clone the Arc for worker threads:
//! let reader_clone = Arc::clone(&index);
//! std::thread::spawn(move || {
//!     let results = reader_clone.search(&[0.0; 128], 10);
//! });
//!
//! let writer_clone = Arc::clone(&index);
//! std::thread::spawn(move || {
//!     writer_clone.add(1, &[1.0; 128]).unwrap();
//! });
//! ```
//!
//! # Tradeoffs & Operational Characteristics
//! - **Read Scalability**: Read throughput scales linearly with available CPU cores when uncontended by writes.
//! - **Writer Latency Under Contention**: Because writers require exclusive access, high read volume can
//!   cause write latency spikes as writers queue behind in-flight search queries.
//! - **Lock Poisoning Discipline**: If a writer thread panics while holding the exclusive lock, the
//!   [`std::sync::RwLock`] becomes poisoned. Rather than attempting to ignore the poison and operating on
//!   corrupted state, subsequent operations explicitly panic with actionable error messages advising
//!   recovery from a checkpoint or WAL replay.

use std::sync::RwLock;

use crate::core::batch::VectorBatch;
use crate::core::flat_index::{FlatIndex, Metric};
use crate::core::topk::ScoredId;

/// Thread-safe concurrent wrapper around [`FlatIndex`] using reader-writer synchronization.
pub struct ConcurrentFlatIndex {
    inner: RwLock<FlatIndex>,
}

impl ConcurrentFlatIndex {
    /// Create a new concurrent flat index for vectors of the specified dimensionality and metric.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            inner: RwLock::new(FlatIndex::new(dim, metric)),
        }
    }

    /// Construct a [`ConcurrentFlatIndex`] wrapping an already existing [`FlatIndex`].
    pub fn from_flat_index(index: FlatIndex) -> Self {
        Self {
            inner: RwLock::new(index),
        }
    }

    /// Search for the top-`k` nearest neighbors to `query` concurrently with other readers.
    ///
    /// Acquires a shared read lock (`self.inner.read()`). Multiple reader threads can execute
    /// `search` concurrently without mutual contention.
    ///
    /// # Lock Poisoning
    /// If a previous writer thread panicked while holding the exclusive lock, the lock is poisoned.
    /// In this scenario, this method panics with an explicit error explaining that the index state
    /// may be corrupted and must be restored from a checkpoint.
    ///
    /// # Panics
    /// - If the `RwLock` is poisoned.
    /// - If `query.len() != self.dim()`.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredId> {
        let guard = self.inner.read().unwrap_or_else(|poisoned| {
            panic!(
                "ConcurrentFlatIndex::search: RwLock is poisoned due to a previous writer panic. \
                The index may be in an inconsistent state and must be restored from a checkpoint: {}",
                poisoned
            );
        });

        guard.search(query, k)
    }

    /// Insert a single vector with its external ID under an exclusive write lock.
    ///
    /// NOTE: Takes `&self` rather than `&mut self` via interior mutability, allowing
    /// instances shared across threads via `Arc<ConcurrentFlatIndex>` to perform inserts safely.
    ///
    /// # Lock Poisoning & Validation
    /// Validates vector dimensionality and duplicate IDs before mutating the index to avoid
    /// panics while holding the lock (which would poison the lock for all other threads).
    ///
    /// # Errors
    /// Returns `Err(String)` if `vector.len() != self.dim()` or if `id` already exists.
    pub fn add(&self, id: u64, vector: &[f32]) -> Result<(), String> {
        let mut guard = self.inner.write().unwrap_or_else(|poisoned| {
            panic!(
                "ConcurrentFlatIndex::add: RwLock is poisoned due to a previous writer panic. \
                The index may be in an inconsistent state and must be restored from a checkpoint: {}",
                poisoned
            );
        });

        if vector.len() != guard.dim() {
            return Err(format!(
                "ConcurrentFlatIndex::add: vector dimension mismatch: expected {}, got {}",
                guard.dim(),
                vector.len()
            ));
        }

        if guard.ids.contains(&id) {
            return Err(format!(
                "ConcurrentFlatIndex::add: duplicate id {}: already exists in index",
                id
            ));
        }

        guard.add(id, vector);
        Ok(())
    }

    /// Bulk-insert vectors with their external IDs under an exclusive write lock.
    ///
    /// # Errors
    /// Returns `Err(String)` if IDs count does not match vector count, if dimensionality
    /// differs from the index, or if any ID already exists in the index or incoming batch.
    pub fn add_batch(&self, ids: &[u64], vectors: &VectorBatch) -> Result<(), String> {
        let mut guard = self.inner.write().unwrap_or_else(|poisoned| {
            panic!(
                "ConcurrentFlatIndex::add_batch: RwLock is poisoned due to a previous writer panic. \
                The index may be in an inconsistent state and must be restored from a checkpoint: {}",
                poisoned
            );
        });

        if ids.len() != vectors.len() {
            return Err(format!(
                "ConcurrentFlatIndex::add_batch: ids count ({}) != vectors count ({})",
                ids.len(),
                vectors.len()
            ));
        }

        if vectors.dim != guard.dim() {
            return Err(format!(
                "ConcurrentFlatIndex::add_batch: vectors dimension ({}) != index dimension ({})",
                vectors.dim,
                guard.dim()
            ));
        }

        // Check for duplicates within incoming IDs
        let mut unique_check = std::collections::HashSet::with_capacity(ids.len());
        for &id in ids {
            if !unique_check.insert(id) {
                return Err(format!(
                    "ConcurrentFlatIndex::add_batch: duplicate id {} within incoming batch",
                    id
                ));
            }
            if guard.ids.contains(&id) {
                return Err(format!(
                    "ConcurrentFlatIndex::add_batch: duplicate id {} already in index",
                    id
                ));
            }
        }

        guard.add_batch(ids, vectors);
        Ok(())
    }

    /// Return the number of vectors stored in the index.
    pub fn len(&self) -> usize {
        let guard = self.inner.read().unwrap_or_else(|poisoned| {
            panic!("ConcurrentFlatIndex::len: RwLock is poisoned: {}", poisoned);
        });
        guard.len()
    }

    /// Return true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the vector dimensionality of the index.
    pub fn dim(&self) -> usize {
        let guard = self.inner.read().unwrap_or_else(|poisoned| {
            panic!("ConcurrentFlatIndex::dim: RwLock is poisoned: {}", poisoned);
        });
        guard.dim()
    }

    /// Return the distance metric used by the index.
    pub fn metric(&self) -> Metric {
        let guard = self.inner.read().unwrap_or_else(|poisoned| {
            panic!(
                "ConcurrentFlatIndex::metric: RwLock is poisoned: {}",
                poisoned
            );
        });
        guard.metric
    }
}

// ============================================================================
// Concurrency Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    /// Test 1: Basic correctness: single-threaded add() then search() through ConcurrentFlatIndex
    /// produces the SAME results as equivalent direct FlatIndex calls.
    #[test]
    fn test_concurrent_single_threaded_equivalence() {
        let dim = 4;
        let metric = Metric::Euclidean;

        let mut plain_index = FlatIndex::new(dim, metric);
        let concurrent_index = ConcurrentFlatIndex::new(dim, metric);

        let vectors: Vec<(u64, [f32; 4])> = vec![
            (1, [1.0, 0.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0, 0.0]),
            (3, [0.0, 0.0, 1.0, 0.0]),
            (4, [0.5, 0.5, 0.0, 0.0]),
            (5, [0.1, 0.9, 0.0, 0.0]),
        ];

        for (id, vec) in &vectors {
            plain_index.add(*id, vec);
            concurrent_index.add(*id, vec).unwrap();
        }

        assert_eq!(plain_index.len(), concurrent_index.len());
        assert_eq!(plain_index.dim(), concurrent_index.dim());
        assert_eq!(plain_index.metric, concurrent_index.metric());

        let query = [0.2, 0.8, 0.0, 0.0];
        let plain_res = plain_index.search(&query, 3);
        let concurrent_res = concurrent_index.search(&query, 3);

        assert_eq!(plain_res.len(), concurrent_res.len());
        for i in 0..plain_res.len() {
            assert_eq!(plain_res[i].id, concurrent_res[i].id);
            assert_eq!(plain_res[i].score, concurrent_res[i].score);
        }
    }

    /// Test 2: Concurrent READS test: spawn multiple threads (8 threads), all calling search()
    /// simultaneously on the SAME pre-populated ConcurrentFlatIndex, confirm ALL threads get
    /// correct, consistent results with no crashes/panics/data races.
    #[test]
    fn test_concurrent_reads() {
        let dim = 8;
        let metric = Metric::Cosine;
        let num_vectors = 500;
        let num_threads = 8;
        let searches_per_thread = 50;

        let concurrent_index = Arc::new(ConcurrentFlatIndex::new(dim, metric));

        // Populate index
        let mut rng = StdRng::seed_from_u64(42);
        for id in 0..num_vectors as u64 {
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                vec.push(rng.gen_range(-1.0..1.0));
            }
            concurrent_index.add(id, &vec).unwrap();
        }

        // Generate baseline expected results for 5 deterministic queries
        let test_queries: Vec<Vec<f32>> = (0..5)
            .map(|seed| {
                let mut q_rng = StdRng::seed_from_u64(seed + 1000);
                (0..dim).map(|_| q_rng.gen_range(-1.0..1.0)).collect()
            })
            .collect();

        let expected_results: Vec<Vec<ScoredId>> = test_queries
            .iter()
            .map(|q| concurrent_index.search(q, 5))
            .collect();

        // Spawn reader threads
        let mut handles = Vec::new();
        for thread_idx in 0..num_threads {
            let index_clone = Arc::clone(&concurrent_index);
            let queries_clone = test_queries.clone();
            let expected_clone = expected_results.clone();

            let handle = thread::spawn(move || {
                for iter in 0..searches_per_thread {
                    let q_idx = (thread_idx + iter) % queries_clone.len();
                    let res = index_clone.search(&queries_clone[q_idx], 5);
                    let exp = &expected_clone[q_idx];
                    assert_eq!(res.len(), exp.len());
                    for i in 0..res.len() {
                        assert_eq!(res[i].id, exp[i].id);
                        assert_eq!(res[i].score, exp[i].score);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Reader thread panicked");
        }

        assert_eq!(concurrent_index.len(), num_vectors);
    }

    /// Test 3: Concurrent WRITES test: spawn multiple threads, each calling add() with DIFFERENT,
    /// non-conflicting ids simultaneously, confirm after joining all threads that len() equals
    /// the total number of inserts attempted, and every id is retrievable/searchable correctly.
    #[test]
    fn test_concurrent_writes() {
        let dim = 4;
        let metric = Metric::Euclidean;
        let num_threads = 8;
        let vectors_per_thread = 50;
        let total_expected = num_threads * vectors_per_thread;

        let concurrent_index = Arc::new(ConcurrentFlatIndex::new(dim, metric));

        let mut handles = Vec::new();
        for thread_idx in 0..num_threads {
            let index_clone = Arc::clone(&concurrent_index);
            let handle = thread::spawn(move || {
                let base_id = (thread_idx * 1000) as u64;
                for i in 0..vectors_per_thread as u64 {
                    let id = base_id + i;
                    let vec = [(thread_idx as f32) * 10.0, i as f32, 0.0, 0.0];
                    index_clone.add(id, &vec).unwrap();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Writer thread panicked");
        }

        // 1. Total length must match sum of all threads' inserts
        assert_eq!(
            concurrent_index.len(),
            total_expected,
            "Expected {} total vectors, found {}",
            total_expected,
            concurrent_index.len()
        );

        // 2. Spot check that each thread's vectors are searchable
        for thread_idx in 0..num_threads {
            let target_id = (thread_idx * 1000) as u64;
            let query = [(thread_idx as f32) * 10.0, 0.0, 0.0, 0.0];
            let top1 = concurrent_index.search(&query, 1);
            assert_eq!(
                top1[0].id, target_id,
                "Failed to find exact vector for thread {}",
                thread_idx
            );
            assert_eq!(top1[0].score, 0.0);
        }
    }

    /// Test 4: Mixed concurrent reads+writes test: spawn some threads doing search() and others
    /// doing add() simultaneously against the same index, run for a bounded duration/iteration
    /// count, confirm no panics/crashes occur and the final state is consistent.
    #[test]
    fn test_mixed_concurrent_reads_and_writes() {
        let dim = 4;
        let metric = Metric::Euclidean;
        let num_writers = 4;
        let num_readers = 4;
        let writes_per_writer = 50;

        let concurrent_index = Arc::new(ConcurrentFlatIndex::new(dim, metric));

        // Initial seeding so readers have vectors to search right away
        for id in 1..=10u64 {
            concurrent_index
                .add(id, &[id as f32, 0.0, 0.0, 0.0])
                .unwrap();
        }

        let stop_flag = Arc::new(AtomicBool::new(false));

        // 1. Spawn readers
        let mut reader_handles = Vec::new();
        for _ in 0..num_readers {
            let index_clone = Arc::clone(&concurrent_index);
            let stop_clone = Arc::clone(&stop_flag);
            let handle = thread::spawn(move || {
                let query = [1.0f32, 0.0, 0.0, 0.0];
                let mut read_count = 0;
                while !stop_clone.load(Ordering::Relaxed) {
                    let results = index_clone.search(&query, 3);
                    assert!(!results.is_empty());
                    read_count += 1;
                }
                read_count
            });
            reader_handles.push(handle);
        }

        // 2. Spawn writers
        let mut writer_handles = Vec::new();
        for writer_idx in 0..num_writers {
            let index_clone = Arc::clone(&concurrent_index);
            let handle = thread::spawn(move || {
                let base_id = 100 + (writer_idx * 1000) as u64;
                for i in 0..writes_per_writer as u64 {
                    let id = base_id + i;
                    let vec = [(writer_idx as f32) * 10.0 + 5.0, i as f32, 1.0, 0.0];
                    index_clone.add(id, &vec).unwrap();
                }
            });
            writer_handles.push(handle);
        }

        // Wait for all writers to finish
        for handle in writer_handles {
            handle.join().expect("Writer thread panicked");
        }

        // Signal readers to stop and join them
        stop_flag.store(true, Ordering::Relaxed);
        let mut total_reads = 0;
        for handle in reader_handles {
            total_reads += handle.join().expect("Reader thread panicked");
        }

        println!(
            "\nPhase 31 Test 4: Mixed Concurrency: Completed {} writes and {} reads concurrently with zero errors.",
            num_writers * writes_per_writer,
            total_reads
        );

        let expected_total = 10 + (num_writers * writes_per_writer);
        assert_eq!(concurrent_index.len(), expected_total);
    }

    /// Test 5: A basic throughput/contention sanity check: time how long N concurrent search-only
    /// threads take vs running the same N searches sequentially on a single thread — confirm
    /// concurrent reads show a meaningful speedup (proving RwLock genuinely allows concurrent reads,
    /// rather than accidentally serializing everything) — print both timings.
    #[test]
    fn test_read_concurrency_speedup_vs_sequential() {
        let dim = 64;
        let metric = Metric::Euclidean;
        let num_vectors = 8000;
        let num_threads = 4;
        let queries_per_thread = 20;
        let total_queries = num_threads * queries_per_thread;

        let concurrent_index = Arc::new(ConcurrentFlatIndex::new(dim, metric));

        // Populate index with random vectors
        let mut rng = StdRng::seed_from_u64(9999);
        let mut ids = Vec::with_capacity(num_vectors);
        let mut batch = VectorBatch::new(dim);

        for id in 0..num_vectors as u64 {
            ids.push(id);
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-1.0..1.0));
            }
            batch.push(&v);
        }
        concurrent_index.add_batch(&ids, &batch).unwrap();

        // Generate deterministic queries
        let queries: Vec<Vec<f32>> = (0..total_queries)
            .map(|i| {
                let mut q_rng = StdRng::seed_from_u64((i as u64) + 12345);
                (0..dim).map(|_| q_rng.gen_range(-1.0..1.0)).collect()
            })
            .collect();

        // 1. Sequential search timing on a single thread
        let t_seq_start = Instant::now();
        for q in &queries {
            let _ = concurrent_index.search(q, 10);
        }
        let seq_duration = t_seq_start.elapsed();

        // 2. Concurrent search timing across `num_threads` worker threads
        let queries_arc = Arc::new(queries);
        let t_par_start = Instant::now();
        let mut handles = Vec::new();

        for thread_idx in 0..num_threads {
            let index_clone = Arc::clone(&concurrent_index);
            let queries_clone = Arc::clone(&queries_arc);

            let handle = thread::spawn(move || {
                let start_idx = thread_idx * queries_per_thread;
                let end_idx = start_idx + queries_per_thread;
                for idx in start_idx..end_idx {
                    let _ = index_clone.search(&queries_clone[idx], 10);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Concurrent reader thread panicked");
        }
        let par_duration = t_par_start.elapsed();

        let speedup = seq_duration.as_secs_f64() / par_duration.as_secs_f64();

        println!(
            "\n================================================================================"
        );
        println!(
            "Phase 31 Test 5: Concurrent vs Sequential Read Throughput (N={}, Dim={}, Queries={})",
            num_vectors, dim, total_queries
        );
        println!(
            "================================================================================"
        );
        println!("  Sequential 1-thread duration:  {:.2?}", seq_duration);
        println!(
            "  Concurrent {}-threads duration: {:.2?}",
            num_threads, par_duration
        );
        println!("  Parallel Speedup:             {:.2}x faster", speedup);
        println!(
            "================================================================================"
        );

        // Sanity check that concurrent reads achieved true parallelism (speedup > 1.0)
        assert!(
            speedup > 1.0,
            "Expected concurrent read speedup > 1.0x, got {:.2}x (seq: {:.2?}, par: {:.2?})",
            speedup,
            seq_duration,
            par_duration
        );
    }
}
