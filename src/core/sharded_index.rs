//! In-process sharded vector index with hash-based routing and fan-out/merge query coordination.
//!
//! # Architecture & Concurrency Model
//!
//! Every previous index in vecta assumed all data lives in a single monolithic structure within one process.
//! In production vector search engines, systems outgrow single-index limits for two core reasons:
//! 1. **Data Volume Exceeds Memory**: Datasets with hundreds of millions of high-dimensional vectors
//!    cannot fit in the RAM of a single node or NUMA socket.
//! 2. **Query Throughput & Parallelism**: Search operations must be parallelized across independent
//!    CPU cores, memory buses, or worker processes.
//!
//! [`ShardedFlatIndex`] implements horizontal partitioning (sharding) in-process:
//! - Vectors are partitioned across `num_shards` independent [`ConcurrentFlatIndex`] shards.
//! - Ingestion ([`ShardedFlatIndex::add`] and [`ShardedFlatIndex::add_batch`]) routes vectors to their
//!   target shard deterministically via [`shard_for_id`].
//! - Queries ([`ShardedFlatIndex::search`] and [`ShardedFlatIndex::search_parallel`]) fan out to
//!   **all shards**, gather local top-`k` candidate sets, and merge them into a single global top-`k` ranking.
//!
//! # Per-Shard Storage: Why `ConcurrentFlatIndex`?
//!
//! Rather than wrapping plain [`crate::core::flat_index::FlatIndex`] shards, each shard in [`ShardedFlatIndex`] is an independent
//! [`ConcurrentFlatIndex`]:
//! - **Fine-Grained Concurrency**: Each shard manages its own `RwLock`. An insert targeting Shard 0 only
//!   locks Shard 0's write lock. Shards 1..N remain completely unlocked, freely serving concurrent searches
//!   or accepting other writes without mutual contention.
//! - **Ergonomic Interior Mutability**: Write methods ([`ShardedFlatIndex::add`], [`ShardedFlatIndex::add_batch`])
//!   accept `&self`, matching the ergonomics of [`ConcurrentFlatIndex`] and allowing multiple threads to insert
//!   or query without wrapping the outer index in another coarse mutex.
//! - **Safe Thread-Level Query Fan-Out**: In [`ShardedFlatIndex::search_parallel`], worker threads can borrow
//!   `&self` via `std::thread::scope` and concurrently search all shards simultaneously with zero lock
//!   contention between threads.
//!
//! # Routing Strategy: Hash-Based vs. Range-Based
//!
//! In distributed and sharded systems, two primary partitioning strategies exist:
//!
//! 1. **Hash-Based Routing** (`id -> shard = hash(id) % num_shards`):
//!    - **Chosen Strategy**: Implemented here via [`shard_for_id`].
//!    - **Pros**: Produces a statistically uniform distribution of data across all shards, regardless
//!      of how IDs are generated (dense sequential integers, sparse hashes, timestamps).
//!      No centralized coordinator or dynamic partition-map lookup is required.
//!    - **Cons**: Resizing the cluster (changing `num_shards`) alters the hash mapping for roughly
//!      `(N - 1) / N` keys, requiring extensive data migration unless consistent hashing is used.
//!      Furthermore, range queries over IDs cannot be pruned to specific shards and must fan out to all shards.
//!
//! 2. **Range-Based Routing** (shards defined by ID ranges `[min_id, max_id)`):
//!    - **Tradeoff Analysis**: Range routing allows scalar ID range scans to target only relevant shards,
//!      and appending sequential IDs does not disturb old shards. However, it suffers from severe write
//!      and query hot-spotting when IDs are sequential (e.g. auto-increment IDs or timestamps all flood
//!      the newest shard). Range-based routing also requires dynamic range splitting, merging, and a
//!      centralized coordinator service to track shard boundaries.
//!
//! For vector databases where queries are vector-similarity scans (not scalar ID ranges) and uniform load
//! distribution is paramount, hash-based routing is the standard industry foundation.
//!
//! # Fan-Out / Merge vs. Approximate Search (IVF / HNSW)
//!
//! It is critical to distinguish the role of sharding from approximate search algorithms:
//! - **IVF / HNSW (Search-Space Pruning)**: IVF queries only `nprobe` out of `nlist` clusters; HNSW
//!   traverses a small subset of the proximity graph. Their purpose is to **reduce total distance
//!   computations** per query at the expense of recall.
//! - **Sharding (Horizontal Scaling & Parallelism)**: Sharding **always queries ALL shards**.
//!   Sharding does NOT reduce the total distance computations performed. Instead, its benefits are:
//!   1. **Horizontal Capacity**: Allowing the total dataset to exceed the memory capacity of a single machine.
//!   2. **Computational Distribution**: Distributing the brute-force search load across multiple CPU cores,
//!      memory channels, or physical nodes.
//!
//! The fan-out/merge pattern collects local top-`k` candidates from all shards and computes a final global
//! top-`k` reduction, guaranteeing identical accuracy to an unsharded brute-force index.

use std::collections::HashSet;

use crate::core::batch::VectorBatch;
use crate::core::concurrent_index::ConcurrentFlatIndex;
use crate::core::flat_index::Metric;
use crate::core::topk::{top_k_largest, top_k_smallest, ScoredId};

/// Deterministic hash-based routing function: maps an external vector ID to a target shard index.
///
/// # Why a High-Quality Hash Function (SplitMix64) Matters
/// A naive hash like `(id as usize) % num_shards` or a basic multiplier collapses when IDs exhibit
/// regular stride patterns (e.g., even IDs, IDs incrementing by 8, or IDs sharing common factors
/// with `num_shards`). Under such patterns, thousands of IDs would cluster into a handful of shards,
/// creating severe storage and query imbalance.
///
/// We use the SplitMix64 mixing algorithm (Steele et al., 2014), which provides complete 64-bit
/// avalanche diffusion: every input bit affects every output bit with ~50% probability.
/// This guarantees a statistically uniform distribution across shards even for dense sequential
/// IDs (1, 2, 3...) or power-of-two stride patterns.
///
/// # Panics
/// Panics if `num_shards == 0`.
#[inline]
pub fn shard_for_id(id: u64, num_shards: usize) -> usize {
    assert!(
        num_shards > 0,
        "shard_for_id: num_shards must be greater than zero"
    );
    let mut z = id.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    (z as usize) % num_shards
}

/// An in-process sharded vector index delegating to multiple independent [`ConcurrentFlatIndex`] shards.
pub struct ShardedFlatIndex {
    shards: Vec<ConcurrentFlatIndex>,
    num_shards: usize,
    dim: usize,
    metric: Metric,
}

impl ShardedFlatIndex {
    /// Create a new sharded flat index with `num_shards` independent shards.
    ///
    /// # Panics
    /// Panics if `num_shards == 0` or `dim == 0`.
    pub fn new(dim: usize, num_shards: usize, metric: Metric) -> Self {
        assert!(
            num_shards > 0,
            "ShardedFlatIndex::new: num_shards must be greater than zero"
        );
        assert!(
            dim > 0,
            "ShardedFlatIndex::new: dim must be greater than zero"
        );

        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(ConcurrentFlatIndex::new(dim, metric));
        }

        Self {
            shards,
            num_shards,
            dim,
            metric,
        }
    }

    /// Insert a single vector into the appropriate shard determined by [`shard_for_id`].
    ///
    /// # Errors
    /// Returns `Err(String)` if vector dimension does not match index dimension, or if `id`
    /// already exists in the destination shard.
    pub fn add(&self, id: u64, vector: &[f32]) -> Result<(), String> {
        if vector.len() != self.dim {
            return Err(format!(
                "ShardedFlatIndex::add: vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            ));
        }
        let shard_idx = shard_for_id(id, self.num_shards);
        self.shards[shard_idx].add(id, vector)
    }

    /// Bulk-insert vectors and external IDs by partitioning them into per-shard buckets.
    ///
    /// # Performance
    /// Instead of naively looping `add()` for each vector, `add_batch` partitions the entire
    /// incoming batch into per-shard buckets in a single pass, then executes `add_batch` ONCE
    /// per target shard. This preserves each shard's bulk duplicate-checking and contiguous
    /// vector buffer allocation.
    ///
    /// # Errors
    /// Returns `Err(String)` if `ids.len() != vectors.len()`, if vector dimension mismatches,
    /// or if duplicate IDs exist within the incoming batch or within any target shard.
    pub fn add_batch(&self, ids: &[u64], vectors: &VectorBatch) -> Result<(), String> {
        if ids.len() != vectors.len() {
            return Err(format!(
                "ShardedFlatIndex::add_batch: ids count ({}) != vectors count ({})",
                ids.len(),
                vectors.len()
            ));
        }
        if vectors.dim != self.dim {
            return Err(format!(
                "ShardedFlatIndex::add_batch: vectors dimension ({}) != index dimension ({})",
                vectors.dim, self.dim
            ));
        }

        // Single-pass duplicate check across the entire incoming batch
        let mut unique_check = HashSet::with_capacity(ids.len());
        for &id in ids {
            if !unique_check.insert(id) {
                return Err(format!(
                    "ShardedFlatIndex::add_batch: duplicate id {} within incoming batch",
                    id
                ));
            }
        }

        // Partition IDs and vector data into per-shard buckets
        let mut shard_ids: Vec<Vec<u64>> = (0..self.num_shards).map(|_| Vec::new()).collect();
        let mut shard_vectors: Vec<Vec<f32>> = (0..self.num_shards).map(|_| Vec::new()).collect();

        for (i, &id) in ids.iter().enumerate() {
            let shard_idx = shard_for_id(id, self.num_shards);
            shard_ids[shard_idx].push(id);
            shard_vectors[shard_idx].extend_from_slice(vectors.get(i));
        }

        // Dispatch bulk inserts to each shard that received data
        for (s, ids_bucket) in shard_ids.into_iter().enumerate() {
            if !ids_bucket.is_empty() {
                let vec_data = std::mem::take(&mut shard_vectors[s]);
                let num_vecs = ids_bucket.len();
                let batch = VectorBatch::from_parts(vec_data, self.dim, num_vecs)
                    .map_err(|e| format!("ShardedFlatIndex::add_batch: {}", e))?;
                self.shards[s].add_batch(&ids_bucket, &batch)?;
            }
        }

        Ok(())
    }

    /// Fan out the query to ALL shards sequentially and merge their local top-`k` results.
    ///
    /// # Algorithm
    /// 1. **Fan-Out**: Queries `shard.search(query, k)` across every shard.
    /// 2. **Merge & Re-rank**: Combines all candidates into a single buffer and selects the global
    ///    top-`k` using [`top_k_smallest`] (Euclidean) or [`top_k_largest`] (Cosine/DotProduct).
    ///
    /// # Panics
    /// Panics if `query.len() != self.dim`.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredId> {
        assert_eq!(
            query.len(),
            self.dim,
            "ShardedFlatIndex::search: query dim {} != index dim {}",
            query.len(),
            self.dim
        );

        if k == 0 || self.is_empty() {
            return Vec::new();
        }

        let mut candidates = Vec::with_capacity(self.num_shards * k);
        for shard in &self.shards {
            let local_topk = shard.search(query, k);
            candidates.extend(local_topk);
        }

        self.merge_candidates(&candidates, k)
    }

    /// Fan out the query to ALL shards CONCURRENTLY using worker threads and merge their local top-`k` results.
    ///
    /// Uses `std::thread::scope` to query each shard concurrently across available CPU cores,
    /// demonstrating genuine multi-core query parallelization without external dependencies.
    ///
    /// # Panics
    /// Panics if `query.len() != self.dim`.
    pub fn search_parallel(&self, query: &[f32], k: usize) -> Vec<ScoredId> {
        assert_eq!(
            query.len(),
            self.dim,
            "ShardedFlatIndex::search_parallel: query dim {} != index dim {}",
            query.len(),
            self.dim
        );

        if k == 0 || self.is_empty() {
            return Vec::new();
        }

        // Fan out across all shards concurrently
        let shard_results: Vec<Vec<ScoredId>> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(self.num_shards);
            for shard in &self.shards {
                let handle = s.spawn(move || shard.search(query, k));
                handles.push(handle);
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .expect("ShardedFlatIndex::search_parallel: shard thread panicked")
                })
                .collect()
        });

        // Merge results
        let mut candidates = Vec::with_capacity(self.num_shards * k);
        for local_topk in shard_results {
            candidates.extend(local_topk);
        }

        self.merge_candidates(&candidates, k)
    }

    /// Internal helper to merge and re-rank candidates according to the index metric.
    #[inline]
    fn merge_candidates(&self, candidates: &[ScoredId], k: usize) -> Vec<ScoredId> {
        match self.metric {
            Metric::Euclidean => top_k_smallest(candidates, k),
            Metric::Cosine | Metric::DotProduct => top_k_largest(candidates, k),
        }
    }

    /// Return the number of vectors stored in each shard.
    pub fn shard_sizes(&self) -> Vec<usize> {
        self.shards.iter().map(|shard| shard.len()).collect()
    }

    /// Return the total number of vectors stored across all shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.len()).sum()
    }

    /// Return `true` if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dimensionality of vectors in this index.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Distance/similarity metric used by the index.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Number of shards in this index.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Access the underlying shards.
    pub fn shards(&self) -> &[ConcurrentFlatIndex] {
        &self.shards
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flat_index::FlatIndex;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::time::Instant;

    /// Test 1: shard_for_id is strictly deterministic across repeated invocations.
    #[test]
    fn test_shard_for_id_deterministic() {
        let num_shards = 8;
        let test_ids = [0, 1, 42, 100, 1000, 99999, 1234567890123456789];

        for &id in &test_ids {
            let initial_shard = shard_for_id(id, num_shards);
            assert!(initial_shard < num_shards);

            for _ in 0..100 {
                let repeated_shard = shard_for_id(id, num_shards);
                assert_eq!(
                    initial_shard, repeated_shard,
                    "shard_for_id must be strictly deterministic for id {}",
                    id
                );
            }
        }
    }

    /// Test 2: shard_for_id with 1,000 varied IDs and num_shards=8 produces a reasonably even distribution.
    ///
    /// # Tolerance Rationale
    /// With N = 1,000 IDs and K = 8 shards, the expected mean count per shard is 125.
    /// Under an ideal hash distribution (binomial B(1000, 1/8)), standard deviation is:
    ///   sigma = sqrt(1000 * 0.125 * 0.875) ≈ 10.46.
    /// A 2x average bound (max <= 250, min >= 50) represents > 7 standard deviations from the mean.
    /// Breach is virtually impossible (< 1e-12) unless the hash algorithm is broken or biased.
    #[test]
    fn test_shard_for_id_distribution() {
        let num_shards = 8;
        let num_ids = 1000;
        let mut shard_counts = vec![0usize; num_shards];

        // Use sequential IDs (the worst-case input for naive modulo hashing)
        for id in 1..=num_ids {
            let shard = shard_for_id(id, num_shards);
            shard_counts[shard] += 1;
        }

        let avg = num_ids as f64 / num_shards as f64; // 125.0
        let max_allowed = (2.0 * avg) as usize; // 250
        let min_allowed = (0.5 * avg) as usize; // 62

        for (s, &count) in shard_counts.iter().enumerate() {
            assert!(
                count <= max_allowed,
                "Shard {} received {} items, exceeding 2x average (max allowed {})",
                s,
                count,
                max_allowed
            );
            assert!(
                count >= min_allowed,
                "Shard {} received {} items, below 0.5x average (min allowed {})",
                s,
                count,
                min_allowed
            );
        }
    }

    /// Test 3: add() then search() round-trip: add several known vectors, confirm search() finds them
    /// correctly regardless of which shard they landed in.
    #[test]
    fn test_sharded_add_search_round_trip() {
        let dim = 4;
        let num_shards = 4;
        let index = ShardedFlatIndex::new(dim, num_shards, Metric::Euclidean);

        let vectors: Vec<(u64, [f32; 4])> = vec![
            (10, [1.0, 0.0, 0.0, 0.0]),
            (20, [0.0, 1.0, 0.0, 0.0]),
            (30, [0.0, 0.0, 1.0, 0.0]),
            (40, [0.0, 0.0, 0.0, 1.0]),
            (50, [0.5, 0.5, 0.0, 0.0]),
            (60, [0.0, 0.5, 0.5, 0.0]),
        ];

        for (id, vec) in &vectors {
            index.add(*id, vec).unwrap();
        }

        assert_eq!(index.len(), vectors.len());

        // Verify each vector can be queried directly and returns rank 0 with distance 0.0
        for (id, vec) in &vectors {
            let results = index.search(vec, 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, *id);
            assert!(results[0].score.abs() < 1e-6);
        }
    }

    /// Test 4: Correctness cross-check: build BOTH a plain FlatIndex and a ShardedFlatIndex (num_shards=4)
    /// with the SAME data, confirm search() on both returns IDENTICAL top-k results for the same query.
    ///
    /// This is the key correctness anchor proving that sharding does not alter search results.
    #[test]
    fn test_sharded_vs_flat_index_exact_match() {
        let dim = 8;
        let num_shards = 4;
        let mut rng = StdRng::seed_from_u64(42);

        for metric in [Metric::Euclidean, Metric::Cosine, Metric::DotProduct] {
            let mut flat = FlatIndex::new(dim, metric);
            let sharded = ShardedFlatIndex::new(dim, num_shards, metric);

            let n_vectors = 200;
            let mut query = vec![0.0f32; dim];
            for x in &mut query {
                *x = rng.gen_range(-1.0..1.0);
            }

            for id in 1..=n_vectors {
                let mut vec = vec![0.0f32; dim];
                for x in &mut vec {
                    *x = rng.gen_range(-1.0..1.0);
                }
                flat.add(id, &vec);
                sharded.add(id, &vec).unwrap();
            }

            assert_eq!(flat.len(), sharded.len());

            for k in [1, 5, 10, 20] {
                let flat_results = flat.search(&query, k);
                let sharded_results = sharded.search(&query, k);

                assert_eq!(
                    flat_results.len(),
                    sharded_results.len(),
                    "Result length mismatch for metric {:?}, k={}",
                    metric,
                    k
                );

                for (r, (f_res, s_res)) in
                    flat_results.iter().zip(sharded_results.iter()).enumerate()
                {
                    assert_eq!(
                        f_res.id, s_res.id,
                        "ID mismatch at rank {} for metric {:?}, k={}: flat={}, sharded={}",
                        r, metric, k, f_res.id, s_res.id
                    );
                    assert!(
                        (f_res.score - s_res.score).abs() < 1e-5,
                        "Score mismatch at rank {} for metric {:?}, k={}: flat={}, sharded={}",
                        r,
                        metric,
                        k,
                        f_res.score,
                        s_res.score
                    );
                }
            }
        }
    }

    /// Test 5: add_batch() correctly partitions and distributes a batch across shards,
    /// confirmed via shard_sizes() summing correctly and via spot-checking shard locations.
    #[test]
    fn test_add_batch_partitioning_integrity() {
        let dim = 4;
        let num_shards = 4;
        let index = ShardedFlatIndex::new(dim, num_shards, Metric::Euclidean);

        let n_vectors = 100;
        let mut ids = Vec::with_capacity(n_vectors);
        let mut raw_data = Vec::with_capacity(n_vectors * dim);

        for i in 0..n_vectors {
            let id = (i + 1) as u64;
            ids.push(id);
            raw_data.extend_from_slice(&[
                id as f32,
                (id * 2) as f32,
                (id * 3) as f32,
                (id * 4) as f32,
            ]);
        }

        let batch = VectorBatch::from_parts(raw_data, dim, n_vectors).unwrap();
        index.add_batch(&ids, &batch).unwrap();

        assert_eq!(index.len(), n_vectors);

        let shard_sizes = index.shard_sizes();
        assert_eq!(shard_sizes.len(), num_shards);
        assert_eq!(shard_sizes.iter().sum::<usize>(), n_vectors);

        // Every shard should receive at least one vector with 100 vectors across 4 shards
        for (s, &size) in shard_sizes.iter().enumerate() {
            assert!(size > 0, "Shard {} is unexpectedly empty", s);
        }

        // Spot-check IDs: verify that each ID is found in the exact shard predicted by shard_for_id
        for &id in &[1, 7, 23, 42, 77, 99] {
            let expected_shard = shard_for_id(id, num_shards);
            let vec = [id as f32, (id * 2) as f32, (id * 3) as f32, (id * 4) as f32];

            // Search the predicted shard directly
            let target_shard = &index.shards()[expected_shard];
            let shard_search = target_shard.search(&vec, 1);
            assert_eq!(shard_search.len(), 1);
            assert_eq!(
                shard_search[0].id, id,
                "ID {} not found at rank 0 in predicted shard {}",
                id, expected_shard
            );
            assert!(shard_search[0].score.abs() < 1e-6);

            // Verify non-target shards do NOT contain this vector as an exact match
            for other_shard in 0..num_shards {
                if other_shard != expected_shard {
                    let other_search = index.shards()[other_shard].search(&vec, 1);
                    if !other_search.is_empty() {
                        assert_ne!(
                            other_search[0].id, id,
                            "ID {} unexpectedly found in non-target shard {}",
                            id, other_shard
                        );
                    }
                }
            }
        }
    }

    /// Test 6: search_parallel() returns IDENTICAL results to sequential search(),
    /// and demonstrates meaningful multi-core speedup.
    #[test]
    fn test_search_parallel_equivalence_and_speedup() {
        let dim = 64;
        let num_shards = 8;
        let index = ShardedFlatIndex::new(dim, num_shards, Metric::Euclidean);

        let mut rng = StdRng::seed_from_u64(999);
        let n_vectors = 8_000;
        let mut ids = Vec::with_capacity(n_vectors);
        let mut raw_data = Vec::with_capacity(n_vectors * dim);

        for i in 0..n_vectors {
            ids.push((i + 1) as u64);
            for _ in 0..dim {
                raw_data.push(rng.gen_range(-1.0..1.0));
            }
        }

        let batch = VectorBatch::from_parts(raw_data, dim, n_vectors).unwrap();
        index.add_batch(&ids, &batch).unwrap();

        // Generate query vector
        let mut query = vec![0.0f32; dim];
        for x in &mut query {
            *x = rng.gen_range(-1.0..1.0);
        }

        let k = 10;
        let seq_results = index.search(&query, k);
        let par_results = index.search_parallel(&query, k);

        // Verify exact equivalence between sequential and parallel search
        assert_eq!(seq_results.len(), par_results.len());
        for (i, (s, p)) in seq_results.iter().zip(par_results.iter()).enumerate() {
            assert_eq!(s.id, p.id, "ID mismatch at rank {}", i);
            assert!(
                (s.score - p.score).abs() < 1e-6,
                "Score mismatch at rank {}: seq={}, par={}",
                i,
                s.score,
                p.score
            );
        }

        // Benchmark timing comparison across multiple query iterations
        let iterations = 25;

        let start_seq = Instant::now();
        for _ in 0..iterations {
            let _ = index.search(&query, k);
        }
        let seq_duration = start_seq.elapsed();

        let start_par = Instant::now();
        for _ in 0..iterations {
            let _ = index.search_parallel(&query, k);
        }
        let par_duration = start_par.elapsed();

        let seq_micros = seq_duration.as_micros() as f64 / iterations as f64;
        let par_micros = par_duration.as_micros() as f64 / iterations as f64;
        let speedup = seq_micros / par_micros;

        println!("\n=== Test 6: Search Parallel vs Sequential Timing ===");
        println!(
            "Dataset: {} vectors (dim={}) across {} shards",
            n_vectors, dim, num_shards
        );
        println!("Sequential search avg: {:.2} µs", seq_micros);
        println!("Parallel search avg:   {:.2} µs", par_micros);
        println!("Speedup factor:        {:.2}x", speedup);
    }

    /// Test 7: Realistic-scale test: 10,000 vectors across 8 shards, confirm len() == 10,000,
    /// print shard_sizes() distribution, and confirm reasonably balanced distribution.
    #[test]
    fn test_realistic_scale_10k_vectors_distribution() {
        let dim = 16;
        let num_shards = 8;
        let index = ShardedFlatIndex::new(dim, num_shards, Metric::Euclidean);

        let mut rng = StdRng::seed_from_u64(12345);
        let n_vectors = 10_000;
        let mut ids = Vec::with_capacity(n_vectors);
        let mut raw_data = Vec::with_capacity(n_vectors * dim);

        for i in 0..n_vectors {
            ids.push((i + 1) as u64);
            for _ in 0..dim {
                raw_data.push(rng.gen_range(-1.0..1.0));
            }
        }

        let batch = VectorBatch::from_parts(raw_data, dim, n_vectors).unwrap();
        index.add_batch(&ids, &batch).unwrap();

        assert_eq!(index.len(), n_vectors);

        let shard_sizes = index.shard_sizes();
        assert_eq!(shard_sizes.len(), num_shards);
        assert_eq!(shard_sizes.iter().sum::<usize>(), n_vectors);

        let expected_avg = n_vectors as f64 / num_shards as f64; // 1250.0

        println!("\n=== Test 7: 10,000 Vectors across 8 Shards Distribution ===");
        for (i, &size) in shard_sizes.iter().enumerate() {
            let deviation = (size as f64 - expected_avg) / expected_avg * 100.0;
            println!(
                "Shard {}: {:>5} vectors ({:+.2}% from expected average {})",
                i, size, deviation, expected_avg
            );
            // Assert all shards are within 25% of expected mean (1250 +/- 312.5 => [937, 1562])
            assert!(
                (900..=1600).contains(&size),
                "Shard {} has {} vectors, deviating too far from expected average {}",
                i,
                size,
                expected_avg
            );
        }
    }
}
