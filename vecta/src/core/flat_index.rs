// Brute-force flat index — the ground-truth oracle for vecta.
//
// Every future ANN algorithm (IVF, HNSW, PQ) gets its recall@k validated
// against this index. It stays in the codebase permanently, mirroring
// FAISS's IndexFlatL2/IndexFlatIP role as a correctness baseline.

use super::batch::{
    batch_cosine_similarity, batch_dot_product, batch_euclidean_distance, VectorBatch,
};
use super::topk::{top_k_largest, top_k_smallest, ScoredId};

/// Distance / similarity metric used for search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// A flat (brute-force) vector index backed by contiguous storage.
///
/// Vectors are stored in a [`VectorBatch`] (single flat `Vec<f32>`),
/// with a parallel `ids` vec mapping each row index to an external ID.
#[derive(Debug, Clone)]
pub struct FlatIndex {
    /// Contiguous vector storage.
    pub batch: VectorBatch,
    /// External ID for each row: `ids[i]` corresponds to `batch.get(i)`.
    pub ids: Vec<u64>,
    /// Distance metric used for search queries.
    pub metric: Metric,
}

impl FlatIndex {
    /// Create an empty index for vectors of the given dimensionality.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            batch: VectorBatch::new(dim),
            ids: Vec::new(),
            metric,
        }
    }

    /// Insert a single vector with its external ID.
    ///
    /// # Panics
    /// - If `vector.len() != self.dim()`.
    /// - If `id` already exists in the index.
    pub fn add(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.batch.dim,
            "FlatIndex::add: expected dim {}, got {}",
            self.batch.dim,
            vector.len()
        );
        assert!(
            !self.ids.contains(&id),
            "FlatIndex::add: duplicate id {}",
            id
        );
        self.batch.push(vector);
        self.ids.push(id);
    }

    /// Bulk-insert vectors with their external IDs.
    ///
    /// Performs ONE upfront duplicate-ID check across both the incoming batch
    /// and the existing index, rather than N individual checks per vector.
    ///
    /// # Panics
    /// - If `ids.len() != vectors.num_vectors`.
    /// - If `vectors.dim != self.batch.dim`.
    /// - If any ID in `ids` already exists in the index.
    /// - If `ids` contains duplicates within itself.
    pub fn add_batch(&mut self, ids: &[u64], vectors: &VectorBatch) {
        assert_eq!(
            ids.len(),
            vectors.num_vectors,
            "FlatIndex::add_batch: ids count ({}) != vectors count ({})",
            ids.len(),
            vectors.num_vectors
        );
        assert_eq!(
            vectors.dim, self.batch.dim,
            "FlatIndex::add_batch: incoming dim {} != index dim {}",
            vectors.dim, self.batch.dim
        );

        // ONE bulk duplicate check: incoming IDs vs existing index + within themselves.
        for (i, &new_id) in ids.iter().enumerate() {
            assert!(
                !self.ids.contains(&new_id),
                "FlatIndex::add_batch: duplicate id {} (already in index)",
                new_id
            );
            // Also check for duplicates within the incoming batch itself.
            assert!(
                !ids[..i].contains(&new_id),
                "FlatIndex::add_batch: duplicate id {} (within incoming batch)",
                new_id
            );
        }

        // All checks passed — append data in bulk.
        self.batch.data.extend_from_slice(&vectors.data);
        self.batch.num_vectors += vectors.num_vectors;
        self.ids.extend_from_slice(ids);
    }

    /// Number of vectors currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.batch.num_vectors
    }

    /// Returns `true` if the index contains no vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.batch.num_vectors == 0
    }

    /// Dimensionality of vectors in this index.
    #[inline]
    pub fn dim(&self) -> usize {
        self.batch.dim
    }

    /// Look up a vector by its external ID.
    ///
    /// Linear scan through `self.ids` — acceptable for correctness testing,
    /// not intended as a high-performance lookup path.
    ///
    /// Returns `None` if the ID is not found.
    pub fn get_vector(&self, id: u64) -> Option<&[f32]> {
        self.ids
            .iter()
            .position(|&stored_id| stored_id == id)
            .map(|idx| self.batch.get(idx))
    }

    /// Brute-force k-nearest-neighbor search.
    ///
    /// Computes the distance/similarity between `query` and every stored vector,
    /// then returns the top `k` results sorted by relevance (ascending distance
    /// for Euclidean, descending similarity for Cosine/DotProduct).
    ///
    /// This is the **correctness oracle** for the entire project: every future
    /// ANN algorithm validates its recall against this method's output.
    ///
    /// # Panics
    /// Panics if `query.len() != self.dim()`.
    ///
    /// # Edge cases
    /// - Empty index → returns empty `Vec`
    /// - `k == 0` → returns empty `Vec`
    /// - `k > self.len()` → returns all vectors, correctly sorted
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredId> {
        // Edge cases: nothing to search.
        if k == 0 || self.is_empty() {
            // Still validate dimension even on empty index — catch misuse early.
            assert_eq!(
                query.len(),
                self.batch.dim,
                "FlatIndex::search: query dim {} != index dim {}",
                query.len(),
                self.batch.dim
            );
            return Vec::new();
        }

        assert_eq!(
            query.len(),
            self.batch.dim,
            "FlatIndex::search: query dim {} != index dim {}",
            query.len(),
            self.batch.dim
        );

        // Step 1: Compute all distances/similarities in one batched pass.
        let scores = match self.metric {
            Metric::Euclidean => batch_euclidean_distance(query, &self.batch),
            Metric::Cosine => batch_cosine_similarity(query, &self.batch),
            Metric::DotProduct => batch_dot_product(query, &self.batch),
        };

        // Step 2: Zip scores with external IDs into ScoredId candidates.
        let candidates: Vec<ScoredId> = scores
            .into_iter()
            .zip(self.ids.iter())
            .map(|(score, &id)| ScoredId { id, score })
            .collect();

        // Step 3: Select top-k using the correct direction for the metric.
        match self.metric {
            // Euclidean: smaller distance = closer = better.
            Metric::Euclidean => top_k_smallest(&candidates, k),
            // Cosine / DotProduct: larger similarity = more aligned = better.
            Metric::Cosine | Metric::DotProduct => top_k_largest(&candidates, k),
        }
    }

    /// Run [`search`] once per query row in the given [`VectorBatch`].
    ///
    /// Returns one `Vec<ScoredId>` per query, in order. Each individual
    /// query runs single-threaded — parallelism across queries is a later
    /// optimization phase.
    ///
    /// # Panics
    /// Panics if `queries.dim != self.dim()`.
    pub fn search_batch(&self, queries: &VectorBatch, k: usize) -> Vec<Vec<ScoredId>> {
        assert_eq!(
            queries.dim, self.batch.dim,
            "FlatIndex::search_batch: queries dim {} != index dim {}",
            queries.dim, self.batch.dim
        );
        (0..queries.num_vectors)
            .map(|i| self.search(queries.get(i), k))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(actual: f32, expected: f32, epsilon: f32) {
        assert!(
            (actual - expected).abs() < epsilon,
            "expected ≈ {expected}, got {actual} (ε = {epsilon})"
        );
    }

    // ── Shared test vectors (2D) ──────────────────────────────────────
    //
    // Hand-picked so that ALL metrics produce distinct scores (no ties),
    // making expected orderings deterministic.
    //
    //   v0 (id=0): [2.0, 1.0]
    //   v1 (id=1): [0.5, 0.0]
    //   v2 (id=2): [1.0, 3.0]
    //   v3 (id=3): [-1.0, 3.0]
    //   v4 (id=4): [4.0, -3.0]
    //   Query:     [1.0, 1.0]

    fn build_test_index(metric: Metric) -> FlatIndex {
        let mut index = FlatIndex::new(2, metric);
        index.add(0, &[2.0, 1.0]);
        index.add(1, &[0.5, 0.0]);
        index.add(2, &[1.0, 3.0]);
        index.add(3, &[-1.0, 3.0]);
        index.add(4, &[4.0, -3.0]);
        index
    }

    // ── Test 1: Euclidean, hand-verified ──────────────────────────────
    //
    // d(q, v0) = √((1-2)²+(1-1)²)     = √1    = 1.0
    // d(q, v1) = √((1-0.5)²+(1-0)²)   = √1.25 ≈ 1.1180
    // d(q, v2) = √((1-1)²+(1-3)²)     = √4    = 2.0
    // d(q, v3) = √((1-(-1))²+(1-3)²)  = √8    ≈ 2.8284
    // d(q, v4) = √((1-4)²+(1-(-3))²)  = √25   = 5.0
    //
    // Top-3 smallest: v0(1.0), v1(1.1180), v2(2.0)  → IDs [0, 1, 2]
    #[test]
    fn test_search_euclidean_hand_verified() {
        let index = build_test_index(Metric::Euclidean);
        let query = [1.0_f32, 1.0];
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 0);
        assert_approx(results[0].score, 1.0, 1e-4);
        assert_eq!(results[1].id, 1);
        assert_approx(results[1].score, 1.25_f32.sqrt(), 1e-4);
        assert_eq!(results[2].id, 2);
        assert_approx(results[2].score, 2.0, 1e-4);
    }

    // ── Test 2: Cosine similarity, hand-verified ─────────────────────
    //
    // ||q|| = √2
    // cos(q, v0) = (2+1)/(√2·√5)     = 3/√10  ≈ 0.94868
    // cos(q, v1) = (0.5+0)/(√2·0.5)  = 0.5/√0.5 = 1/√2 ≈ 0.70711
    // cos(q, v2) = (1+3)/(√2·√10)    = 4/√20  = 2/√5  ≈ 0.89443
    // cos(q, v3) = (-1+3)/(√2·√10)   = 2/√20  = 1/√5  ≈ 0.44721
    // cos(q, v4) = (4-3)/(√2·√25)    = 1/(5√2)       ≈ 0.14142
    //
    // Top-3 largest: v0(0.94868), v2(0.89443), v1(0.70711)  → IDs [0, 2, 1]
    #[test]
    fn test_search_cosine_hand_verified() {
        let index = build_test_index(Metric::Cosine);
        let query = [1.0_f32, 1.0];
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 0);
        assert_approx(results[0].score, 3.0 / 10.0_f32.sqrt(), 1e-4);
        assert_eq!(results[1].id, 2);
        assert_approx(results[1].score, 2.0 / 5.0_f32.sqrt(), 1e-4);
        assert_eq!(results[2].id, 1);
        assert_approx(results[2].score, 1.0 / 2.0_f32.sqrt(), 1e-4);
    }

    // ── Test 3: Dot product, hand-verified ───────────────────────────
    //
    // dot(q, v0) = 2+1   = 3.0
    // dot(q, v1) = 0.5+0 = 0.5
    // dot(q, v2) = 1+3   = 4.0
    // dot(q, v3) = -1+3  = 2.0
    // dot(q, v4) = 4-3   = 1.0
    //
    // Top-3 largest: v2(4.0), v0(3.0), v3(2.0)  → IDs [2, 0, 3]
    #[test]
    fn test_search_dotproduct_hand_verified() {
        let index = build_test_index(Metric::DotProduct);
        let query = [1.0_f32, 1.0];
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 2);
        assert_approx(results[0].score, 4.0, 1e-4);
        assert_eq!(results[1].id, 0);
        assert_approx(results[1].score, 3.0, 1e-4);
        assert_eq!(results[2].id, 3);
        assert_approx(results[2].score, 2.0, 1e-4);
    }

    // ── Test 4: k > number of vectors ────────────────────────────────
    #[test]
    fn test_search_k_greater_than_len() {
        let index = build_test_index(Metric::Euclidean);
        let query = [1.0_f32, 1.0];
        let results = index.search(&query, 100); // only 5 vectors in index

        assert_eq!(results.len(), 5);
        // Should still be sorted ascending by distance.
        for w in results.windows(2) {
            assert!(
                w[0].score <= w[1].score,
                "results not sorted ascending: {} > {}",
                w[0].score,
                w[1].score
            );
        }
    }

    // ── Test 5: Empty index ──────────────────────────────────────────
    #[test]
    fn test_search_empty_index() {
        let index = FlatIndex::new(2, Metric::Euclidean);
        let query = [1.0_f32, 1.0];
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }

    // ── Test 6: k == 0 ──────────────────────────────────────────────
    #[test]
    fn test_search_k_zero() {
        let index = build_test_index(Metric::Euclidean);
        let query = [1.0_f32, 1.0];
        let results = index.search(&query, 0);
        assert!(results.is_empty());
    }

    // ── Test 7: Wrong query dimension panics ─────────────────────────
    #[test]
    #[should_panic(expected = "query dim 3 != index dim 2")]
    fn test_search_wrong_dimension_panics() {
        let index = build_test_index(Metric::Euclidean);
        let _results = index.search(&[1.0, 2.0, 3.0], 3);
    }

    // ── Test 8: Randomized 1,000-vector correctness ──────────────────
    #[test]
    fn test_search_1000_random_vectors() {
        use super::super::vector::euclidean_distance;

        let dim = 128;
        let n = 1000;
        let k = 10;

        let mut rng: u64 = 42;
        let mut next_f32 = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32) / (u32::MAX as f32)
        };

        // Build index with 1,000 random 128-dim vectors.
        let mut index = FlatIndex::new(dim, Metric::Euclidean);
        // Store vectors so we can cross-check later.
        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            index.add(i as u64, &v);
            all_vectors.push(v);
        }

        let query: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let results = index.search(&query, k);

        // (a) Exactly k results.
        assert_eq!(results.len(), k);

        // (b) Sorted ascending (Euclidean = smaller is better).
        for w in results.windows(2) {
            assert!(
                w[0].score <= w[1].score,
                "Euclidean results not sorted ascending: {} > {}",
                w[0].score,
                w[1].score
            );
        }

        // (c) Cross-check: recompute distance for top-1 result.
        let top1 = &results[0];
        let recomputed = euclidean_distance(&query, &all_vectors[top1.id as usize]);
        assert_approx(top1.score, recomputed, 1e-5);

        // Run the same test for Cosine.
        {
            use super::super::vector::cosine_similarity;

            let mut index_cos = FlatIndex::new(dim, Metric::Cosine);
            for (i, v) in all_vectors.iter().enumerate() {
                index_cos.add(i as u64, v);
            }
            let results_cos = index_cos.search(&query, k);
            assert_eq!(results_cos.len(), k);

            // Sorted descending (Cosine = larger is better).
            for w in results_cos.windows(2) {
                assert!(
                    w[0].score >= w[1].score,
                    "Cosine results not sorted descending: {} < {}",
                    w[0].score,
                    w[1].score
                );
            }

            // Cross-check top-1.
            let top1_cos = &results_cos[0];
            let recomputed_cos = cosine_similarity(&query, &all_vectors[top1_cos.id as usize]);
            assert_approx(top1_cos.score, recomputed_cos, 1e-5);
        }

        // Run the same test for DotProduct.
        {
            use super::super::vector::dot_product;

            let mut index_dp = FlatIndex::new(dim, Metric::DotProduct);
            for (i, v) in all_vectors.iter().enumerate() {
                index_dp.add(i as u64, v);
            }
            let results_dp = index_dp.search(&query, k);
            assert_eq!(results_dp.len(), k);

            // Sorted descending (DotProduct = larger is better).
            for w in results_dp.windows(2) {
                assert!(
                    w[0].score >= w[1].score,
                    "DotProduct results not sorted descending: {} < {}",
                    w[0].score,
                    w[1].score
                );
            }

            // Cross-check top-1.
            let top1_dp = &results_dp[0];
            let recomputed_dp = dot_product(&query, &all_vectors[top1_dp.id as usize]);
            assert_approx(top1_dp.score, recomputed_dp, 1e-5);
        }
    }

    // ── Test 9: Timing baseline — 10k vectors, 100 queries ──────────
    #[test]
    fn bench_search_10k_100queries() {
        use std::time::Instant;

        let dim = 128;
        let n = 10_000;
        let num_queries = 100;
        let k = 10;

        let mut rng: u64 = 7;
        let mut next_f32 = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32) / (u32::MAX as f32)
        };

        // Build index.
        let mut batch = VectorBatch::new(dim);
        let ids: Vec<u64> = (0..n as u64).collect();
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            batch.push(&v);
        }
        let mut index = FlatIndex::new(dim, Metric::Euclidean);
        index.add_batch(&ids, &batch);

        // Build query batch.
        let mut queries = VectorBatch::new(dim);
        for _ in 0..num_queries {
            let q: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            queries.push(&q);
        }

        // Time search_batch.
        let start = Instant::now();
        let all_results = index.search_batch(&queries, k);
        let elapsed = start.elapsed();

        // Sanity check: each query got k results.
        assert_eq!(all_results.len(), num_queries);
        for (i, results) in all_results.iter().enumerate() {
            assert_eq!(
                results.len(),
                k,
                "query {} returned {} results, expected {}",
                i,
                results.len(),
                k
            );
        }

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let avg_ms = total_ms / num_queries as f64;
        println!(
            "\n[BENCH] FlatIndex::search_batch (Phase 6 baseline):\n  \
             index:   {} vectors × {} dims\n  \
             queries: {} × k={}\n  \
             total:   {:.3}ms\n  \
             avg/q:   {:.3}ms\n  \
             QPS:     {:.0}",
            n,
            dim,
            num_queries,
            k,
            total_ms,
            avg_ms,
            num_queries as f64 / (total_ms / 1000.0)
        );
    }

    // ── Previous Phase 4 tests (preserved) ───────────────────────────

    #[test]
    fn test_empty_index() {
        let index = FlatIndex::new(3, Metric::Euclidean);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_three_vectors() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(1, &[1.0, 2.0, 3.0]);
        index.add(2, &[4.0, 5.0, 6.0]);
        index.add(3, &[7.0, 8.0, 9.0]);

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    #[should_panic(expected = "expected dim 3, got 2")]
    fn test_add_wrong_dimension_panics() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(1, &[1.0, 2.0]); // dim 2 into dim-3 index
    }

    #[test]
    #[should_panic(expected = "duplicate id 42")]
    fn test_add_duplicate_id_panics() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(42, &[1.0, 2.0, 3.0]);
        index.add(42, &[4.0, 5.0, 6.0]); // duplicate
    }

    #[test]
    fn test_add_batch_inserts_correctly() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(0, &[0.0, 0.0, 0.0]); // pre-existing

        let mut incoming = VectorBatch::new(3);
        incoming.push(&[1.0, 2.0, 3.0]);
        incoming.push(&[4.0, 5.0, 6.0]);
        incoming.push(&[7.0, 8.0, 9.0]);

        index.add_batch(&[10, 20, 30], &incoming);
        assert_eq!(index.len(), 4); // 1 + 3
    }

    #[test]
    fn test_get_vector_found() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(10, &[1.0, 2.0, 3.0]);
        index.add(20, &[4.0, 5.0, 6.0]);
        index.add(30, &[7.0, 8.0, 9.0]);

        let v = index.get_vector(20).expect("id 20 should exist");
        assert_eq!(v, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_get_vector_not_found() {
        let index = FlatIndex::new(3, Metric::Euclidean);
        assert!(index.get_vector(999).is_none());
    }

    #[test]
    fn test_add_batch_1000_vectors() {
        let dim = 16;
        let n = 1000;

        let mut batch = VectorBatch::new(dim);
        let ids: Vec<u64> = (0..n as u64).collect();

        // Deterministic fill.
        let mut rng: u64 = 7;
        for _ in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng >> 33) as f32) / (u32::MAX as f32)
                })
                .collect();
            batch.push(&v);
        }

        let mut index = FlatIndex::new(dim, Metric::Euclidean);
        index.add_batch(&ids, &batch);
        assert_eq!(index.len(), n);
    }
}
