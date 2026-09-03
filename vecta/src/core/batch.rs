// Batched distance computation: one query vs. N stored vectors.
//
// This is the actual hot path of every search operation in vecta — brute-force,
// IVF cluster scans, HNSW neighbor comparisons all funnel through here.
//
// # Why a flat Vec<f32> instead of Vec<Vec<f32>>?
//
// Vec<Vec<f32>> means N separate heap allocations scattered across memory.
// When iterating row-by-row, the CPU cache prefetcher cannot predict the next
// pointer, leading to cache misses on nearly every row access.
//
// A single flat Vec<f32> of length (N × D) stores all rows contiguously.
// Sequential row iteration becomes a linear memory scan — the hardware
// prefetcher keeps the cache lines warm, and the compiler can auto-vectorize
// the inner loops (SIMD). This is exactly how FAISS and NumPy lay out their
// matrices internally.

use super::vector::{cosine_similarity, dot_product, euclidean_distance};

/// A contiguous batch of vectors stored in a single flat buffer.
///
/// Row `i` occupies `data[i * dim .. (i + 1) * dim]`.
pub struct VectorBatch {
    /// Flat storage: length = `num_vectors * dim`.
    pub data: Vec<f32>,
    /// Dimensionality of each vector.
    pub dim: usize,
    /// Number of vectors currently stored.
    pub num_vectors: usize,
}

impl VectorBatch {
    /// Create an empty batch for vectors of the given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
            num_vectors: 0,
        }
    }

    /// Append a vector to the batch.
    ///
    /// # Panics
    /// Panics if `vector.len() != self.dim`.
    pub fn push(&mut self, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.dim,
            "VectorBatch::push: expected dim {}, got {}",
            self.dim,
            vector.len()
        );
        self.data.extend_from_slice(vector);
        self.num_vectors += 1;
    }

    /// Return an immutable slice view of the vector at `index`.
    ///
    /// Zero-copy — just a borrow into the flat buffer.
    ///
    /// # Panics
    /// Panics if `index >= self.num_vectors`.
    #[inline]
    pub fn get(&self, index: usize) -> &[f32] {
        assert!(
            index < self.num_vectors,
            "VectorBatch::get: index {} out of bounds (num_vectors = {})",
            index,
            self.num_vectors
        );
        let start = index * self.dim;
        &self.data[start..start + self.dim]
    }
}

/// Compute dot products of `query` against every row in `batch`.
///
/// Returns a `Vec<f32>` of length `batch.num_vectors`, pre-allocated
/// to avoid repeated reallocation.
///
/// # Panics
/// Panics if `query.len() != batch.dim`.
#[inline]
pub fn batch_dot_product(query: &[f32], batch: &VectorBatch) -> Vec<f32> {
    assert_eq!(
        query.len(),
        batch.dim,
        "batch_dot_product: query dim {} != batch dim {}",
        query.len(),
        batch.dim
    );
    let mut results = Vec::with_capacity(batch.num_vectors);
    for i in 0..batch.num_vectors {
        results.push(dot_product(query, batch.get(i)));
    }
    results
}

/// Compute Euclidean distances from `query` to every row in `batch`.
///
/// Returns a `Vec<f32>` of length `batch.num_vectors`, pre-allocated
/// to avoid repeated reallocation.
///
/// # Panics
/// Panics if `query.len() != batch.dim`.
#[inline]
pub fn batch_euclidean_distance(query: &[f32], batch: &VectorBatch) -> Vec<f32> {
    assert_eq!(
        query.len(),
        batch.dim,
        "batch_euclidean_distance: query dim {} != batch dim {}",
        query.len(),
        batch.dim
    );
    let mut results = Vec::with_capacity(batch.num_vectors);
    for i in 0..batch.num_vectors {
        results.push(euclidean_distance(query, batch.get(i)));
    }
    results
}

/// Compute cosine similarities of `query` against every row in `batch`.
///
/// Returns a `Vec<f32>` of length `batch.num_vectors`, pre-allocated
/// to avoid repeated reallocation.
///
/// # Panics
/// Panics if `query.len() != batch.dim`.
#[inline]
pub fn batch_cosine_similarity(query: &[f32], batch: &VectorBatch) -> Vec<f32> {
    assert_eq!(
        query.len(),
        batch.dim,
        "batch_cosine_similarity: query dim {} != batch dim {}",
        query.len(),
        batch.dim
    );
    let mut results = Vec::with_capacity(batch.num_vectors);
    for i in 0..batch.num_vectors {
        results.push(cosine_similarity(query, batch.get(i)));
    }
    results
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

    // --- VectorBatch struct tests ---

    #[test]
    fn test_push_increments_count() {
        let mut batch = VectorBatch::new(3);
        assert_eq!(batch.num_vectors, 0);
        batch.push(&[1.0, 2.0, 3.0]);
        assert_eq!(batch.num_vectors, 1);
        batch.push(&[4.0, 5.0, 6.0]);
        assert_eq!(batch.num_vectors, 2);
        assert_eq!(batch.data.len(), 6); // 2 * 3
    }

    #[test]
    fn test_get_returns_correct_slices() {
        let mut batch = VectorBatch::new(3);
        batch.push(&[1.0, 2.0, 3.0]);
        batch.push(&[4.0, 5.0, 6.0]);
        batch.push(&[7.0, 8.0, 9.0]);

        assert_eq!(batch.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(batch.get(1), &[4.0, 5.0, 6.0]);
        assert_eq!(batch.get(2), &[7.0, 8.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "expected dim 3, got 2")]
    fn test_push_dimension_mismatch_panics() {
        let mut batch = VectorBatch::new(3);
        batch.push(&[1.0, 2.0]); // wrong dim
    }

    // --- Batched distance function tests ---
    //
    // Test vectors (dim=3):
    //   v0 = [1, 0, 0]
    //   v1 = [0, 1, 0]
    //   v2 = [1, 1, 0]
    // Query = [1, 1, 1]

    fn make_test_batch() -> VectorBatch {
        let mut batch = VectorBatch::new(3);
        batch.push(&[1.0, 0.0, 0.0]);
        batch.push(&[0.0, 1.0, 0.0]);
        batch.push(&[1.0, 1.0, 0.0]);
        batch
    }

    #[test]
    fn test_batch_dot_product() {
        let batch = make_test_batch();
        let query = [1.0_f32, 1.0, 1.0];
        let results = batch_dot_product(&query, &batch);

        assert_eq!(results.len(), 3);
        // dot([1,1,1], [1,0,0]) = 1
        assert_approx(results[0], 1.0, 1e-4);
        // dot([1,1,1], [0,1,0]) = 1
        assert_approx(results[1], 1.0, 1e-4);
        // dot([1,1,1], [1,1,0]) = 2
        assert_approx(results[2], 2.0, 1e-4);
    }
    

    #[test]
    fn test_batch_euclidean_distance() {
        let batch = make_test_batch();
        let query = [1.0_f32, 1.0, 1.0];
        let results = batch_euclidean_distance(&query, &batch);

        assert_eq!(results.len(), 3);
        // dist([1,1,1], [1,0,0]) = sqrt(0 + 1 + 1) = sqrt(2)
        assert_approx(results[0], 2.0_f32.sqrt(), 1e-4);
        // dist([1,1,1], [0,1,0]) = sqrt(1 + 0 + 1) = sqrt(2)
        assert_approx(results[1], 2.0_f32.sqrt(), 1e-4);
        // dist([1,1,1], [1,1,0]) = sqrt(0 + 0 + 1) = 1.0
        assert_approx(results[2], 1.0, 1e-4);
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let batch = make_test_batch();
        let query = [1.0_f32, 1.0, 1.0];
        let results = batch_cosine_similarity(&query, &batch);

        assert_eq!(results.len(), 3);
        // cos([1,1,1], [1,0,0]) = 1 / (sqrt(3) * 1) ≈ 0.5774
        assert_approx(results[0], 1.0 / 3.0_f32.sqrt(), 1e-4);
        // cos([1,1,1], [0,1,0]) = 1 / (sqrt(3) * 1) ≈ 0.5774
        assert_approx(results[1], 1.0 / 3.0_f32.sqrt(), 1e-4);
        // cos([1,1,1], [1,1,0]) = 2 / (sqrt(3) * sqrt(2)) ≈ 0.8165
        assert_approx(results[2], 2.0 / (3.0_f32.sqrt() * 2.0_f32.sqrt()), 1e-4);
    }

    // --- Timing test ---

    #[test]
    fn bench_batch_euclidean_10k_128d() {
        use std::time::Instant;

        let dim = 128;
        let n = 10_000;

        // Deterministic pseudo-random fill (no external crate needed).
        // Simple LCG: x_{n+1} = (a * x_n + c) mod m, mapped to [0, 1).
        let mut rng_state: u64 = 42;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Take upper bits, map to [0, 1)
            ((rng_state >> 33) as f32) / (u32::MAX as f32)
        };

        let mut batch = VectorBatch::new(dim);
        batch.data.reserve(n * dim);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            batch.push(&v);
        }

        let query: Vec<f32> = (0..dim).map(|_| next_f32()).collect();

        let start = Instant::now();
        let results = batch_euclidean_distance(&query, &batch);
        let elapsed = start.elapsed();

        assert_eq!(results.len(), n);
        println!(
            "\n[BENCH] batch_euclidean_distance: {} vectors × {} dims in {:.3}ms",
            n,
            dim,
            elapsed.as_secs_f64() * 1000.0
        );
    }
}
