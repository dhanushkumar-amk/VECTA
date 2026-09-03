//! K-Means Clustering Algorithm.
//!
//! Provides Lloyd's algorithm with k-means++ initialization for vector quantization
//! and Voronoi partitioning. Used as the foundational clustering engine for
//! Inverted File (IVF) index building (Phases 10-14).
//!
//! # Distance Metric Design Note
//! K-means mathematically minimizes the sum of squared Euclidean (L2) distances
//! from data points to their assigned cluster centroids (inertia / Voronoi cells).
//! The arithmetic mean of a subset of vectors is the unique analytical minimizer
//! for squared Euclidean distance. Therefore, L2 Euclidean distance is used here
//! for partitioning, even if an index built on top later uses Cosine or Inner
//! Product during query-time ANN ranking.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::core::batch::{batch_euclidean_distance, VectorBatch};
use crate::core::vector::euclidean_distance;

/// Configuration parameters for K-Means clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeansConfig {
    /// Number of clusters (centroids) to produce.
    pub k: usize,
    /// Maximum number of Lloyd iterations before terminating.
    pub max_iterations: usize,
    /// Convergence threshold: early stop if sum of centroid shifts < tolerance.
    pub tolerance: f32,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 8,
            max_iterations: 100,
            tolerance: 1e-4,
        }
    }
}

/// Result of K-Means clustering.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// The learned cluster centroids (`k` vectors of dimension `dim`).
    pub centroids: VectorBatch,
    /// Cluster assignments: `assignments[i]` is the centroid index for vector `i`.
    pub assignments: Vec<usize>,
    /// Number of Lloyd iterations executed until convergence or max_iterations.
    pub iterations: usize,
}

/// Compute squared Euclidean distance between two equal-length slices.
#[inline]
fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

/// Initialize `k` cluster centroids using the k-means++ seeding scheme.
///
/// 1. First centroid is chosen uniformly at random from `data`.
/// 2. Each subsequent centroid is chosen with probability proportional to
///    its squared distance to the nearest already-chosen centroid ($D(x)^2$).
///
/// This avoids the poor local minima and clumping of standard random initialization.
pub fn kmeans_plus_plus_init(data: &VectorBatch, k: usize, rng: &mut impl Rng) -> VectorBatch {
    let n = data.len();
    let dim = data.dim();
    assert!(n > 0, "kmeans_plus_plus_init: data batch cannot be empty");
    assert!(k > 0, "kmeans_plus_plus_init: k must be greater than 0");

    let mut centroids = VectorBatch::new(dim);

    // If k >= n, use all available points (and repeat if k > n)
    if k >= n {
        for i in 0..k {
            centroids.push(data.get(i % n));
        }
        return centroids;
    }

    // Step 1: Pick the first centroid uniformly at random
    let first_idx = rng.gen_range(0..n);
    centroids.push(data.get(first_idx));

    // Maintain D(x)^2: minimum squared distance to nearest chosen centroid for each point
    let mut min_dists_sq: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let d2 = squared_euclidean_distance(data.get(i), centroids.get(0));
        min_dists_sq.push(d2);
    }

    // Step 2: Greedily pick remaining k - 1 centroids with D(x)^2 probability distribution
    for _ in 1..k {
        let total_weight: f32 = min_dists_sq.iter().sum();

        let chosen_idx = if total_weight <= 1e-12 {
            // All points are identical or already chosen as centroids
            rng.gen_range(0..n)
        } else {
            // Sample proportional to D(x)^2
            let mut threshold: f32 = rng.gen_range(0.0..total_weight);
            let mut selected = n - 1;
            for (idx, &w) in min_dists_sq.iter().enumerate() {
                if threshold <= w {
                    selected = idx;
                    break;
                }
                threshold -= w;
            }
            selected
        };

        let new_centroid = data.get(chosen_idx);
        centroids.push(new_centroid);

        // Update minimum squared distances with the newly added centroid
        for (i, d2_min) in min_dists_sq.iter_mut().enumerate() {
            let d2 = squared_euclidean_distance(data.get(i), new_centroid);
            if d2 < *d2_min {
                *d2_min = d2;
            }
        }
    }

    centroids
}

/// Run K-Means clustering on the input `data` using Lloyd's algorithm with k-means++ seeding.
///
/// # Arguments
/// * `data` - Set of vectors to cluster.
/// * `config` - Number of clusters `k`, iteration cap, and convergence tolerance.
/// * `seed` - 64-bit random seed for deterministic, reproducible clustering.
///
/// # Returns
/// A [`KMeansResult`] containing centroids, vector assignments, and total iterations.
pub fn kmeans(data: &VectorBatch, config: &KMeansConfig, seed: u64) -> KMeansResult {
    let n = data.len();
    let dim = data.dim();

    if n == 0 || config.k == 0 {
        return KMeansResult {
            centroids: VectorBatch::new(dim),
            assignments: Vec::new(),
            iterations: 0,
        };
    }

    let k = config.k.min(n);
    let mut rng = StdRng::seed_from_u64(seed);

    // 1. Initialize centroids via k-means++
    let mut centroids = kmeans_plus_plus_init(data, k, &mut rng);

    // Pre-allocated reusable buffers across iterations to eliminate per-loop heap allocations
    let mut assignments = vec![0usize; n];
    let mut best_distances = vec![f32::INFINITY; n];
    let mut cluster_counts = vec![0usize; k];
    let mut cluster_sums = vec![0.0f32; k * dim];

    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // ─────────────────────────────────────────────────────────────
        // 1. ASSIGN STEP
        // For each centroid, compute Euclidean distances across all data points
        // using the Phase 3 batched distance function.
        // ─────────────────────────────────────────────────────────────
        best_distances.fill(f32::INFINITY);

        for (c_idx, c_vec) in (0..k).map(|c| (c, centroids.get(c))) {
            let dists = batch_euclidean_distance(c_vec, data);
            for i in 0..n {
                if dists[i] < best_distances[i] {
                    best_distances[i] = dists[i];
                    assignments[i] = c_idx;
                }
            }
        }

        // ─────────────────────────────────────────────────────────────
        // 2. UPDATE STEP
        // Recompute each centroid as the arithmetic mean of its assigned points.
        // ─────────────────────────────────────────────────────────────
        cluster_counts.fill(0);
        cluster_sums.fill(0.0);

        for (i, &c_idx) in assignments.iter().enumerate() {
            cluster_counts[c_idx] += 1;
            let pt = data.get(i);
            let offset = c_idx * dim;
            for d in 0..dim {
                cluster_sums[offset + d] += pt[d];
            }
        }

        // Handle normal updates and empty cluster edge cases
        for (c, &count) in cluster_counts.iter().enumerate().take(k) {
            let offset = c * dim;

            if count > 0 {
                let inv = 1.0 / (count as f32);
                for d in 0..dim {
                    cluster_sums[offset + d] *= inv;
                }
            } else {
                // Empty cluster recovery: reinitialize empty centroid to a random data point
                let rand_idx = rng.gen_range(0..n);
                let rand_pt = data.get(rand_idx);
                cluster_sums[offset..offset + dim].copy_from_slice(rand_pt);
            }
        }

        // ─────────────────────────────────────────────────────────────
        // 3. CONVERGENCE CHECK
        // Stop early if the sum of all centroid shifts is below tolerance.
        // ─────────────────────────────────────────────────────────────
        let mut total_shift = 0.0f32;
        for c in 0..k {
            let old_centroid = centroids.get(c);
            let new_centroid = &cluster_sums[c * dim..(c + 1) * dim];
            total_shift += euclidean_distance(old_centroid, new_centroid);
        }

        // Update centroids with new coordinates
        centroids.clear();
        for c in 0..k {
            let offset = c * dim;
            centroids.push(&cluster_sums[offset..offset + dim]);
        }

        if total_shift < config.tolerance {
            break;
        }
    }

    KMeansResult {
        centroids,
        assignments,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Tiny hand-verifiable test: 4 points forming 2 obvious clusters.
    /// [1,1], [1,2] near each other, and [9,9], [9,8] near each other.
    #[test]
    fn test_two_obvious_clusters_grouping() {
        let mut data = VectorBatch::new(2);
        data.push(&[1.0, 1.0]);
        data.push(&[1.0, 2.0]);
        data.push(&[9.0, 9.0]);
        data.push(&[9.0, 8.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 100,
            tolerance: 1e-4,
        };

        let result = kmeans(&data, &config, 42);

        // Grouping assertion: [1,1] and [1,2] must share one cluster,
        // and [9,9] and [9,8] must share the other cluster.
        assert_eq!(
            result.assignments[0], result.assignments[1],
            "Points [1,1] and [1,2] must belong to the same cluster"
        );
        assert_eq!(
            result.assignments[2], result.assignments[3],
            "Points [9,9] and [9,8] must belong to the same cluster"
        );
        assert_ne!(
            result.assignments[0], result.assignments[2],
            "Cluster 1 and Cluster 2 must be distinct"
        );
    }

    /// Test 2: Centroids after convergence are close to hand-computed expected means.
    /// Cluster 1: mean([1,1], [1,2]) = [1.0, 1.5]
    /// Cluster 2: mean([9,9], [9,8]) = [9.0, 8.5]
    #[test]
    fn test_two_obvious_clusters_centroid_values() {
        let mut data = VectorBatch::new(2);
        data.push(&[1.0, 1.0]);
        data.push(&[1.0, 2.0]);
        data.push(&[9.0, 9.0]);
        data.push(&[9.0, 8.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 100,
            tolerance: 1e-4,
        };

        let result = kmeans(&data, &config, 42);
        assert_eq!(result.centroids.len(), 2);

        let c0 = result.centroids.get(0);
        let c1 = result.centroids.get(1);

        // Identify which centroid corresponds to which cluster based on x-coordinate
        let (c_low, c_high) = if c0[0] < c1[0] { (c0, c1) } else { (c1, c0) };

        println!("Hand-verification for Test 2:");
        println!("  Expected low cluster centroid:  [1.0, 1.5]");
        println!("  Actual low cluster centroid:    {:?}", c_low);
        println!("  Expected high cluster centroid: [9.0, 8.5]");
        println!("  Actual high cluster centroid:   {:?}", c_high);

        assert!(
            (c_low[0] - 1.0).abs() < 1e-4 && (c_low[1] - 1.5).abs() < 1e-4,
            "Low centroid expected [1.0, 1.5], got {:?}",
            c_low
        );
        assert!(
            (c_high[0] - 9.0).abs() < 1e-4 && (c_high[1] - 8.5).abs() < 1e-4,
            "High centroid expected [9.0, 8.5], got {:?}",
            c_high
        );
    }

    /// Test 3: k-means++ init produces k DISTINCT centroids (not duplicates).
    #[test]
    fn test_kmeans_plus_plus_distinct_centroids() {
        let mut data = VectorBatch::new(2);
        // Add 100 points clumped around (0,0), (50,50), and (100,100)
        for _ in 0..30 {
            data.push(&[0.1, -0.1]);
        }
        for _ in 0..30 {
            data.push(&[50.2, 49.8]);
        }
        for _ in 0..40 {
            data.push(&[100.1, 99.9]);
        }

        let mut rng = StdRng::seed_from_u64(12345);
        let centroids = kmeans_plus_plus_init(&data, 3, &mut rng);

        assert_eq!(centroids.len(), 3);
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dist = euclidean_distance(centroids.get(i), centroids.get(j));
                assert!(
                    dist > 1.0,
                    "Centroids {} and {} must be distinct, got dist={}",
                    i,
                    j,
                    dist
                );
            }
        }
    }

    /// Test 4: Reproducibility: same seed produces IDENTICAL results twice.
    #[test]
    fn test_reproducibility_same_seed() {
        let mut data = VectorBatch::new(4);
        for i in 0..50 {
            let v = i as f32;
            data.push(&[v, v * 0.5, -v, v * 2.0]);
        }

        let config = KMeansConfig {
            k: 4,
            max_iterations: 50,
            tolerance: 1e-4,
        };

        let seed = 987654321;
        let res1 = kmeans(&data, &config, seed);
        let res2 = kmeans(&data, &config, seed);

        assert_eq!(res1.iterations, res2.iterations);
        assert_eq!(res1.assignments, res2.assignments);
        assert_eq!(res1.centroids.data(), res2.centroids.data());
    }

    /// Test 5: Empty cluster handling on adversarial input.
    /// k=5 with only 2 distinct point positions.
    #[test]
    fn test_empty_cluster_handling_no_nan() {
        let mut data = VectorBatch::new(2);
        for _ in 0..10 {
            data.push(&[0.0, 0.0]);
        }
        for _ in 0..10 {
            data.push(&[100.0, 100.0]);
        }

        let config = KMeansConfig {
            k: 5,
            max_iterations: 30,
            tolerance: 1e-4,
        };

        let result = kmeans(&data, &config, 99);

        assert_eq!(result.centroids.len(), 5);
        for (i, val) in result.centroids.data().iter().enumerate() {
            assert!(
                !val.is_nan(),
                "Centroid value at index {} must not be NaN",
                i
            );
            assert!(
                !val.is_infinite(),
                "Centroid value at index {} must not be infinite",
                i
            );
        }
    }

    /// Test 6: Convergence speed: returns in fewer than max_iterations.
    #[test]
    fn test_fast_convergence_on_separated_data() {
        let mut data = VectorBatch::new(2);
        data.push(&[1.0, 1.0]);
        data.push(&[1.0, 2.0]);
        data.push(&[10.0, 10.0]);
        data.push(&[10.0, 11.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 100,
            tolerance: 1e-4,
        };

        let result = kmeans(&data, &config, 42);

        println!("Test 6 iterations to converge: {}", result.iterations);
        assert!(
            result.iterations < 10,
            "Expected convergence in <10 iterations, took {}",
            result.iterations
        );
    }

    /// Test 7: Larger test on 1,000 vectors of dimension 128 with k=10.
    #[test]
    fn test_large_sift_scale_clustering() {
        let n = 1000;
        let dim = 128;
        let k = 10;

        let mut data = VectorBatch::new(dim);
        let mut rng = StdRng::seed_from_u64(777);

        for _ in 0..n {
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                vec.push(rng.gen_range(-10.0..10.0));
            }
            data.push(&vec);
        }

        let config = KMeansConfig {
            k,
            max_iterations: 20,
            tolerance: 1e-3,
        };

        let result = kmeans(&data, &config, 12345);

        assert_eq!(result.centroids.len(), k);
        assert_eq!(result.centroids.dim(), dim);
        assert_eq!(result.assignments.len(), n);

        for val in result.centroids.data() {
            assert!(!val.is_nan());
            assert!(!val.is_infinite());
        }

        for &cluster_id in &result.assignments {
            assert!(cluster_id < k);
        }
    }
}
