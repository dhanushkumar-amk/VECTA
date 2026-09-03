//! Inverted File Index (IVF) data structure, training, and insertion.
//!
//! Provides coarse quantizer Voronoi partitioning:
//! - A set of `k` centroid vectors ([`VectorBatch`]).
//! - An inverted list for each cluster, stored as an internal [`FlatIndex`].
//!
//! # Architecture & Lifecycle
//! IVF enforces a two-phase lifecycle mirroring FAISS (`IndexIVFFlat`):
//! 1. **`train()`**: Learns coarse cluster centroids via k-means clustering on a
//!    representative data sample. Does not insert vectors.
//! 2. **`add()` / `add_batch()`**: Routes vectors into their nearest centroid's
//!    inverted list. Calling `add()` prior to `train()` returns an error.

use crate::core::batch::{batch_euclidean_distance, VectorBatch};
use crate::core::flat_index::{FlatIndex, Metric};
use crate::core::kmeans::{kmeans, KMeansConfig};
use crate::core::topk::{top_k_smallest, ScoredId};
use crate::core::vector::euclidean_distance;

/// An Inverted File (IVF) index structure.
///
/// Partitions high-dimensional vector space into Voronoi cells centered
/// around `k` learned centroids. Each centroid maintains an inverted list
/// (implemented as a [`FlatIndex`]) containing the vectors assigned to that cluster.
#[derive(Debug, Clone)]
pub struct IVFIndex {
    /// Centroid coordinates for each cluster (shape: `num_clusters x dim`).
    pub centroids: VectorBatch,
    /// Inverted lists: one `FlatIndex` per cluster.
    /// `inverted_lists[i]` stores all vectors assigned to centroid `i`.
    pub inverted_lists: Vec<FlatIndex>,
    /// Dimensionality of vectors in this index.
    pub dim: usize,
    /// Distance/similarity metric for distance evaluations.
    pub metric: Metric,
    /// Whether centroids have been trained via k-means clustering.
    pub is_trained: bool,
}

impl IVFIndex {
    /// Create a new, untrained IVF index with `num_clusters` empty inverted lists.
    ///
    /// # Arguments
    /// * `dim` - Dimensionality of vectors.
    /// * `num_clusters` - Number of Voronoi partitions (centroids / inverted lists).
    /// * `metric` - Distance/similarity metric for search queries.
    pub fn new(dim: usize, num_clusters: usize, metric: Metric) -> Self {
        let mut inverted_lists = Vec::with_capacity(num_clusters);
        for _ in 0..num_clusters {
            inverted_lists.push(FlatIndex::new(dim, metric));
        }

        Self {
            centroids: VectorBatch::new(dim),
            inverted_lists,
            dim,
            metric,
            is_trained: false,
        }
    }

    /// Return the number of clusters (inverted lists).
    #[inline]
    pub fn num_clusters(&self) -> usize {
        self.inverted_lists.len()
    }

    /// Return the total vector count across ALL inverted lists.
    pub fn len(&self) -> usize {
        self.inverted_lists.iter().map(|list| list.len()).sum()
    }

    /// Return `true` if the index contains no vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of vectors in each inverted list, in centroid order.
    ///
    /// Diagnostic helper for detecting cluster imbalance or empty clusters.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.inverted_lists.iter().map(|list| list.len()).collect()
    }

    /// Return vector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return distance metric.
    #[inline]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Return whether index centroids have been trained.
    #[inline]
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Train the coarse quantizer centroids on a representative sample of data.
    ///
    /// Runs Lloyd's k-means clustering with k-means++ seeding.
    ///
    /// # Panics
    /// - If `training_data.dim != self.dim`.
    /// - If `config.k != self.num_clusters()`.
    ///
    /// # Lifecycle Note
    /// In accordance with the FAISS `train`-then-`add` lifecycle, training only learns
    /// and populates cluster centroids. It does NOT automatically add vectors from
    /// `training_data` into the inverted lists. The caller must explicitly invoke
    /// `add` or `add_batch` to index data points.
    pub fn train(&mut self, training_data: &VectorBatch, config: &KMeansConfig, seed: u64) {
        assert_eq!(
            training_data.dim, self.dim,
            "IVFIndex::train: training_data dimension {} != index dimension {}",
            training_data.dim, self.dim
        );
        assert_eq!(
            config.k,
            self.inverted_lists.len(),
            "IVFIndex::train: config.k ({}) != num_clusters ({})",
            config.k,
            self.inverted_lists.len()
        );

        let result = kmeans(training_data, config, seed);
        self.centroids = result.centroids;
        self.is_trained = true;
    }

    /// Find the index of the closest centroid to `vector` under Euclidean distance.
    ///
    /// # Panics
    /// - If `!self.is_trained` or `self.centroids.is_empty()`.
    /// - If `vector.len() != self.dim`.
    pub fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        assert!(
            self.is_trained && !self.centroids.is_empty(),
            "IVFIndex::find_nearest_centroid: index is not trained"
        );
        assert_eq!(
            vector.len(),
            self.dim,
            "IVFIndex::find_nearest_centroid: expected dim {}, got {}",
            self.dim,
            vector.len()
        );

        let mut best_idx = 0;
        let mut min_dist = f32::INFINITY;

        for c in 0..self.centroids.len() {
            let dist = euclidean_distance(vector, self.centroids.get(c));
            if dist < min_dist {
                min_dist = dist;
                best_idx = c;
            }
        }

        best_idx
    }

    /// Insert a single vector with its external ID into the index.
    ///
    /// Routes the vector to the inverted list corresponding to its nearest centroid.
    ///
    /// # Errors
    /// Returns `Err` if the index has not been trained yet. Calling `add` on an untrained
    /// index is a recoverable operational error, hence returning `Result` rather than panicking.
    ///
    /// # Panics
    /// Panics if `vector.len() != self.dim` (programming error).
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), String> {
        if !self.is_trained {
            return Err("IVFIndex must be trained before adding vectors".to_string());
        }

        assert_eq!(
            vector.len(),
            self.dim,
            "IVFIndex::add: expected dim {}, got {}",
            self.dim,
            vector.len()
        );

        let nearest_idx = self.find_nearest_centroid(vector);
        self.inverted_lists[nearest_idx].add(id, vector);
        Ok(())
    }

    /// Bulk-insert vectors with their external IDs into the index.
    ///
    /// Routes each vector to the inverted list of its nearest centroid.
    ///
    /// # Errors
    /// - Returns `Err` if `!self.is_trained`.
    /// - Returns `Err` if `ids.len() != vectors.len()`.
    ///
    /// # Panics
    /// Panics if `vectors.dim != self.dim`.
    pub fn add_batch(&mut self, ids: &[u64], vectors: &VectorBatch) -> Result<(), String> {
        if !self.is_trained {
            return Err("IVFIndex must be trained before adding vectors".to_string());
        }

        assert_eq!(
            vectors.dim, self.dim,
            "IVFIndex::add_batch: expected dim {}, got {}",
            self.dim, vectors.dim
        );

        if ids.len() != vectors.len() {
            return Err(format!(
                "IVFIndex::add_batch: ids count ({}) != vectors count ({})",
                ids.len(),
                vectors.len()
            ));
        }

        for (i, &id) in ids.iter().enumerate() {
            let vec = vectors.get(i);
            let nearest_idx = self.find_nearest_centroid(vec);
            self.inverted_lists[nearest_idx].add(id, vec);
        }

        Ok(())
    }

    /// Coarse search: compare query against all centroids to select the `nprobe` nearest clusters.
    ///
    /// # Arguments
    /// * `query` - High-dimensional query vector.
    /// * `nprobe` - Number of nearest centroids/clusters to return for fine scanning.
    ///
    /// # Panics
    /// - If `query.len() != self.dim`.
    /// - If `!self.is_trained` (searching an untrained index is a programming bug).
    ///
    /// # Metric Selection Note
    /// Centroid comparison ALWAYS utilizes Euclidean (L2) distance, regardless of whether
    /// `self.metric` is Cosine or Dot Product. This preserves geometric consistency with the
    /// Voronoi partitioning learned during k-means clustering. In Phase 13, vectors within
    /// the selected inverted lists are evaluated using `self.metric`.
    pub fn find_nearest_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        assert!(
            self.is_trained,
            "IVFIndex::find_nearest_clusters: cannot search an untrained index"
        );
        assert_eq!(
            query.len(),
            self.dim,
            "IVFIndex::find_nearest_clusters: query dimension {} != index dimension {}",
            query.len(),
            self.dim
        );

        if nprobe == 0 || self.centroids.is_empty() {
            return Vec::new();
        }

        let k = nprobe.min(self.centroids.len());

        // Compute Euclidean distance from query to every centroid in parallel using Phase 3 batch engine
        let dists = batch_euclidean_distance(query, &self.centroids);

        // Map distances to candidate ScoredIds with centroid index
        let candidates: Vec<ScoredId> = dists
            .into_iter()
            .enumerate()
            .map(|(idx, dist)| ScoredId {
                id: idx as u64,
                score: dist,
            })
            .collect();

        // Select the k nearest centroids, sorted ascending (nearest-first)
        let top_clusters = top_k_smallest(&candidates, k);
        top_clusters.into_iter().map(|s| s.id as usize).collect()
    }

    /// Return the total number of vectors across the `nprobe` nearest clusters to `query`.
    ///
    /// Diagnostic helper for inspecting the speed/recall trade-off surface.
    pub fn nprobe_coverage(&self, query: &[f32], nprobe: usize) -> usize {
        let selected = self.find_nearest_clusters(query, nprobe);
        selected
            .into_iter()
            .map(|idx| self.inverted_lists[idx].len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Test 1: Calling add() before train() returns Err, does not panic, does not corrupt state.
    #[test]
    fn test_add_before_train_returns_err() {
        let mut index = IVFIndex::new(3, 4, Metric::Euclidean);
        assert!(!index.is_trained());

        let res = index.add(1, &[1.0, 2.0, 3.0]);
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            "IVFIndex must be trained before adding vectors"
        );
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        let mut batch = VectorBatch::new(3);
        batch.push(&[1.0, 2.0, 3.0]);
        let batch_res = index.add_batch(&[1], &batch);
        assert!(batch_res.is_err());
        assert_eq!(index.len(), 0);
    }

    /// Test 2: train() with mismatched training_data dimension panics with clear message.
    #[test]
    #[should_panic(expected = "training_data dimension 4 != index dimension 3")]
    fn test_train_mismatched_dimension_panics() {
        let mut index = IVFIndex::new(3, 2, Metric::Euclidean);
        let mut wrong_dim_data = VectorBatch::new(4);
        wrong_dim_data.push(&[1.0, 2.0, 3.0, 4.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 10,
            tolerance: 1e-4,
        };
        index.train(&wrong_dim_data, &config, 42);
    }

    /// Test 3: train() with config.k != num_clusters panics with clear message.
    #[test]
    #[should_panic(expected = "config.k (5) != num_clusters (2)")]
    fn test_train_mismatched_k_panics() {
        let mut index = IVFIndex::new(2, 2, Metric::Euclidean);
        let mut data = VectorBatch::new(2);
        data.push(&[1.0, 1.0]);
        data.push(&[2.0, 2.0]);

        let config = KMeansConfig {
            k: 5, // Mismatch with num_clusters=2
            max_iterations: 10,
            tolerance: 1e-4,
        };
        index.train(&data, &config, 42);
    }

    /// Test 4: After train(), is_trained is true, centroids has correct shape, and inverted lists remain empty.
    #[test]
    fn test_train_sets_trained_and_centroids_without_inserting() {
        let mut index = IVFIndex::new(2, 2, Metric::Euclidean);
        let mut data = VectorBatch::new(2);
        data.push(&[1.0, 1.0]);
        data.push(&[1.0, 2.0]);
        data.push(&[9.0, 9.0]);
        data.push(&[9.0, 8.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 50,
            tolerance: 1e-4,
        };
        index.train(&data, &config, 42);

        assert!(index.is_trained());
        assert_eq!(index.centroids.len(), 2);
        assert_eq!(index.centroids.dim(), 2);
        // Important: train does NOT automatically index the training data!
        assert_eq!(index.len(), 0);
        assert_eq!(index.cluster_sizes(), vec![0, 0]);
    }

    /// Test 5: Hand-verifiable test:
    /// Train on 4 points forming 2 clusters:
    /// Cluster A: [1,1], [1,2]
    /// Cluster B: [9,9], [9,8]
    /// Add a 5th point [1.2, 1.3] (near Cluster A) and 6th point [9.1, 8.9] (near Cluster B).
    #[test]
    fn test_hand_verified_two_clusters_routing() {
        let mut index = IVFIndex::new(2, 2, Metric::Euclidean);
        let mut train_data = VectorBatch::new(2);
        train_data.push(&[1.0, 1.0]);
        train_data.push(&[1.0, 2.0]);
        train_data.push(&[9.0, 9.0]);
        train_data.push(&[9.0, 8.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 100,
            tolerance: 1e-4,
        };
        index.train(&train_data, &config, 42);

        // Identify which centroid is near [1, 1.5] (Cluster A) vs [9, 8.5] (Cluster B)
        let c0 = index.centroids.get(0);
        let cluster_a_idx = if c0[0] < 5.0 { 0 } else { 1 };
        let cluster_b_idx = 1 - cluster_a_idx;

        // 5th point: [1.2, 1.3] (obviously Cluster A)
        let pt5 = [1.2, 1.3];
        index
            .add(500, &pt5)
            .expect("add should succeed after train");

        println!("Hand-verified Test 5 routing:");
        println!("  Cluster A index: {}", cluster_a_idx);
        println!("  Cluster B index: {}", cluster_b_idx);
        println!(
            "  Point 5 [1.2, 1.3] routed to inverted list: {}",
            cluster_a_idx
        );

        // 6th point: [9.1, 8.9] (obviously Cluster B)
        let pt6 = [9.1, 8.9];
        index
            .add(600, &pt6)
            .expect("add should succeed after train");

        assert_eq!(index.len(), 2);
        let sizes = index.cluster_sizes();
        assert_eq!(
            sizes[cluster_a_idx], 1,
            "Cluster A must contain exactly 1 vector"
        );
        assert_eq!(
            sizes[cluster_b_idx], 1,
            "Cluster B must contain exactly 1 vector"
        );

        // Verify vectors stored in the respective inverted lists
        assert_eq!(index.inverted_lists[cluster_a_idx].ids, vec![500]);
        assert_eq!(index.inverted_lists[cluster_b_idx].ids, vec![600]);
    }

    /// Test 6: add_batch() correctly distributes a batch across multiple inverted lists.
    #[test]
    fn test_add_batch_distribution_across_clusters() {
        let mut index = IVFIndex::new(2, 3, Metric::Euclidean);

        // Train 3 well-separated clusters
        let mut train_data = VectorBatch::new(2);
        for _ in 0..10 {
            train_data.push(&[0.0, 0.0]);
        }
        for _ in 0..10 {
            train_data.push(&[50.0, 50.0]);
        }
        for _ in 0..10 {
            train_data.push(&[100.0, 100.0]);
        }

        let config = KMeansConfig {
            k: 3,
            max_iterations: 30,
            tolerance: 1e-4,
        };
        index.train(&train_data, &config, 777);

        // Construct a batch with 3 vectors clearly matching each cluster
        let mut batch = VectorBatch::new(2);
        batch.push(&[0.1, -0.1]); // Near cluster ~ (0,0)
        batch.push(&[50.2, 49.8]); // Near cluster ~ (50,50)
        batch.push(&[100.1, 99.9]); // Near cluster ~ (100,100)

        index
            .add_batch(&[1, 2, 3], &batch)
            .expect("add_batch should succeed");

        assert_eq!(index.len(), 3);
        let sizes = index.cluster_sizes();
        assert_eq!(sizes.len(), 3);
        // Each cluster should have received exactly 1 point
        assert_eq!(sizes, vec![1, 1, 1]);
    }

    /// Test 7: Large-scale test: train on 1,000 random 128-dim vectors with k=10,
    /// then add all 1,000 via add_batch. Confirm len, total sum, and inspect distribution.
    #[test]
    fn test_large_scale_1000_vectors_balanced_distribution() {
        let n = 1000;
        let dim = 128;
        let k = 10;

        let mut index = IVFIndex::new(dim, k, Metric::Euclidean);
        let mut data = VectorBatch::new(dim);
        let mut rng = StdRng::seed_from_u64(9999);

        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-10.0..10.0));
            }
            data.push(&v);
        }

        let config = KMeansConfig {
            k,
            max_iterations: 25,
            tolerance: 1e-3,
        };
        index.train(&data, &config, 42);

        let ids: Vec<u64> = (0..n as u64).collect();
        index
            .add_batch(&ids, &data)
            .expect("add_batch should succeed");

        assert_eq!(index.len(), n);
        let sizes = index.cluster_sizes();
        let total_sum: usize = sizes.iter().sum();
        assert_eq!(total_sum, n);

        println!("\nTest 7: 1,000 vectors clustered across k=10 inverted lists:");
        for (c, &size) in sizes.iter().enumerate() {
            println!(
                "  Inverted List {}: {} vectors ({:.1}%)",
                c,
                size,
                (size as f64 / n as f64) * 100.0
            );
            if size == 0 {
                println!("  [WARNING] Cluster {} is empty!", c);
            }
        }

        // Verify every cluster has a reasonable share (not empty or degenerate)
        let min_cluster = *sizes.iter().min().unwrap();
        let max_cluster = *sizes.iter().max().unwrap();
        println!(
            "  Min cluster size: {}, Max cluster size: {}",
            min_cluster, max_cluster
        );
        assert!(
            min_cluster > 0,
            "No cluster should be completely empty in uniform random distribution"
        );
    }

    /// Phase 12 Test 1: Hand-verifiable coarse search with nprobe=1.
    /// Query clearly near cluster A returns cluster A's centroid index.
    #[test]
    fn test_coarse_search_hand_verified_nprobe_1() {
        let mut index = IVFIndex::new(2, 2, Metric::Euclidean);
        let mut train_data = VectorBatch::new(2);
        train_data.push(&[1.0, 1.0]);
        train_data.push(&[1.0, 2.0]);
        train_data.push(&[9.0, 9.0]);
        train_data.push(&[9.0, 8.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 100,
            tolerance: 1e-4,
        };
        index.train(&train_data, &config, 42);

        let c0 = index.centroids.get(0);
        let cluster_a_idx = if c0[0] < 5.0 { 0 } else { 1 };
        let cluster_b_idx = 1 - cluster_a_idx;

        // Query point: [1.1, 1.4] is obviously closest to Cluster A
        let query_a = [1.1, 1.4];
        let nearest_1 = index.find_nearest_clusters(&query_a, 1);

        println!("Phase 12 Test 1 results:");
        println!("  Cluster A index: {}", cluster_a_idx);
        println!("  Cluster B index: {}", cluster_b_idx);
        println!("  Query [1.1, 1.4] returned centroid: {:?}", nearest_1);

        assert_eq!(nearest_1.len(), 1);
        assert_eq!(nearest_1[0], cluster_a_idx);
    }

    /// Phase 12 Test 2: Hand-verifiable coarse search with nprobe=2.
    /// Returns both clusters in nearest-first sorted order.
    #[test]
    fn test_coarse_search_hand_verified_nprobe_2() {
        let mut index = IVFIndex::new(2, 2, Metric::Euclidean);
        let mut train_data = VectorBatch::new(2);
        train_data.push(&[1.0, 1.0]);
        train_data.push(&[1.0, 2.0]);
        train_data.push(&[9.0, 9.0]);
        train_data.push(&[9.0, 8.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 100,
            tolerance: 1e-4,
        };
        index.train(&train_data, &config, 42);

        let c0 = index.centroids.get(0);
        let cluster_a_idx = if c0[0] < 5.0 { 0 } else { 1 };
        let cluster_b_idx = 1 - cluster_a_idx;

        // Query point closer to A than B
        let query_a = [1.1, 1.4];
        let nearest_2 = index.find_nearest_clusters(&query_a, 2);

        println!("Phase 12 Test 2 results (nprobe=2):");
        println!("  Returned cluster order: {:?}", nearest_2);

        assert_eq!(nearest_2.len(), 2);
        assert_eq!(nearest_2[0], cluster_a_idx);
        assert_eq!(nearest_2[1], cluster_b_idx);
    }

    /// Phase 12 Test 3: nprobe >= num_clusters returns all cluster indices.
    #[test]
    fn test_coarse_search_nprobe_ge_num_clusters() {
        let mut index = IVFIndex::new(2, 2, Metric::Euclidean);
        let mut train_data = VectorBatch::new(2);
        train_data.push(&[1.0, 1.0]);
        train_data.push(&[9.0, 9.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 10,
            tolerance: 1e-4,
        };
        index.train(&train_data, &config, 42);

        // Request nprobe=100 on k=2 index
        let res = index.find_nearest_clusters(&[1.0, 1.0], 100);
        assert_eq!(res.len(), 2);
        assert!(res.contains(&0));
        assert!(res.contains(&1));
    }

    /// Phase 12 Test 4: nprobe == 0 returns empty Vec.
    #[test]
    fn test_coarse_search_nprobe_zero() {
        let mut index = IVFIndex::new(2, 2, Metric::Euclidean);
        let mut train_data = VectorBatch::new(2);
        train_data.push(&[1.0, 1.0]);
        train_data.push(&[9.0, 9.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 10,
            tolerance: 1e-4,
        };
        index.train(&train_data, &config, 42);

        let res = index.find_nearest_clusters(&[1.0, 1.0], 0);
        assert!(res.is_empty());
    }

    /// Phase 12 Test 5: Calling find_nearest_clusters before train() panics with clear message.
    #[test]
    #[should_panic(expected = "cannot search an untrained index")]
    fn test_find_nearest_clusters_untrained_panics() {
        let index = IVFIndex::new(2, 2, Metric::Euclidean);
        let _ = index.find_nearest_clusters(&[1.0, 1.0], 1);
    }

    /// Phase 12 Test 6: Wrong query dimension panics with clear message.
    #[test]
    #[should_panic(expected = "query dimension 3 != index dimension 2")]
    fn test_find_nearest_clusters_wrong_dim_panics() {
        let mut index = IVFIndex::new(2, 2, Metric::Euclidean);
        let mut train_data = VectorBatch::new(2);
        train_data.push(&[1.0, 1.0]);
        train_data.push(&[9.0, 9.0]);

        let config = KMeansConfig {
            k: 2,
            max_iterations: 10,
            tolerance: 1e-4,
        };
        index.train(&train_data, &config, 42);

        let _ = index.find_nearest_clusters(&[1.0, 2.0, 3.0], 1);
    }

    /// Phase 12 Test 7: nprobe_coverage on 1,000 vectors / k=10:
    /// Confirm monotonic increase (1 -> 3 -> 5 -> 10) and full coverage at nprobe=10.
    #[test]
    fn test_nprobe_coverage_monotonicity() {
        let n = 1000;
        let dim = 128;
        let k = 10;

        let mut index = IVFIndex::new(dim, k, Metric::Euclidean);
        let mut data = VectorBatch::new(dim);
        let mut rng = StdRng::seed_from_u64(12345);

        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-10.0..10.0));
            }
            data.push(&v);
        }

        let config = KMeansConfig {
            k,
            max_iterations: 20,
            tolerance: 1e-3,
        };
        index.train(&data, &config, 99);

        let ids: Vec<u64> = (0..n as u64).collect();
        index.add_batch(&ids, &data).unwrap();

        // Sample query
        let query = data.get(0);

        let cov1 = index.nprobe_coverage(query, 1);
        let cov3 = index.nprobe_coverage(query, 3);
        let cov5 = index.nprobe_coverage(query, 5);
        let cov10 = index.nprobe_coverage(query, 10);

        println!(
            "\nPhase 12 Test 7: nprobe coverage curve (N={}, k={}):",
            n, k
        );
        println!(
            "  nprobe=1:   {:>4} vectors ({:>5.1}%)",
            cov1,
            (cov1 as f64 / n as f64) * 100.0
        );
        println!(
            "  nprobe=3:   {:>4} vectors ({:>5.1}%)",
            cov3,
            (cov3 as f64 / n as f64) * 100.0
        );
        println!(
            "  nprobe=5:   {:>4} vectors ({:>5.1}%)",
            cov5,
            (cov5 as f64 / n as f64) * 100.0
        );
        println!(
            "  nprobe=10:  {:>4} vectors ({:>5.1}%)",
            cov10,
            (cov10 as f64 / n as f64) * 100.0
        );

        // Monotonic non-decreasing check
        assert!(
            cov1 <= cov3,
            "Coverage must not decrease: cov1={} > cov3={}",
            cov1,
            cov3
        );
        assert!(
            cov3 <= cov5,
            "Coverage must not decrease: cov3={} > cov5={}",
            cov3,
            cov5
        );
        assert!(
            cov5 <= cov10,
            "Coverage must not decrease: cov5={} > cov10={}",
            cov5,
            cov10
        );

        // Full coverage when nprobe == k
        assert_eq!(cov10, n, "nprobe=k must cover 100% of vectors");
    }
}
