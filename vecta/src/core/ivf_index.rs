//! Inverted File Index (IVF) data structure.
//!
//! Provides the core layout for coarse quantizer Voronoi partitioning:
//! - A set of `k` centroid vectors ([`VectorBatch`]).
//! - An inverted list for each cluster, stored as an internal [`FlatIndex`].
//!
//! # Architecture
//! IVF sits alongside [`FlatIndex`], not replacing it. Each inverted list
//! is an individual `FlatIndex`, completely reusing contiguous batch storage,
//! ID tracking, and metric evaluation without code duplication.

use crate::core::batch::VectorBatch;
use crate::core::flat_index::{FlatIndex, Metric};

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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: IVFIndex::new produces exactly 10 empty inverted lists, each with correct dim.
    #[test]
    fn test_new_produces_correct_inverted_lists() {
        let dim = 128;
        let num_clusters = 10;
        let index = IVFIndex::new(dim, num_clusters, Metric::Euclidean);

        assert_eq!(index.inverted_lists.len(), num_clusters);
        for list in &index.inverted_lists {
            assert_eq!(list.dim(), dim);
            assert_eq!(list.len(), 0);
            assert!(list.is_empty());
            assert_eq!(list.metric, Metric::Euclidean);
        }
        assert_eq!(index.centroids.len(), 0);
        assert_eq!(index.centroids.dim(), dim);
        assert!(!index.is_trained());
    }

    /// Test 2: num_clusters() returns 10 for the above.
    #[test]
    fn test_num_clusters_accessor() {
        let index = IVFIndex::new(128, 10, Metric::Euclidean);
        assert_eq!(index.num_clusters(), 10);

        let index_small = IVFIndex::new(32, 4, Metric::Cosine);
        assert_eq!(index_small.num_clusters(), 4);
    }

    /// Test 3: len() returns 0 for a freshly-created (untrained, empty) index.
    #[test]
    fn test_len_empty_index() {
        let index = IVFIndex::new(64, 8, Metric::DotProduct);
        assert_eq!(index.len(), 0);
    }

    /// Test 4: is_empty() returns true for a freshly-created index.
    #[test]
    fn test_is_empty_fresh_index() {
        let index = IVFIndex::new(64, 8, Metric::DotProduct);
        assert!(index.is_empty());
    }

    /// Test 5: cluster_sizes() returns a Vec of zeros for a freshly-created index.
    #[test]
    fn test_cluster_sizes_zeros_initially() {
        let num_clusters = 10;
        let index = IVFIndex::new(128, num_clusters, Metric::Euclidean);
        let sizes = index.cluster_sizes();

        assert_eq!(sizes.len(), num_clusters);
        assert_eq!(sizes, vec![0; num_clusters]);
    }

    /// Test 6: Manually construct an IVFIndex, directly push vectors into specific
    /// inverted lists, and confirm len() and cluster_sizes() correctly reflect the data.
    #[test]
    fn test_manual_vector_insertion_and_cluster_sizes() {
        let dim = 3;
        let num_clusters = 4;
        let mut index = IVFIndex::new(dim, num_clusters, Metric::Euclidean);

        // Add 2 vectors to cluster 0
        index.inverted_lists[0].add(101, &[1.0, 2.0, 3.0]);
        index.inverted_lists[0].add(102, &[1.1, 2.1, 3.1]);

        // Add 0 vectors to cluster 1 (remains empty)

        // Add 3 vectors to cluster 2
        index.inverted_lists[2].add(201, &[5.0, 5.0, 5.0]);
        index.inverted_lists[2].add(202, &[5.1, 5.2, 5.3]);
        index.inverted_lists[2].add(203, &[5.2, 5.1, 5.0]);

        // Add 1 vector to cluster 3
        index.inverted_lists[3].add(301, &[9.0, 9.0, 9.0]);

        // Assertions
        assert_eq!(index.len(), 6, "Total vectors should be 2 + 0 + 3 + 1 = 6");
        assert!(!index.is_empty(), "Index is not empty");

        let expected_sizes = vec![2, 0, 3, 1];
        assert_eq!(index.cluster_sizes(), expected_sizes);
    }
}
