//! Inverted File with Product Quantization (IndexIVFPQ) index structure.
//!
//! Combines coarse Voronoi cell partitioning (IVF) with Product Quantization (PQ)
//! compression:
//! - Coarse quantizer: A set of `num_clusters` full-precision centroids ([`VectorBatch`]).
//! - Fine storage: Inverted lists per cluster storing compact `(u64, PQCode)` pairs
//!   rather than full-precision vectors.
//! - Query search: Evaluated using Asymmetric Distance Computation (ADC) lookup tables,
//!   precomputed once per query and evaluated against compressed codes without vector reconstruction.
//!
//! # Metric Support Note
//! PQ codebooks and ADC lookup tables operate in squared Euclidean distance space.
//! In v1, [`IVFPQIndex`] specifically supports [`Metric::Euclidean`]. Cosine or Dot Product
//! metrics are not supported for fine ADC search in v1.

use crate::core::batch::VectorBatch;
use crate::core::flat_index::Metric;
use crate::core::ivf_index::{find_nearest_centroid, find_nearest_clusters};
use crate::core::kmeans::{kmeans, KMeansConfig};
use crate::core::pq::{
    adc_distance, build_adc_table, encode_vector, train_pq, PQCode, PQCodebooks, PQConfig,
};
use crate::core::topk::{top_k_smallest, ScoredId};

/// Inverted File with Product Quantization (IndexIVFPQ) index structure.
///
/// Partitions high-dimensional vector space into Voronoi cells centered around
/// `num_clusters` learned centroids (kept at full precision). Each inverted list stores
/// compact byte codes ([`PQCode`]) paired with external IDs.
#[derive(Debug, Clone)]
pub struct IVFPQIndex {
    /// Centroid coordinates for each cluster at FULL precision (shape: `num_clusters x dim`).
    pub centroids: VectorBatch,
    /// Trained Product Quantization subquantizer codebooks (`None` until trained).
    pub pq_codebooks: Option<PQCodebooks>,
    /// Inverted lists: one list of `(id, compressed_code)` pairs per cluster.
    pub inverted_lists: Vec<Vec<(u64, PQCode)>>,
    /// Dimensionality of vectors in this index.
    pub dim: usize,
    /// Distance metric (hardcoded to [`Metric::Euclidean`] for v1).
    pub metric: Metric,
    /// Whether both coarse centroids and PQ codebooks have been trained.
    pub is_trained: bool,
    /// Configuration for Product Quantization training and codebook dimensions.
    pub pq_config: PQConfig,
}

impl IVFPQIndex {
    /// Create a new, untrained IVFPQIndex.
    ///
    /// # Arguments
    /// * `dim` - Dimensionality of vectors. Must be evenly divisible by `pq_config.m`.
    /// * `num_clusters` - Number of coarse Voronoi clusters (inverted lists).
    /// * `pq_config` - Product Quantization configuration parameters.
    ///
    /// # Errors
    /// Returns `Err(String)` if:
    /// - `dim == 0` or `num_clusters == 0`
    /// - `pq_config.m == 0`
    /// - `dim % pq_config.m != 0` (dim not evenly divisible by m)
    /// - `pq_config.k_per_subvector == 0` or `pq_config.k_per_subvector > 256`
    pub fn new(dim: usize, num_clusters: usize, pq_config: PQConfig) -> Result<Self, String> {
        if dim == 0 {
            return Err("IVFPQIndex::new: dim must be greater than 0".to_string());
        }
        if num_clusters == 0 {
            return Err("IVFPQIndex::new: num_clusters must be greater than 0".to_string());
        }
        if pq_config.m == 0 {
            return Err("IVFPQIndex::new: pq_config.m must be greater than 0".to_string());
        }
        if !dim.is_multiple_of(pq_config.m) {
            return Err(format!(
                "IVFPQIndex::new: dimension {} is not evenly divisible by m={} subvectors",
                dim, pq_config.m
            ));
        }
        if pq_config.k_per_subvector == 0 {
            return Err("IVFPQIndex::new: k_per_subvector must be greater than 0".to_string());
        }
        if pq_config.k_per_subvector > 256 {
            return Err(format!(
                "IVFPQIndex::new: k_per_subvector ({}) cannot exceed 256 for 1-byte encoding",
                pq_config.k_per_subvector
            ));
        }

        let inverted_lists = vec![Vec::new(); num_clusters];

        Ok(Self {
            centroids: VectorBatch::new(dim),
            pq_codebooks: None,
            inverted_lists,
            dim,
            metric: Metric::Euclidean,
            is_trained: false,
            pq_config,
        })
    }

    /// Return the number of coarse clusters (inverted lists).
    #[inline]
    pub fn num_clusters(&self) -> usize {
        self.inverted_lists.len()
    }

    /// Return the total number of vectors across all inverted lists.
    #[inline]
    pub fn len(&self) -> usize {
        self.inverted_lists.iter().map(|list| list.len()).sum()
    }

    /// Return `true` if the index contains zero vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return vector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return whether index centroids and PQ codebooks have been trained.
    #[inline]
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Return the number of vectors in each inverted list, in cluster index order.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.inverted_lists.iter().map(|list| list.len()).collect()
    }

    /// Train coarse cluster centroids and PQ codebooks on a representative sample of data.
    ///
    /// # Two-Stage Training:
    /// 1. Coarse Quantizer: Runs Lloyd's k-means clustering ([`kmeans`]) to learn `num_clusters` centroids.
    /// 2. Product Quantizer: Runs subvector k-means clustering ([`train_pq`]) to learn `m` codebooks.
    ///
    /// Both steps must succeed before `self.is_trained` is marked `true`.
    ///
    /// # Errors
    /// Returns `Err(String)` if:
    /// - `training_data.dim != self.dim`
    /// - `kmeans_config.k != self.num_clusters()`
    /// - `training_data.len() < kmeans_config.k` or `training_data.len() < self.pq_config.k_per_subvector`
    /// - Either training step returns an error.
    pub fn train(
        &mut self,
        training_data: &VectorBatch,
        kmeans_config: &KMeansConfig,
        ivf_seed: u64,
        pq_seed: u64,
    ) -> Result<(), String> {
        if training_data.dim != self.dim {
            return Err(format!(
                "IVFPQIndex::train: training_data dimension {} != index dimension {}",
                training_data.dim, self.dim
            ));
        }
        if kmeans_config.k != self.num_clusters() {
            return Err(format!(
                "IVFPQIndex::train: kmeans_config.k ({}) != num_clusters ({})",
                kmeans_config.k,
                self.num_clusters()
            ));
        }
        if training_data.len() < kmeans_config.k {
            return Err(format!(
                "IVFPQIndex::train: insufficient training vectors ({}) for k={}",
                training_data.len(),
                kmeans_config.k
            ));
        }

        // 1. Train coarse centroids (full precision)
        let kmeans_result = kmeans(training_data, kmeans_config, ivf_seed);

        // 2. Train PQ subquantizer codebooks on the same data
        let pq_codebooks = train_pq(training_data, &self.pq_config, pq_seed)?;

        self.centroids = kmeans_result.centroids;
        self.pq_codebooks = Some(pq_codebooks);
        self.is_trained = true;

        Ok(())
    }

    /// Insert a single vector with its external ID into the index.
    ///
    /// Identifies the nearest coarse centroid, encodes the vector into a compact [`PQCode`],
    /// and pushes `(id, code)` into the corresponding cluster's inverted list.
    ///
    /// # Errors
    /// Returns `Err(String)` if:
    /// - Index is not trained (`!self.is_trained`)
    /// - Vector dimension mismatch (`vector.len() != self.dim`)
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), String> {
        if !self.is_trained {
            return Err("IVFPQIndex must be trained before adding vectors".to_string());
        }
        if vector.len() != self.dim {
            return Err(format!(
                "IVFPQIndex::add: expected vector dim {}, got {}",
                self.dim,
                vector.len()
            ));
        }

        let pq_codebooks = self.pq_codebooks.as_ref().ok_or_else(|| {
            "IVFPQIndex::add: PQ codebooks missing despite is_trained=true".to_string()
        })?;

        // 1. Coarse routing
        let nearest_cluster = find_nearest_centroid(&self.centroids, vector);

        // 2. Fine quantization
        let code = encode_vector(pq_codebooks, vector)?;

        // 3. Store compressed code
        self.inverted_lists[nearest_cluster].push((id, code));

        Ok(())
    }

    /// Bulk-insert vectors with their external IDs into the index.
    ///
    /// # Errors
    /// Returns `Err(String)` if:
    /// - Index is not trained (`!self.is_trained`)
    /// - Batch dimension mismatch (`vectors.dim != self.dim`)
    /// - `ids.len() != vectors.len()`
    pub fn add_batch(&mut self, ids: &[u64], vectors: &VectorBatch) -> Result<(), String> {
        if !self.is_trained {
            return Err("IVFPQIndex must be trained before adding vectors".to_string());
        }
        if vectors.dim != self.dim {
            return Err(format!(
                "IVFPQIndex::add_batch: expected vector dim {}, got {}",
                self.dim, vectors.dim
            ));
        }
        if ids.len() != vectors.len() {
            return Err(format!(
                "IVFPQIndex::add_batch: ids count ({}) != vectors count ({})",
                ids.len(),
                vectors.len()
            ));
        }

        for (i, &id) in ids.iter().enumerate() {
            self.add(id, vectors.get(i))?;
        }

        Ok(())
    }

    /// Search for the top-`k` nearest neighbors to `query` across the `nprobe` closest clusters
    /// using Asymmetric Distance Computation (ADC).
    ///
    /// # Search Lifecycle:
    /// 1. **Coarse Search**: Selects `nprobe` nearest coarse centroids via [`find_nearest_clusters`].
    /// 2. **ADC Table Construction**: Computes ONE `ADCLookupTable` for `query` across all codebooks.
    ///    Built once per query and reused across all probed inverted lists.
    /// 3. **Fine ADC Scoring**: For each `(id, code)` in selected inverted lists, computes
    ///    `adc_distance(&table, code)` via $m$ table lookups and additions.
    /// 4. **Top-k Heap Selection**: Finds the globally smallest $k$ distances using [`top_k_smallest`].
    ///
    /// # Errors
    /// Returns `Err(String)` if:
    /// - Index is not trained (`!self.is_trained`)
    /// - Query dimension mismatch (`query.len() != self.dim`)
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Result<Vec<ScoredId>, String> {
        if !self.is_trained {
            return Err("IVFPQIndex must be trained before searching".to_string());
        }
        if query.len() != self.dim {
            return Err(format!(
                "IVFPQIndex::search: query dimension {} != index dimension {}",
                query.len(),
                self.dim
            ));
        }
        if self.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let pq_codebooks = self.pq_codebooks.as_ref().ok_or_else(|| {
            "IVFPQIndex::search: PQ codebooks missing despite is_trained=true".to_string()
        })?;

        // 1. Coarse routing
        let nprobe_clamped = nprobe.max(1).min(self.centroids.len());
        let selected_clusters = find_nearest_clusters(&self.centroids, query, nprobe_clamped);

        // 2. Precompute ADC lookup table ONCE for this query
        let adc_table = build_adc_table(pq_codebooks, query)?;

        // 3. Fine scanning across selected inverted lists
        let candidate_cap: usize = selected_clusters
            .iter()
            .map(|&c| self.inverted_lists[c].len())
            .sum();
        let mut candidates = Vec::with_capacity(candidate_cap);

        for &cluster_idx in &selected_clusters {
            for &(id, ref code) in &self.inverted_lists[cluster_idx] {
                let dist = adc_distance(&adc_table, code)?;
                candidates.push(ScoredId { id, score: dist });
            }
        }

        // 4. Global top-k selection (smallest squared distance first)
        Ok(top_k_smallest(&candidates, k))
    }

    /// Bulk search: execute [`search`](Self::search) across a batch of query vectors.
    pub fn search_batch(
        &self,
        queries: &VectorBatch,
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<Vec<ScoredId>>, String> {
        if !self.is_trained {
            return Err("IVFPQIndex must be trained before searching".to_string());
        }
        if queries.dim != self.dim {
            return Err(format!(
                "IVFPQIndex::search_batch: queries dim {} != index dim {}",
                queries.dim, self.dim
            ));
        }

        let mut results = Vec::with_capacity(queries.len());
        for i in 0..queries.len() {
            results.push(self.search(queries.get(i), k, nprobe)?);
        }
        Ok(results)
    }

    /// Calculate the total memory footprint of this index in bytes.
    ///
    /// Sums:
    /// - Vector codes in inverted lists: `num_vectors * m` bytes.
    /// - Coarse centroids: `num_clusters * dim * size_of::<f32>()` bytes.
    /// - PQ Codebooks: `m * k_per_subvector * sub_dim * size_of::<f32>()` bytes.
    pub fn memory_footprint_bytes(&self) -> usize {
        let code_bytes = self.len() * self.pq_config.m;
        let centroids_bytes = self.centroids.len() * self.dim * std::mem::size_of::<f32>();
        let codebooks_bytes = match &self.pq_codebooks {
            Some(cb) => cb.m * cb.k_per_subvector * cb.sub_dim * std::mem::size_of::<f32>(),
            None => 0,
        };
        code_bytes + centroids_bytes + codebooks_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flat_index::FlatIndex;
    use crate::core::ivf_index::IVFIndex;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    /// Test 1: new() with dim not divisible by pq_config.m returns Err immediately.
    #[test]
    fn test_new_dim_not_divisible_by_m_returns_err() {
        let dim = 8;
        let pq_config = PQConfig {
            m: 3, // 8 is not divisible by 3
            k_per_subvector: 4,
            max_iterations: 10,
        };

        let res = IVFPQIndex::new(dim, 4, pq_config);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("dimension 8 is not evenly divisible by m=3 subvectors"),
            "Unexpected error: {}",
            err
        );
    }

    /// Test 2: add() before train() returns Err.
    #[test]
    fn test_add_before_train_returns_err() {
        let dim = 8;
        let pq_config = PQConfig {
            m: 2,
            k_per_subvector: 2,
            max_iterations: 5,
        };

        let mut index = IVFPQIndex::new(dim, 2, pq_config).unwrap();
        let v = [1.0; 8];
        let res = index.add(1, &v);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("IVFPQIndex must be trained before adding vectors"),
            "Unexpected error: {}",
            err
        );
    }

    /// Test 3: Hand-verifiable test:
    /// Small dataset, train IVFPQIndex, add known vectors, search, confirm the TOP result
    /// is reasonably close to what FlatIndex would return.
    #[test]
    fn test_hand_verified_easy_separation() {
        let dim = 4;
        let num_clusters = 2;
        let pq_config = PQConfig {
            m: 2,
            k_per_subvector: 2,
            max_iterations: 20,
        };

        let mut index = IVFPQIndex::new(dim, num_clusters, pq_config).unwrap();

        let mut data = VectorBatch::new(dim);
        // Cluster 1 near [0, 0, 0, 0]
        data.push(&[0.0, 0.1, 0.1, 0.0]); // id 1
        data.push(&[0.1, 0.0, 0.0, 0.1]); // id 2
        data.push(&[0.2, 0.2, 0.1, 0.1]); // id 3

        // Cluster 2 near [10, 10, 10, 10]
        data.push(&[10.0, 9.9, 10.1, 10.0]); // id 4
        data.push(&[9.9, 10.0, 10.0, 9.9]); // id 5
        data.push(&[10.1, 10.1, 9.9, 10.0]); // id 6

        let kmeans_config = KMeansConfig {
            k: num_clusters,
            max_iterations: 20,
            tolerance: 1e-4,
        };

        index.train(&data, &kmeans_config, 42, 42).unwrap();

        let ids: Vec<u64> = (1..=6).collect();
        index.add_batch(&ids, &data).unwrap();

        // Build oracle FlatIndex
        let mut flat = FlatIndex::new(dim, Metric::Euclidean);
        flat.add_batch(&ids, &data);

        // Query point near Cluster 1
        let query = [0.05, 0.05, 0.05, 0.05];

        let flat_results = flat.search(&query, 3);
        let ivfpq_results = index.search(&query, 3, 2).unwrap();

        println!("\nPhase 23 Test 3: Hand-verified search comparison:");
        println!(
            "  Flat Top-1:   id={}, score={:.4}",
            flat_results[0].id, flat_results[0].score
        );
        println!(
            "  IVFPQ Top-1:  id={}, score(dist²)={:.4}",
            ivfpq_results[0].id, ivfpq_results[0].score
        );

        // In an easily-separated 2-cluster setting, the top retrieved neighbor should be in Cluster 1 (id 1, 2, or 3)
        assert!(
            ivfpq_results[0].id <= 3,
            "Top result id {} should belong to Cluster 1",
            ivfpq_results[0].id
        );
        assert_eq!(ivfpq_results.len(), 3);
    }

    /// Test 4: Recall test on real/synthetic data:
    /// Build IVFPQIndex, compare recall@10 against FlatIndex ground truth across a couple
    /// nprobe values. Compare side by side with plain IVFIndex.
    #[test]
    fn test_recall_comparison_vs_plain_ivf() {
        let n = 1000;
        let dim = 32;
        let k_clusters = 16;
        let top_k = 10;
        let num_queries = 20;

        let mut rng = StdRng::seed_from_u64(42);

        // 1. Generate dataset
        let mut data = VectorBatch::new(dim);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-10.0..10.0));
            }
            data.push(&v);
        }

        let ids: Vec<u64> = (0..n as u64).collect();

        // 2. Build FlatIndex Oracle
        let mut flat = FlatIndex::new(dim, Metric::Euclidean);
        flat.add_batch(&ids, &data);

        // 3. Build plain IVFIndex
        let mut ivf = IVFIndex::new(dim, k_clusters, Metric::Euclidean);
        let kmeans_config = KMeansConfig {
            k: k_clusters,
            max_iterations: 20,
            tolerance: 1e-3,
        };
        ivf.train(&data, &kmeans_config, 123);
        ivf.add_batch(&ids, &data).unwrap();

        // 4. Build IVFPQIndex
        let pq_config = PQConfig {
            m: 4,
            k_per_subvector: 16,
            max_iterations: 15,
        };
        let mut ivf_pq = IVFPQIndex::new(dim, k_clusters, pq_config).unwrap();
        ivf_pq.train(&data, &kmeans_config, 123, 456).unwrap();
        ivf_pq.add_batch(&ids, &data).unwrap();

        // 5. Generate queries
        let mut queries = Vec::new();
        for _ in 0..num_queries {
            let mut q = Vec::with_capacity(dim);
            for _ in 0..dim {
                q.push(rng.gen_range(-10.0..10.0));
            }
            queries.push(q);
        }

        // Ground-truth nearest neighbors from FlatIndex
        let mut ground_truth: Vec<HashSet<u64>> = Vec::new();
        for q in &queries {
            let gt_results = flat.search(q, top_k);
            ground_truth.push(gt_results.iter().map(|s| s.id).collect());
        }

        let nprobe_values = [1, 4, 16];

        println!(
            "\nPhase 23 Test 4: Recall@{} comparison: IVFIndex vs IVFPQIndex (N={}, k_clusters={}, m=4):",
            top_k, n, k_clusters
        );

        for &nprobe in &nprobe_values {
            let mut ivf_overlap = 0;
            let mut ivf_pq_overlap = 0;

            for (q_idx, q) in queries.iter().enumerate() {
                let gt = &ground_truth[q_idx];

                let ivf_res = ivf.search(q, top_k, nprobe);
                let ivf_set: HashSet<u64> = ivf_res.iter().map(|s| s.id).collect();
                ivf_overlap += ivf_set.intersection(gt).count();

                let ivfpq_res = ivf_pq.search(q, top_k, nprobe).unwrap();
                let ivfpq_set: HashSet<u64> = ivfpq_res.iter().map(|s| s.id).collect();
                ivf_pq_overlap += ivfpq_set.intersection(gt).count();
            }

            let ivf_recall = (ivf_overlap as f64) / ((num_queries * top_k) as f64);
            let ivfpq_recall = (ivf_pq_overlap as f64) / ((num_queries * top_k) as f64);

            println!(
                "  nprobe={:>2}: IVFIndex recall@{} = {:>5.1}% | IVFPQIndex recall@{} = {:>5.1}%",
                nprobe,
                top_k,
                ivf_recall * 100.0,
                top_k,
                ivfpq_recall * 100.0
            );

            // Sanity check: IVFPQ recall should be non-zero and positive
            assert!(ivfpq_recall > 0.0);
        }
    }

    /// Test 5: memory_footprint_bytes() sanity check:
    /// Compare reported footprint against manually-computed expectation.
    #[test]
    fn test_memory_footprint_bytes_sanity() {
        let dim = 16;
        let num_clusters = 4;
        let m = 4;
        let k_per_sub = 8;

        let pq_config = PQConfig {
            m,
            k_per_subvector: k_per_sub,
            max_iterations: 5,
        };

        let mut index = IVFPQIndex::new(dim, num_clusters, pq_config).unwrap();

        let n = 100;
        let mut data = VectorBatch::new(dim);
        for i in 0..n {
            data.push(&[i as f32; 16]);
        }

        let km_config = KMeansConfig {
            k: num_clusters,
            max_iterations: 5,
            tolerance: 1e-4,
        };

        index.train(&data, &km_config, 1, 1).unwrap();
        let ids: Vec<u64> = (0..n as u64).collect();
        index.add_batch(&ids, &data).unwrap();

        let code_bytes = n * m; // 100 * 4 = 400 bytes
        let centroids_bytes = num_clusters * dim * std::mem::size_of::<f32>(); // 4 * 16 * 4 = 256 bytes
        let sub_dim = dim / m; // 4
        let codebooks_bytes = m * k_per_sub * sub_dim * std::mem::size_of::<f32>(); // 4 * 8 * 4 * 4 = 512 bytes
        let expected_footprint = code_bytes + centroids_bytes + codebooks_bytes; // 1168 bytes

        let actual_footprint = index.memory_footprint_bytes();

        println!("\nPhase 23 Test 5: Memory footprint sanity check:");
        println!("  Expected total footprint: {} bytes", expected_footprint);
        println!("  Reported total footprint: {} bytes", actual_footprint);

        assert_eq!(actual_footprint, expected_footprint);
    }

    /// Test 6: A concrete memory comparison test:
    /// Build BOTH an IVFIndex and an IVFPQIndex on the SAME 1,000 vectors (dim=128, m=8, k=256).
    /// Print the memory footprint of each side by side, demonstrating real compression savings.
    #[test]
    fn test_concrete_memory_comparison() {
        let n = 1000;
        let dim = 128;
        let num_clusters = 16;
        let m = 8;
        let k_per_subvector = 256;

        let mut data = VectorBatch::new(dim);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-1.0..1.0));
            }
            data.push(&v);
        }

        let ids: Vec<u64> = (0..n as u64).collect();

        // 1. Train and build IVFIndex
        let mut ivf = IVFIndex::new(dim, num_clusters, Metric::Euclidean);
        let km_config = KMeansConfig {
            k: num_clusters,
            max_iterations: 5,
            tolerance: 1e-4,
        };
        ivf.train(&data, &km_config, 100);
        ivf.add_batch(&ids, &data).unwrap();

        // 2. Train and build IVFPQIndex
        let pq_config = PQConfig {
            m,
            k_per_subvector,
            max_iterations: 5,
        };
        let mut ivf_pq = IVFPQIndex::new(dim, num_clusters, pq_config).unwrap();
        ivf_pq.train(&data, &km_config, 100, 200).unwrap();
        ivf_pq.add_batch(&ids, &data).unwrap();

        // Memory calculations
        let ivf_vector_bytes = n * dim * std::mem::size_of::<f32>(); // 1000 * 128 * 4 = 512,000 bytes
        let ivf_centroids_bytes = num_clusters * dim * std::mem::size_of::<f32>(); // 16 * 128 * 4 = 8,192 bytes
        let ivf_total_bytes = ivf_vector_bytes + ivf_centroids_bytes;

        let ivfpq_vector_bytes = n * m; // 1000 * 8 = 8,000 bytes
        let ivfpq_total_bytes = ivf_pq.memory_footprint_bytes();

        let vector_compression_ratio = (ivf_vector_bytes as f64) / (ivfpq_vector_bytes as f64);
        let total_compression_ratio = (ivf_total_bytes as f64) / (ivfpq_total_bytes as f64);

        println!(
            "\nPhase 23 Test 6: Concrete Memory Comparison (N={}, dim={}, m={}, k_clusters={}):",
            n, dim, m, num_clusters
        );
        println!(
            "  IVFIndex stored vectors:    {} bytes ({:.1} KB)",
            ivf_vector_bytes,
            ivf_vector_bytes as f64 / 1024.0
        );
        println!(
            "  IVFPQIndex stored vectors:  {} bytes ({:.1} KB)",
            ivfpq_vector_bytes,
            ivfpq_vector_bytes as f64 / 1024.0
        );
        println!(
            "  Vector Data Compression:    {:.1}x",
            vector_compression_ratio
        );
        println!(
            "  IVFIndex total footprint:   {} bytes ({:.1} KB)",
            ivf_total_bytes,
            ivf_total_bytes as f64 / 1024.0
        );
        println!(
            "  IVFPQIndex total footprint: {} bytes ({:.1} KB)",
            ivfpq_total_bytes,
            ivfpq_total_bytes as f64 / 1024.0
        );
        println!(
            "  Total Index Compression:    {:.1}x (including codebooks)",
            total_compression_ratio
        );

        assert_eq!(ivf_vector_bytes, 512_000);
        assert_eq!(ivfpq_vector_bytes, 8_000);
        assert_eq!(vector_compression_ratio, 64.0);
        assert!(ivfpq_total_bytes < ivf_total_bytes);
    }
}
