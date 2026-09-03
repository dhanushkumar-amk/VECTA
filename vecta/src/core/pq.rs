//! Product Quantization (PQ) codebook training.
//!
//! Splits high-dimensional vectors into `m` orthogonal subvector subspaces
//! and trains codebooks via Lloyd's k-means independently for each subvector.
//!
//! Follows the formulation of Jégou, Douze, & Schmid (2011),
//! "Product Quantization for Nearest Neighbor Search".

use crate::core::batch::VectorBatch;
use crate::core::kmeans::{kmeans, KMeansConfig};

/// Configuration parameters for Product Quantization training.
#[derive(Debug, Clone, PartialEq)]
pub struct PQConfig {
    /// Number of subvectors (subquantizers). Must evenly divide `dim`.
    pub m: usize,
    /// Number of centroids per subquantizer codebook (typically 256 for 1 byte/subvector).
    pub k_per_subvector: usize,
    /// Maximum Lloyd iterations per subvector k-means clustering run.
    pub max_iterations: usize,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            m: 8,
            k_per_subvector: 256,
            max_iterations: 100,
        }
    }
}

/// A collection of trained Product Quantization codebooks.
///
/// Contains `m` distinct codebooks, where `codebooks[i]` contains
/// `k_per_subvector` centroids of dimensionality `sub_dim = dim / m`.
#[derive(Debug, Clone)]
pub struct PQCodebooks {
    /// Number of subvectors / codebooks.
    pub m: usize,
    /// Dimensionality of each subvector (`dim / m`).
    pub sub_dim: usize,
    /// Number of centroids per subvector codebook.
    pub k_per_subvector: usize,
    /// VectorBatch of centroids for each subquantizer position `0..m`.
    pub codebooks: Vec<VectorBatch>,
}

/// Train Product Quantization codebooks on a training dataset.
///
/// # Algorithm:
/// 1. Verifies that `data.dim` is evenly divisible by `config.m`.
/// 2. For each subvector position `0..m`:
///    - Slices columns `[pos * sub_dim .. (pos + 1) * sub_dim]` of all training vectors into a sub-batch.
///    - Runs Lloyd's k-means clustering ([`kmeans`]) on the sub-batch using a unique, deterministic seed (`seed + pos`).
///    - Appends the resulting `k_per_subvector` centroids to `codebooks`.
///
/// # Errors:
/// Returns an informative `Err(String)` if:
/// - `config.m == 0`
/// - `data.dim` is not evenly divisible by `config.m`
/// - `data.len() < config.k_per_subvector`
/// - `config.k_per_subvector == 0`
pub fn train_pq(data: &VectorBatch, config: &PQConfig, seed: u64) -> Result<PQCodebooks, String> {
    if config.m == 0 {
        return Err("PQConfig::m must be greater than 0".to_string());
    }
    if config.k_per_subvector == 0 {
        return Err("PQConfig::k_per_subvector must be greater than 0".to_string());
    }
    if !data.dim.is_multiple_of(config.m) {
        return Err(format!(
            "dimension {} is not evenly divisible by m={} subvectors",
            data.dim, config.m
        ));
    }
    if data.len() < config.k_per_subvector {
        return Err(format!(
            "insufficient training vectors ({}) for k_per_subvector={}",
            data.len(),
            config.k_per_subvector
        ));
    }

    let sub_dim = data.dim / config.m;
    let n = data.len();

    let mut codebooks = Vec::with_capacity(config.m);

    for pos in 0..config.m {
        let start_dim = pos * sub_dim;
        let end_dim = start_dim + sub_dim;

        // Slice subvectors across all training rows
        let mut sub_batch = VectorBatch::new(sub_dim);
        for i in 0..n {
            let row = data.get(i);
            sub_batch.push(&row[start_dim..end_dim]);
        }

        // Run k-means with a distinct seed per subvector position
        let km_config = KMeansConfig {
            k: config.k_per_subvector,
            max_iterations: config.max_iterations,
            tolerance: 1e-4,
        };
        let sub_seed = seed.wrapping_add(pos as u64);
        let km_res = kmeans(&sub_batch, &km_config, sub_seed);

        codebooks.push(km_res.centroids);
    }

    Ok(PQCodebooks {
        m: config.m,
        sub_dim,
        k_per_subvector: config.k_per_subvector,
        codebooks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::time::Instant;

    /// Test 1: Non-divisible dimension returns Err with clear message, does not panic.
    #[test]
    fn test_non_divisible_dimension_returns_err() {
        let dim = 8;
        let mut data = VectorBatch::new(dim);
        data.push(&[0.0; 8]);
        data.push(&[1.0; 8]);

        let config = PQConfig {
            m: 3, // 8 is not divisible by 3
            k_per_subvector: 2,
            max_iterations: 10,
        };

        let res = train_pq(&data, &config, 42);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("dimension 8 is not evenly divisible by m=3 subvectors"),
            "Unexpected error message: {}",
            err
        );
    }

    /// Test 2: Evenly divisible dimension produces exactly m codebooks.
    #[test]
    fn test_evenly_divisible_produces_m_codebooks() {
        let dim = 8;
        let mut data = VectorBatch::new(dim);
        for i in 0..10 {
            data.push(&[i as f32; 8]);
        }

        let config = PQConfig {
            m: 4, // sub_dim = 2
            k_per_subvector: 2,
            max_iterations: 10,
        };

        let res = train_pq(&data, &config, 42);
        assert!(res.is_ok());
        let pq = res.unwrap();
        assert_eq!(pq.m, 4);
        assert_eq!(pq.sub_dim, 2);
        assert_eq!(pq.codebooks.len(), 4);
    }

    /// Test 3: Each codebook has exactly k_per_subvector centroids.
    #[test]
    fn test_codebook_centroid_count() {
        let dim = 6;
        let mut data = VectorBatch::new(dim);
        for i in 0..20 {
            data.push(&[i as f32; 6]);
        }

        let k_per_sub = 4;
        let config = PQConfig {
            m: 3, // sub_dim = 2
            k_per_subvector: k_per_sub,
            max_iterations: 10,
        };

        let pq = train_pq(&data, &config, 123).unwrap();
        for (i, cb) in pq.codebooks.iter().enumerate() {
            assert_eq!(
                cb.len(),
                k_per_sub,
                "Codebook at position {} has {} centroids, expected {}",
                i,
                cb.len(),
                k_per_sub
            );
        }
    }

    /// Test 4: Each codebook's centroids have dimension == sub_dim (not full dim).
    #[test]
    fn test_codebook_sub_dimension() {
        let dim = 12;
        let m = 3;
        let expected_sub_dim = 4;

        let mut data = VectorBatch::new(dim);
        for i in 0..15 {
            data.push(&[i as f32; 12]);
        }

        let config = PQConfig {
            m,
            k_per_subvector: 3,
            max_iterations: 10,
        };

        let pq = train_pq(&data, &config, 456).unwrap();
        assert_eq!(pq.sub_dim, expected_sub_dim);
        for (i, cb) in pq.codebooks.iter().enumerate() {
            assert_eq!(
                cb.dim(),
                expected_sub_dim,
                "Codebook {} dimension {} != expected sub_dim {}",
                i,
                cb.dim(),
                expected_sub_dim
            );
        }
    }

    /// Test 5: Hand-verifiable test:
    /// Subvector position 0 values cluster obviously near [0, 0] and [10, 10],
    /// regardless of what remaining dimensions contain.
    #[test]
    fn test_hand_verified_subvector_clustering() {
        let dim = 4;
        let mut data = VectorBatch::new(dim);

        // Group A: dims [0..2] near [0.0, 0.0], dims [2..4] arbitrary
        data.push(&[0.0, 0.1, 99.0, 88.0]);
        data.push(&[0.1, 0.0, 77.0, 66.0]);
        data.push(&[0.2, 0.2, 55.0, 44.0]);

        // Group B: dims [0..2] near [10.0, 10.0], dims [2..4] arbitrary
        data.push(&[10.0, 9.9, 11.0, 22.0]);
        data.push(&[9.9, 10.0, 33.0, 44.0]);
        data.push(&[10.1, 10.1, 55.0, 66.0]);

        let config = PQConfig {
            m: 2, // sub_dim = 2
            k_per_subvector: 2,
            max_iterations: 20,
        };

        let pq = train_pq(&data, &config, 42).unwrap();
        let cb0 = &pq.codebooks[0];
        assert_eq!(cb0.len(), 2);
        assert_eq!(cb0.dim(), 2);

        let mut c0 = cb0.get(0).to_vec();
        let mut c1 = cb0.get(1).to_vec();

        // Sort centroids by first dimension for consistent inspection
        if c0[0] > c1[0] {
            std::mem::swap(&mut c0, &mut c1);
        }

        println!("\nPhase 20 Test 5: Hand-verified subvector 0 centroids:");
        println!(
            "  Centroid 0 (expected near [0.1, 0.1]):   [{:.3}, {:.3}]",
            c0[0], c0[1]
        );
        println!(
            "  Centroid 1 (expected near [10.0, 10.0]): [{:.3}, {:.3}]",
            c1[0], c1[1]
        );

        // Assert Centroid 0 is near [0.1, 0.1]
        assert!((c0[0] - 0.1).abs() < 0.15);
        assert!((c0[1] - 0.1).abs() < 0.15);

        // Assert Centroid 1 is near [10.0, 10.0]
        assert!((c1[0] - 10.0).abs() < 0.15);
        assert!((c1[1] - 10.0).abs() < 0.15);
    }

    /// Test 6: Reproducibility test:
    /// Same seed and training data produces identical codebooks.
    #[test]
    fn test_pq_training_reproducibility() {
        let dim = 8;
        let mut data = VectorBatch::new(dim);
        let mut rng = StdRng::seed_from_u64(999);

        for _ in 0..50 {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-5.0..5.0));
            }
            data.push(&v);
        }

        let config = PQConfig {
            m: 4,
            k_per_subvector: 4,
            max_iterations: 25,
        };

        let pq1 = train_pq(&data, &config, 12345).unwrap();
        let pq2 = train_pq(&data, &config, 12345).unwrap();

        assert_eq!(pq1.codebooks.len(), pq2.codebooks.len());
        for pos in 0..config.m {
            let cb1 = &pq1.codebooks[pos];
            let cb2 = &pq2.codebooks[pos];
            assert_eq!(cb1.len(), cb2.len());
            for k_idx in 0..cb1.len() {
                let v1 = cb1.get(k_idx);
                let v2 = cb2.get(k_idx);
                for d in 0..pq1.sub_dim {
                    assert_eq!(
                        v1[d], v2[d],
                        "Centroid mismatch at subvector pos {}, k_idx {}, dim {}",
                        pos, k_idx, d
                    );
                }
            }
        }
    }

    /// Test 7: Realistic-scale test:
    /// 1,000 vectors of 128 dimensions, m=8 (sub_dim=16), k_per_subvector=256.
    #[test]
    fn test_realistic_scale_pq_training() {
        let n = 1000;
        let dim = 128;
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

        let config = PQConfig {
            m,
            k_per_subvector,
            max_iterations: 20,
        };

        let start = Instant::now();
        let pq = train_pq(&data, &config, 100).unwrap();
        let elapsed = start.elapsed();

        println!(
            "\nPhase 20 Test 7: Realistic-scale PQ training (N={}, dim={}, m={}, k={}):",
            n, dim, m, k_per_subvector
        );
        println!(
            "  Training time: {:.2?} ({:.2} ms)",
            elapsed,
            elapsed.as_secs_f64() * 1000.0
        );
        println!("  Codebooks produced: {}", pq.codebooks.len());
        println!("  Centroids per codebook: {}", pq.codebooks[0].len());
        println!("  Subvector dimension: {}", pq.sub_dim);

        assert_eq!(pq.codebooks.len(), m);
        assert_eq!(pq.sub_dim, dim / m);
        for cb in &pq.codebooks {
            assert_eq!(cb.len(), k_per_subvector);
            assert_eq!(cb.dim(), dim / m);
        }
    }
}
