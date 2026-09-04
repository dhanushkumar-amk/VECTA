//! Product Quantization (PQ) codebook training and vector encoding/decoding.
//!
//! Splits high-dimensional vectors into `m` orthogonal subvector subspaces,
//! trains codebooks via Lloyd's k-means independently for each subvector,
//! and quantizes vectors into compact byte codes ([`PQCode`]).
//!
//! Follows the formulation of Jégou, Douze, & Schmid (2011),
//! "Product Quantization for Nearest Neighbor Search".

use crate::core::batch::VectorBatch;
use crate::core::kmeans::{kmeans, KMeansConfig};
use crate::core::vector::euclidean_distance;

/// Compact representation of a quantized vector.
///
/// Stores `m` centroid IDs (one `u8` per subvector position).
/// Valid as long as `k_per_subvector <= 256`.
pub type PQCode = Vec<u8>;

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

impl PQCodebooks {
    /// Expected full vector dimension (`m * sub_dim`).
    #[inline]
    pub fn dim(&self) -> usize {
        self.m * self.sub_dim
    }
}

/// Extract a subvector slice for subvector position `pos` with dimensionality `sub_dim`.
///
/// Shared helper used across [`train_pq`], [`encode_vector`], and [`build_adc_table`]
/// to guarantee identical slicing layout and bounds handling.
#[inline]
fn subvector_slice(vector: &[f32], pos: usize, sub_dim: usize) -> &[f32] {
    let start = pos * sub_dim;
    let end = start + sub_dim;
    &vector[start..end]
}

/// Compute squared Euclidean distance between two slices: Σ(a_i - b_i)².
///
/// Avoids the square root operation for table construction and distance calculations.
#[inline]
fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Train Product Quantization codebooks on a training dataset.
///
/// # Algorithm:
/// 1. Verifies that `data.dim` is evenly divisible by `config.m`.
/// 2. Verifies that `config.k_per_subvector <= 256` (1-byte representation).
/// 3. For each subvector position `0..m`:
///    - Slices columns `[pos * sub_dim .. (pos + 1) * sub_dim]` of all training vectors into a sub-batch.
///    - Runs Lloyd's k-means clustering ([`kmeans`]) on the sub-batch using a unique, deterministic seed (`seed + pos`).
///    - Appends the resulting `k_per_subvector` centroids to `codebooks`.
///
/// # Errors:
/// Returns an informative `Err(String)` if:
/// - `config.m == 0`
/// - `config.k_per_subvector == 0`
/// - `config.k_per_subvector > 256` (cannot fit into 1 byte)
/// - `data.dim` is not evenly divisible by `config.m`
/// - `data.len() < config.k_per_subvector`
pub fn train_pq(data: &VectorBatch, config: &PQConfig, seed: u64) -> Result<PQCodebooks, String> {
    if config.m == 0 {
        return Err("PQConfig::m must be greater than 0".to_string());
    }
    if config.k_per_subvector == 0 {
        return Err("PQConfig::k_per_subvector must be greater than 0".to_string());
    }
    if config.k_per_subvector > 256 {
        return Err(format!(
            "PQConfig::k_per_subvector ({}) cannot exceed 256 for 1-byte encoding",
            config.k_per_subvector
        ));
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
        // Slice subvectors across all training rows using the shared subvector_slice helper
        let mut sub_batch = VectorBatch::new(sub_dim);
        for i in 0..n {
            let row = data.get(i);
            sub_batch.push(subvector_slice(row, pos, sub_dim));
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

/// Encode a single full-precision vector into a compact [`PQCode`].
///
/// For each subvector position `0..m`, finds the nearest centroid in the corresponding
/// codebook using Euclidean distance and records its index as a `u8`.
///
/// # Errors:
/// Returns an informative `Err(String)` if `vector.len() != codebooks.dim()`.
pub fn encode_vector(codebooks: &PQCodebooks, vector: &[f32]) -> Result<PQCode, String> {
    let expected_dim = codebooks.dim();
    if vector.len() != expected_dim {
        return Err(format!(
            "vector dimension mismatch: expected {}, got {}",
            expected_dim,
            vector.len()
        ));
    }

    let mut code = Vec::with_capacity(codebooks.m);

    for (pos, cb) in codebooks.codebooks.iter().enumerate() {
        let sub_vec = subvector_slice(vector, pos, codebooks.sub_dim);

        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for c_idx in 0..cb.len() {
            let centroid = cb.get(c_idx);
            let d = euclidean_distance(sub_vec, centroid);
            if d < best_dist {
                best_dist = d;
                best_idx = c_idx;
            }
        }

        code.push(best_idx as u8);
    }

    Ok(code)
}

/// Encode a batch of vectors into a `Vec<PQCode>`.
///
/// Validates that `batch.dim` matches `codebooks.dim()` up front.
///
/// # Errors:
/// Returns an informative `Err(String)` if `batch.dim != codebooks.dim()`.
pub fn encode_batch(codebooks: &PQCodebooks, batch: &VectorBatch) -> Result<Vec<PQCode>, String> {
    let expected_dim = codebooks.dim();
    if batch.dim != expected_dim {
        return Err(format!(
            "batch dimension mismatch: expected {}, got {}",
            expected_dim, batch.dim
        ));
    }

    let mut codes = Vec::with_capacity(batch.len());
    for i in 0..batch.len() {
        codes.push(encode_vector(codebooks, batch.get(i))?);
    }

    Ok(codes)
}

/// Reconstruct an approximate full-precision vector from a compact [`PQCode`].
///
/// Looks up each subvector position's stored centroid ID in the corresponding
/// codebook and concatenates the centroid vectors together.
///
/// # Errors:
/// Returns an informative `Err(String)` if `code.len() != codebooks.m` or if
/// a centroid index is out of bounds for the codebook.
pub fn decode_vector(codebooks: &PQCodebooks, code: &PQCode) -> Result<Vec<f32>, String> {
    if code.len() != codebooks.m {
        return Err(format!(
            "PQCode length mismatch: expected {}, got {}",
            codebooks.m,
            code.len()
        ));
    }

    let mut reconstructed = Vec::with_capacity(codebooks.dim());

    for (pos, &centroid_id) in code.iter().enumerate() {
        let cb = &codebooks.codebooks[pos];
        let idx = centroid_id as usize;
        if idx >= cb.len() {
            return Err(format!(
                "centroid index {} out of bounds for codebook {} with len {}",
                idx,
                pos,
                cb.len()
            ));
        }
        reconstructed.extend_from_slice(cb.get(idx));
    }

    Ok(reconstructed)
}

/// Precomputed lookup table for Asymmetric Distance Computation (ADC).
///
/// Precomputes squared Euclidean distances from each subvector of a query vector
/// to all `k_per_subvector` centroids in the corresponding codebook.
///
/// Query distance to any compressed [`PQCode`] is evaluated via `m` table lookups
/// and additions without decoding the stored vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ADCLookupTable {
    /// Number of subvectors (subquantizers).
    pub m: usize,
    /// Number of centroids per subvector codebook.
    pub k_per_subvector: usize,
    /// Precomputed squared Euclidean distances:
    /// `tables[i][j]` = squared distance from query subvector `i` to codebook `i` centroid `j`.
    pub tables: Vec<Vec<f32>>,
}

/// Precompute an Asymmetric Distance Computation (ADC) lookup table for a full-precision query.
///
/// For each subvector position `i` in `0..m`, extracts the query's subvector slice using
/// `subvector_slice` and computes squared Euclidean distance to each centroid `j` in codebook `i`.
///
/// # Mathematical Note:
/// We compute **squared Euclidean distance**, NOT the square-rooted version. By orthogonality
/// of the `m` subvector subspaces:
/// ```text
/// ||q - v̂||² = Σ_{i=0}^{m-1} ||q_i - c_{i, j_i}||²
/// ```
/// Summing squared Euclidean distances across orthogonal subvectors exactly equals the squared
/// Euclidean distance of the full reconstructed vector. Taking square roots per-subvector and
/// summing those would NOT be mathematically equivalent to any valid distance metric.
///
/// # Errors:
/// Returns an informative `Err(String)` if `query.len() != codebooks.dim()`.
pub fn build_adc_table(codebooks: &PQCodebooks, query: &[f32]) -> Result<ADCLookupTable, String> {
    let expected_dim = codebooks.dim();
    if query.len() != expected_dim {
        return Err(format!(
            "query dimension mismatch: expected {}, got {}",
            expected_dim,
            query.len()
        ));
    }

    let mut tables = Vec::with_capacity(codebooks.m);

    for (pos, cb) in codebooks.codebooks.iter().enumerate() {
        let query_sub = subvector_slice(query, pos, codebooks.sub_dim);
        let mut sub_dists = Vec::with_capacity(cb.len());

        for c_idx in 0..cb.len() {
            let centroid = cb.get(c_idx);
            sub_dists.push(squared_euclidean_distance(query_sub, centroid));
        }

        tables.push(sub_dists);
    }

    Ok(ADCLookupTable {
        m: codebooks.m,
        k_per_subvector: codebooks.k_per_subvector,
        tables,
    })
}

/// Compute approximate squared Euclidean distance between a query and a compressed [`PQCode`]
/// using precomputed ADC lookup tables.
///
/// Does NOT decode or reconstruct the vector floats; computes distance entirely via
/// `m` table lookups and additions.
///
/// # SQUARED Distance and Ranking Note:
/// This returns **squared Euclidean distance** consistent with [`build_adc_table`].
/// Callers requiring true Euclidean distance can call `.sqrt()` on the result.
/// However, for top-k neighbor ranking and nearest-neighbor search, squared Euclidean
/// distance is strictly monotonic with Euclidean distance ($a < b \iff a^2 < b^2$),
/// preserving the exact same ordering without the floating-point `.sqrt()` overhead.
///
/// # Errors:
/// Returns an informative `Err(String)` if `code.len() != table.m` or if a code entry
/// is out of bounds for the table.
pub fn adc_distance(table: &ADCLookupTable, code: &PQCode) -> Result<f32, String> {
    if code.len() != table.m {
        return Err(format!(
            "PQCode length mismatch: expected {}, got {}",
            table.m,
            code.len()
        ));
    }

    let mut total_dist = 0.0f32;

    for (pos, &centroid_id) in code.iter().enumerate() {
        let idx = centroid_id as usize;
        let sub_table = &table.tables[pos];
        if idx >= sub_table.len() {
            return Err(format!(
                "centroid index {} out of bounds for subvector table {} with len {}",
                idx,
                pos,
                sub_table.len()
            ));
        }
        total_dist += sub_table[idx];
    }

    Ok(total_dist)
}

/// Compute approximate squared Euclidean distances between a query and a batch of [`PQCode`]s.
///
/// Calls [`adc_distance`] for each code in the batch. Used during search queries:
/// precompute table once with [`build_adc_table`], then evaluate across thousands of stored codes.
///
/// # Errors:
/// Returns an informative `Err(String)` if any code in `codes` has invalid length or centroid ID.
pub fn adc_distance_batch(table: &ADCLookupTable, codes: &[PQCode]) -> Result<Vec<f32>, String> {
    let mut distances = Vec::with_capacity(codes.len());
    for code in codes {
        distances.push(adc_distance(table, code)?);
    }
    Ok(distances)
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

    // ==========================================
    // Phase 21 Tests: PQ Vector Encoding/Decoding
    // ==========================================

    /// Test 1: train_pq() with k_per_subvector > 256 (e.g. 300) returns Err.
    #[test]
    fn test_train_pq_k_exceeding_256_returns_err() {
        let dim = 8;
        let mut data = VectorBatch::new(dim);
        for _ in 0..350 {
            data.push(&[0.5; 8]);
        }

        let config = PQConfig {
            m: 2,
            k_per_subvector: 300, // Exceeds 256 limit for 1-byte code
            max_iterations: 10,
        };

        let res = train_pq(&data, &config, 42);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("cannot exceed 256 for 1-byte encoding"),
            "Unexpected error: {}",
            err
        );
    }

    /// Test 2: encode_vector() with wrong-length input vector returns Err.
    #[test]
    fn test_encode_vector_wrong_dimension_returns_err() {
        let dim = 8;
        let mut data = VectorBatch::new(dim);
        for _ in 0..10 {
            data.push(&[1.0; 8]);
        }

        let config = PQConfig {
            m: 2,
            k_per_subvector: 2,
            max_iterations: 5,
        };
        let codebooks = train_pq(&data, &config, 42).unwrap();

        // Vector length 7 != expected 8
        let v_wrong = [1.0; 7];
        let res = encode_vector(&codebooks, &v_wrong);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("vector dimension mismatch"));
    }

    /// Test 3: decode_vector() with wrong-length code returns Err.
    #[test]
    fn test_decode_vector_wrong_length_returns_err() {
        let dim = 8;
        let mut data = VectorBatch::new(dim);
        for _ in 0..10 {
            data.push(&[1.0; 8]);
        }

        let config = PQConfig {
            m: 4,
            k_per_subvector: 2,
            max_iterations: 5,
        };
        let codebooks = train_pq(&data, &config, 42).unwrap();

        // Code length 3 != expected m=4
        let code_wrong: PQCode = vec![0, 1, 0];
        let res = decode_vector(&codebooks, &code_wrong);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("PQCode length mismatch"));
    }

    /// Test 4: Hand-verifiable test:
    /// Using the SAME hand-crafted 2-cluster subvector data from Phase 20's test 5
    /// (dimension-0-1 values near [0,0] or [10,10]), encode a vector known to be
    /// near [0,0] in that subvector position — confirm the resulting code's first
    /// byte corresponds to whichever codebook index landed near [0,0].
    #[test]
    fn test_hand_verified_encoding() {
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

        // Determine which centroid index (0 or 1) landed near [0, 0] vs [10, 10]
        let c0 = cb0.get(0);
        let c1 = cb0.get(1);
        let (idx_zero, idx_ten) = if c0[0] < 5.0 { (0u8, 1u8) } else { (1u8, 0u8) };

        println!("\nPhase 21 Test 4: Hand-verified codebook index assignments:");
        println!("  Codebook 0, centroid 0: [{:.3}, {:.3}]", c0[0], c0[1]);
        println!("  Codebook 0, centroid 1: [{:.3}, {:.3}]", c1[0], c1[1]);
        println!(
            "  Assignment: near-[0,0] -> index {}, near-[10,10] -> index {}",
            idx_zero, idx_ten
        );

        // Encode vector with dims 0..2 near [0, 0]
        let v_near_zero = [0.05, 0.05, 50.0, 50.0];
        let code_zero = encode_vector(&pq, &v_near_zero).unwrap();
        println!(
            "  Encoded v_near_zero: code[0] = {} (expected {})",
            code_zero[0], idx_zero
        );
        assert_eq!(
            code_zero[0], idx_zero,
            "Expected code[0] to match near-[0,0] centroid index"
        );

        // Encode vector with dims 0..2 near [10, 10]
        let v_near_ten = [9.95, 10.05, 50.0, 50.0];
        let code_ten = encode_vector(&pq, &v_near_ten).unwrap();
        println!(
            "  Encoded v_near_ten:  code[0] = {} (expected {})",
            code_ten[0], idx_ten
        );
        assert_eq!(
            code_ten[0], idx_ten,
            "Expected code[0] to match near-[10,10] centroid index"
        );
    }

    /// Test 5: Round-trip test: encode a vector, then decode it,
    /// confirm the decoded vector is CLOSE to (not identical to) the original.
    #[test]
    fn test_encode_decode_round_trip_lossy() {
        let dim = 16;
        let m = 4;
        let k_per_subvector = 16;
        let mut rng = StdRng::seed_from_u64(1234);

        let mut data = VectorBatch::new(dim);
        for _ in 0..100 {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-2.0..2.0));
            }
            data.push(&v);
        }

        let config = PQConfig {
            m,
            k_per_subvector,
            max_iterations: 20,
        };

        let pq = train_pq(&data, &config, 999).unwrap();

        let original = data.get(0);
        let code = encode_vector(&pq, original).unwrap();
        let decoded = decode_vector(&pq, &code).unwrap();

        assert_eq!(decoded.len(), dim);

        let dist = euclidean_distance(original, &decoded);

        // Assert lossy: NOT identically zero
        assert!(
            dist > 1e-5,
            "Reconstruction should be lossy (dist > 1e-5), got {}",
            dist
        );
        // Assert close: distance is bounded
        assert!(dist < 3.0, "Reconstruction error too large: {}", dist);
    }

    /// Test 6: encode_batch() on a batch of 100 vectors produces exactly 100 codes, each of length m.
    #[test]
    fn test_encode_batch_100_vectors() {
        let dim = 16;
        let m = 4;
        let mut rng = StdRng::seed_from_u64(777);

        let mut data = VectorBatch::new(dim);
        for _ in 0..100 {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-1.0..1.0));
            }
            data.push(&v);
        }

        let config = PQConfig {
            m,
            k_per_subvector: 8,
            max_iterations: 15,
        };

        let pq = train_pq(&data, &config, 42).unwrap();
        let codes = encode_batch(&pq, &data).unwrap();

        assert_eq!(codes.len(), 100);
        for code in &codes {
            assert_eq!(code.len(), m);
        }
    }

    /// Test 7: Compression ratio sanity check:
    /// For a realistic config (dim=128, m=8, k_per_subvector=256), compute and print:
    /// - Original size per vector: 128 * 4 = 512 bytes
    /// - Compressed size per vector: 8 * 1 = 8 bytes
    /// - Compression ratio: 512/8 = 64x
    #[test]
    fn test_compression_ratio_sanity() {
        let dim = 128;
        let m = 8;
        let k_per_subvector = 256;

        let original_bytes = dim * std::mem::size_of::<f32>();
        let compressed_bytes = m * std::mem::size_of::<u8>();
        let compression_ratio = original_bytes as f64 / compressed_bytes as f64;

        let mut rng = StdRng::seed_from_u64(123);
        let mut dummy_batch = VectorBatch::new(dim);
        for _ in 0..k_per_subvector {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-1.0..1.0));
            }
            dummy_batch.push(&v);
        }
        let config = PQConfig {
            m,
            k_per_subvector,
            max_iterations: 1,
        };
        let pq = train_pq(&dummy_batch, &config, 1).unwrap();
        let code = encode_vector(&pq, dummy_batch.get(0)).unwrap();

        println!("\nPhase 21 Test 7: Compression ratio confirmation:");
        println!("  Original size per vector:   {} bytes", original_bytes);
        println!("  Compressed size per vector: {} bytes", compressed_bytes);
        println!("  Compression ratio:          {:.1}x", compression_ratio);
        println!("  Actual PQCode length:       {} bytes", code.len());

        assert_eq!(code.len(), compressed_bytes);
        assert_eq!(original_bytes, 512);
        assert_eq!(compressed_bytes, 8);
        assert_eq!(compression_ratio, 64.0);
    }

    /// Test 8: Realistic-scale test:
    /// Train codebooks on 1,000 vectors, encode_batch() all 1,000, confirm all codes are valid length,
    /// no panics, and print the AVERAGE round-trip reconstruction error.
    #[test]
    fn test_realistic_scale_encoding_and_reconstruction_error() {
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

        let pq = train_pq(&data, &config, 100).unwrap();

        let start_enc = Instant::now();
        let codes = encode_batch(&pq, &data).unwrap();
        let elapsed_enc = start_enc.elapsed();

        assert_eq!(codes.len(), n);

        let mut total_reconstruction_error = 0.0f64;
        for (i, code) in codes.iter().enumerate() {
            assert_eq!(code.len(), m);
            let original = data.get(i);
            let decoded = decode_vector(&pq, code).unwrap();
            total_reconstruction_error += euclidean_distance(original, &decoded) as f64;
        }

        let avg_reconstruction_error = total_reconstruction_error / (n as f64);

        println!(
            "\nPhase 21 Test 8: Realistic-scale PQ encoding & reconstruction (N={}, dim={}, m={}, k={}):",
            n, dim, m, k_per_subvector
        );
        println!(
            "  Encoding time for {} vectors: {:.2?} ({:.2} us/vec)",
            n,
            elapsed_enc,
            (elapsed_enc.as_secs_f64() * 1_000_000.0) / (n as f64)
        );
        println!(
            "  All {} vectors encoded to valid length {}",
            codes.len(),
            m
        );
        println!(
            "  Average round-trip reconstruction error: {:.4}",
            avg_reconstruction_error
        );

        // Sanity check that reconstruction error is reasonable
        assert!(avg_reconstruction_error > 0.0);
        assert!(avg_reconstruction_error < 10.0);
    }

    // ==========================================
    // Phase 22 Tests: Asymmetric Distance Computation (ADC)
    // ==========================================

    /// Test 1: build_adc_table() with wrong-dimension query returns Err.
    #[test]
    fn test_build_adc_table_wrong_dimension_returns_err() {
        let dim = 8;
        let mut data = VectorBatch::new(dim);
        for _ in 0..10 {
            data.push(&[1.0; 8]);
        }

        let config = PQConfig {
            m: 2,
            k_per_subvector: 2,
            max_iterations: 5,
        };
        let codebooks = train_pq(&data, &config, 42).unwrap();

        // Query length 7 != expected 8
        let query_wrong = [1.0; 7];
        let res = build_adc_table(&codebooks, &query_wrong);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("query dimension mismatch"),
            "Unexpected error message: {}",
            err
        );
    }

    /// Test 2: adc_distance() with wrong-length code returns Err.
    #[test]
    fn test_adc_distance_wrong_code_length_returns_err() {
        let dim = 8;
        let mut data = VectorBatch::new(dim);
        for _ in 0..10 {
            data.push(&[1.0; 8]);
        }

        let config = PQConfig {
            m: 4,
            k_per_subvector: 2,
            max_iterations: 5,
        };
        let codebooks = train_pq(&data, &config, 42).unwrap();
        let query = [0.5; 8];
        let table = build_adc_table(&codebooks, &query).unwrap();

        // Code length 3 != expected m=4
        let code_wrong: PQCode = vec![0, 1, 0];
        let res = adc_distance(&table, &code_wrong);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("PQCode length mismatch"),
            "Unexpected error message: {}",
            err
        );
    }

    /// Test 3: Correctness test:
    /// Build codebooks on hand-crafted 2-cluster-per-subvector data, build an ADC table
    /// for a specific query, and compute adc_distance() for an encoded vector.
    /// Compare this against manually computing squared euclidean distance between query
    /// and the decoded vector (using decode_vector()). Confirm both numbers are very close.
    #[test]
    fn test_adc_correctness_vs_decode_then_compute() {
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

        // Vector to encode and a full-precision query vector
        let v = [0.15, 0.15, 85.0, 75.0];
        let code = encode_vector(&pq, &v).unwrap();

        let query = [5.0, 4.0, 50.0, 60.0];
        let table = build_adc_table(&pq, &query).unwrap();

        // Method 1: ADC table lookup
        let adc_dist = adc_distance(&table, &code).unwrap();

        // Method 2: Decode vector then compute squared Euclidean distance
        let decoded = decode_vector(&pq, &code).unwrap();
        let decoded_dist = squared_euclidean_distance(&query, &decoded);

        println!(
            "\nPhase 22 Test 3: Correctness test (ADC lookup-table method vs decode-then-compute):"
        );
        println!(
            "  ADC distance (lookup-table method):          {:.6}",
            adc_dist
        );
        println!(
            "  Decoded squared Euclidean dist (decode-then): {:.6}",
            decoded_dist
        );
        println!(
            "  Absolute difference:                         {:.6e}",
            (adc_dist - decoded_dist).abs()
        );

        // Both methods measure the exact same squared distance
        assert!(
            (adc_dist - decoded_dist).abs() < 1e-5,
            "ADC distance {} and decoded distance {} mismatch beyond tolerance",
            adc_dist,
            decoded_dist
        );
    }

    /// Test 4: adc_distance_batch() on 50 codes returns exactly 50 distances,
    /// and spot-checks match adc_distance() called individually.
    #[test]
    fn test_adc_distance_batch_50_codes() {
        let dim = 16;
        let m = 4;
        let mut rng = StdRng::seed_from_u64(888);

        let mut data = VectorBatch::new(dim);
        for _ in 0..50 {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-2.0..2.0));
            }
            data.push(&v);
        }

        let config = PQConfig {
            m,
            k_per_subvector: 4,
            max_iterations: 10,
        };

        let pq = train_pq(&data, &config, 42).unwrap();
        let codes = encode_batch(&pq, &data).unwrap();

        let mut query = Vec::with_capacity(dim);
        for _ in 0..dim {
            query.push(rng.gen_range(-2.0..2.0));
        }

        let table = build_adc_table(&pq, &query).unwrap();
        let batch_dists = adc_distance_batch(&table, &codes).unwrap();

        assert_eq!(batch_dists.len(), 50);

        // Spot-check individual codes
        for &idx in &[0, 10, 25, 37, 49] {
            let single_dist = adc_distance(&table, &codes[idx]).unwrap();
            assert_eq!(
                batch_dists[idx], single_dist,
                "Batch distance mismatch at index {}",
                idx
            );
        }
    }

    /// Test 5: Squared-distance ordering test:
    /// Confirm that ranking encoded vectors by adc_distance() (squared) produces
    /// the identical order as ranking by sqrt(adc_distance()) (real distance).
    /// Validates the claim that .sqrt() is unnecessary overhead for top-k ranking.
    #[test]
    fn test_adc_squared_distance_ranking_order() {
        let dim = 32;
        let m = 4;
        let n = 60;
        let mut rng = StdRng::seed_from_u64(1234);

        let mut data = VectorBatch::new(dim);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-5.0..5.0));
            }
            data.push(&v);
        }

        let config = PQConfig {
            m,
            k_per_subvector: 8,
            max_iterations: 10,
        };

        let pq = train_pq(&data, &config, 99).unwrap();
        let codes = encode_batch(&pq, &data).unwrap();

        let mut query = Vec::with_capacity(dim);
        for _ in 0..dim {
            query.push(rng.gen_range(-5.0..5.0));
        }

        let table = build_adc_table(&pq, &query).unwrap();
        let sq_dists = adc_distance_batch(&table, &codes).unwrap();

        // Rank indices by squared distance
        let mut order_squared: Vec<usize> = (0..n).collect();
        order_squared.sort_by(|&a, &b| {
            sq_dists[a]
                .partial_cmp(&sq_dists[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Rank indices by sqrt(distance) (Euclidean distance)
        let mut order_sqrt: Vec<usize> = (0..n).collect();
        order_sqrt.sort_by(|&a, &b| {
            sq_dists[a]
                .sqrt()
                .partial_cmp(&sq_dists[b].sqrt())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        println!("\nPhase 22 Test 5: Top-5 Ranking comparison (squared vs sqrt):");
        for rank in 0..5 {
            let idx_sq = order_squared[rank];
            let idx_sqrt = order_sqrt[rank];
            println!(
                "  Rank #{}: squared_idx={}, dist²={:.4} | sqrt_idx={}, dist={:.4}",
                rank + 1,
                idx_sq,
                sq_dists[idx_sq],
                idx_sqrt,
                sq_dists[idx_sqrt].sqrt()
            );
        }

        assert_eq!(
            order_squared, order_sqrt,
            "Ranking by squared distance must be identical to ranking by sqrt distance"
        );
    }

    /// Test 6: Performance comparison test:
    /// On a realistic-scale setup (1,000 encoded vectors, dim=128, m=8, k=256),
    /// time how long it takes to compute distances to all 1,000 via adc_distance_batch()
    /// versus decoding all 1,000 vectors and computing real euclidean_distance().
    /// ADC should win clearly, demonstrating the computational advantage.
    #[test]
    fn test_adc_performance_comparison() {
        let n = 1000;
        let dim = 128;
        let m = 8;
        let k_per_subvector = 256;
        let mut rng = StdRng::seed_from_u64(42);

        let mut data = VectorBatch::new(dim);
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
            max_iterations: 5,
        };

        let pq = train_pq(&data, &config, 100).unwrap();
        let codes = encode_batch(&pq, &data).unwrap();

        let mut query = Vec::with_capacity(dim);
        for _ in 0..dim {
            query.push(rng.gen_range(-1.0..1.0));
        }

        // 1. Time ADC: build lookup table once, then run batch lookup
        let start_table = Instant::now();
        let table = build_adc_table(&pq, &query).unwrap();
        let elapsed_table = start_table.elapsed();

        let start_adc = Instant::now();
        let adc_dists = adc_distance_batch(&table, &codes).unwrap();
        let elapsed_adc = start_adc.elapsed();

        assert_eq!(adc_dists.len(), n);

        // 2. Time Decode-then-Compute: decode all codes and compute real Euclidean distance
        let start_decode = Instant::now();
        let mut decoded_dists = Vec::with_capacity(n);
        for code in &codes {
            let decoded = decode_vector(&pq, code).unwrap();
            decoded_dists.push(euclidean_distance(&query, &decoded));
        }
        let elapsed_decode = start_decode.elapsed();

        assert_eq!(decoded_dists.len(), n);

        let speedup = elapsed_decode.as_secs_f64() / elapsed_adc.as_secs_f64().max(1e-9);

        println!(
            "\nPhase 22 Test 6: Performance comparison (N={}, dim={}, m={}, k={}):",
            n, dim, m, k_per_subvector
        );
        println!("  ADC table build time:            {:.2?}", elapsed_table);
        println!(
            "  ADC batch distance computation:  {:.2?} ({:.2} us/vec)",
            elapsed_adc,
            (elapsed_adc.as_secs_f64() * 1_000_000.0) / (n as f64)
        );
        println!(
            "  Decode-then-compute computation: {:.2?} ({:.2} us/vec)",
            elapsed_decode,
            (elapsed_decode.as_secs_f64() * 1_000_000.0) / (n as f64)
        );
        println!("  Speedup factor (ADC vs Decode):  {:.2}x", speedup);

        // Sanity check: ADC lookup should be significantly faster than reconstructing floats and computing distance
        assert!(
            elapsed_adc < elapsed_decode,
            "ADC batch lookup ({:?}) should be faster than decode-then-compute ({:?})",
            elapsed_adc,
            elapsed_decode
        );
    }
}
