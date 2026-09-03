//! Probabilistic layer assignment for HNSW.
//!
//! Controls the pyramid structure of the HNSW graph (many nodes at layer 0,
//! exponentially fewer at higher layers) following Malkov & Yashunin (2018).

use rand::Rng;

/// Configuration parameters for HNSW index construction and query traversal.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of bidirectional links per node per layer.
    ///
    /// # HNSW Convention:
    /// In accordance with the original HNSW paper, layer 0 conventionally allows
    /// up to `2 * m` bidirectional connections to guarantee high connectivity
    /// and recall at the ground layer where all vectors reside.
    pub m: usize,

    /// Size of dynamic candidate list during graph construction (insertion).
    /// Higher values yield higher graph quality and search recall at the cost of build time.
    pub ef_construction: usize,

    /// Size of dynamic candidate list during query search.
    /// Higher values yield higher search recall at the cost of latency.
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

/// Compute the scale factor `mL = 1.0 / ln(M)` for layer assignment.
///
/// # Panics
/// Panics if `m <= 1` since `ln(1) == 0` causes division by zero.
pub fn ml_factor(m: usize) -> f32 {
    assert!(
        m > 1,
        "HnswConfig::ml_factor: m must be greater than 1 (got {})",
        m
    );
    1.0 / (m as f32).ln()
}

/// Assign a maximum layer for a new node using the standard exponential decay distribution.
///
/// # Mathematical Formulation:
/// `layer = floor(-ln(u) * mL)` where `u ~ Uniform(0, 1]`.
///
/// Theoretical probability of landing on layer `L`:
/// `P(layer = L) = (M - 1) / M^(L + 1)`
/// For `M = 16`, Layer 0 accounts for `15/16 = 93.75%` of all inserted nodes.
///
/// # ln(0.0) Guarding Strategy:
/// Standard `rng.gen::<f32>()` produces `u` in `[0.0, 1.0)`. If `u == 0.0`, `-ln(0.0) == +inf`.
/// We guard against this by clamping `u` from below to `f32::EPSILON` (`~1.192e-7`),
/// guaranteeing `u > 0.0` without looping or discarding random entropy.
pub fn assign_layer(ml: f32, rng: &mut impl Rng) -> usize {
    let u: f32 = rng.gen();
    assign_layer_from_uniform(ml, u)
}

/// Deterministic layer assignment from a given uniform value `u`.
///
/// Factored out to verify math invariants and edge case guards in isolation.
#[inline]
pub fn assign_layer_from_uniform(ml: f32, u: f32) -> usize {
    // Guard: clamp u to [f32::EPSILON, 1.0] to prevent ln(0.0) = -inf or invalid logs
    let clamped_u = if u <= 0.0 {
        f32::EPSILON
    } else if u > 1.0 {
        1.0
    } else {
        u
    };

    let raw = (-clamped_u.ln() * ml).floor();
    if raw.is_nan() || raw < 0.0 {
        0
    } else {
        raw as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashMap;

    /// Test 1: ml_factor with m=16 returns approx 1/ln(16) ≈ 0.36067.
    #[test]
    fn test_ml_factor_m16() {
        let ml = ml_factor(16);
        let expected = 1.0 / (16.0f32).ln();
        assert!((ml - expected).abs() < 1e-6);
        assert!((ml - 0.36067376).abs() < 1e-4);
    }

    /// Test 2: ml_factor with m <= 1 panics with clear error message.
    #[test]
    #[should_panic(expected = "m must be greater than 1")]
    fn test_ml_factor_m1_panics() {
        let _ = ml_factor(1);
    }

    #[test]
    #[should_panic(expected = "m must be greater than 1")]
    fn test_ml_factor_m0_panics() {
        let _ = ml_factor(0);
    }

    /// Test 3: assign_layer with a fixed seeded RNG returns deterministic, reproducible values.
    #[test]
    fn test_assign_layer_reproducibility() {
        let ml = ml_factor(16);

        let mut rng1 = StdRng::seed_from_u64(12345);
        let run1: Vec<usize> = (0..50).map(|_| assign_layer(ml, &mut rng1)).collect();

        let mut rng2 = StdRng::seed_from_u64(12345);
        let run2: Vec<usize> = (0..50).map(|_| assign_layer(ml, &mut rng2)).collect();

        assert_eq!(run1, run2);
    }

    /// Test 4: Statistical distribution test over 100,000 draws with m=16.
    /// Confirms the pyramid shape:
    /// - Theoretical Layer 0 probability: (16 - 1) / 16 = 93.75%
    /// - Counts strictly decrease across layers
    /// - Prints full distribution histogram
    #[test]
    fn test_statistical_layer_distribution_100k() {
        let n = 100_000;
        let m = 16;
        let ml = ml_factor(m);

        let mut rng = StdRng::seed_from_u64(42);
        let mut histogram: HashMap<usize, usize> = HashMap::new();

        for _ in 0..n {
            let layer = assign_layer(ml, &mut rng);
            *histogram.entry(layer).or_insert(0) += 1;
        }

        // Theoretical expectations for M=16:
        // P(L = 0) = 15/16 = 0.9375 (93.75%)
        // P(L = 1) = 15/256 ≈ 0.05859 (5.86%)
        // P(L = 2) = 15/4096 ≈ 0.00366 (0.366%)
        let layer0_count = *histogram.get(&0).unwrap_or(&0);
        let layer0_fraction = (layer0_count as f64) / (n as f64);
        let theoretical_layer0 = 15.0 / 16.0;

        let max_observed_layer = *histogram.keys().max().unwrap_or(&0);

        println!(
            "\nPhase 16 Test 4: HNSW Layer Histogram (N={}, M={}):",
            n, m
        );
        println!(
            "  Theoretical Layer 0 fraction: {:.4} ({:.2}%)",
            theoretical_layer0,
            theoretical_layer0 * 100.0
        );
        println!(
            "  Empirical   Layer 0 fraction: {:.4} ({:.2}%)",
            layer0_fraction,
            layer0_fraction * 100.0
        );
        println!("--------------------------------------------------");
        for l in 0..=max_observed_layer {
            let count = *histogram.get(&l).unwrap_or(&0);
            let pct = (count as f64 / n as f64) * 100.0;
            println!("  Layer {:>2}: {:>7} nodes ({:>6.3}%)", l, count, pct);
        }
        println!("--------------------------------------------------");

        // 1. Assert layer 0 is within 0.5% tolerance of theoretical 93.75%
        assert!(
            (layer0_fraction - theoretical_layer0).abs() < 0.005,
            "Layer 0 fraction {:.4} differs from theoretical {:.4}",
            layer0_fraction,
            theoretical_layer0
        );

        // 2. Assert strictly decreasing counts for populated layers with substantial samples
        for l in 0..3 {
            let curr = *histogram.get(&l).unwrap_or(&0);
            let next = *histogram.get(&(l + 1)).unwrap_or(&0);
            assert!(
                curr > next,
                "Layer count must strictly decrease: Layer {} ({}) <= Layer {} ({})",
                l,
                curr,
                l + 1,
                next
            );
        }
    }

    /// Test 5: Edge cases in assign_layer_from_uniform never panic or return NaN/invalid layer.
    #[test]
    fn test_assign_layer_edge_cases_and_nan_safety() {
        let ml = ml_factor(16);

        // Exact 0.0 guarded via epsilon -> valid finite layer
        let l_zero = assign_layer_from_uniform(ml, 0.0);
        assert!(l_zero < 100);

        // Negative value clamped to epsilon
        let l_neg = assign_layer_from_uniform(ml, -0.5);
        assert!(l_neg < 100);

        // Exact 1.0 -> -ln(1.0) = 0.0 -> layer 0
        let l_one = assign_layer_from_uniform(ml, 1.0);
        assert_eq!(l_one, 0);

        // Greater than 1.0 clamped to 1.0 -> layer 0
        let l_large = assign_layer_from_uniform(ml, 2.5);
        assert_eq!(l_large, 0);
    }
}
