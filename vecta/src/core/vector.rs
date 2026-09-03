// Core vector math primitives.
//
// Every function here operates on plain &[f32] slices with zero heap allocation,
// zero PyO3 dependency, and is designed to be drop-in replaceable with a
// SIMD-optimized or BLAS-backed implementation later without changing its
// public signature.
//
// Dimension mismatches panic via assert_eq! — this is consistent with Rust
// conventions (like Index/SliceIndex) and avoids Result overhead in hot loops
// where callers are expected to guarantee matching dimensions.

/// Sum of element-wise products: Σ(a_i * b_i).
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product: dimension mismatch ({} vs {})", a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 (Euclidean) norm: sqrt(Σ(v_i²)).
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// L1 (Manhattan) norm: Σ|v_i|.
#[inline]
pub fn l1_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).sum()
}

/// Euclidean distance: sqrt(Σ(a_i - b_i)²).
///
/// Computed in a single pass — no intermediate vector is allocated.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "euclidean_distance: dimension mismatch ({} vs {})", a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity: dot(a, b) / (‖a‖₂ · ‖b‖₂).
///
/// Returns 0.0 if either vector has zero norm (avoids NaN / division by zero).
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: dimension mismatch ({} vs {})", a.len(), b.len());
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b;
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// In-place L2 normalization: v_i = v_i / ‖v‖₂.
///
/// If the vector has zero norm, it is left unchanged (no division by zero).
#[inline]
pub fn normalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm == 0.0 {
        return;
    }
    let inv_norm = 1.0 / norm;
    v.iter_mut().for_each(|x| *x *= inv_norm);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert two f32 values are approximately equal within epsilon.
    fn assert_approx(actual: f32, expected: f32, epsilon: f32) {
        assert!(
            (actual - expected).abs() < epsilon,
            "expected ≈ {expected}, got {actual} (ε = {epsilon})"
        );
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_approx(dot_product(&a, &b), 32.0, 1e-4);
    }

    #[test]
    fn test_l2_norm() {
        let v = [3.0_f32, 4.0];
        // sqrt(9 + 16) = sqrt(25) = 5
        assert_approx(l2_norm(&v), 5.0, 1e-4);
    }

    #[test]
    fn test_l1_norm() {
        let v = [3.0_f32, 4.0];
        // |3| + |4| = 7
        assert_approx(l1_norm(&v), 7.0, 1e-4);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [3.0_f32, 4.0];
        let b = [1.0_f32, 2.0];
        // sqrt((3-1)² + (4-2)²) = sqrt(4 + 4) = sqrt(8) ≈ 2.8284
        assert_approx(euclidean_distance(&a, &b), 8.0_f32.sqrt(), 1e-4);
    }

    #[test]
    fn test_cosine_similarity_positive() {
        let a = [3.0_f32, 4.0];
        let b = [1.0_f32, 2.0];
        // dot = 3 + 8 = 11, norm_a = 5, norm_b = sqrt(5) ≈ 2.2361
        // cos = 11 / (5 * 2.2361) ≈ 0.9839
        assert_approx(cosine_similarity(&a, &b), 0.9839, 1e-4);
    }

    #[test]
    fn test_cosine_similarity_negative() {
        let a = [3.0_f32, 4.0];
        let b = [-3.0_f32, 1.0];
        // dot = -9 + 4 = -5, norm_a = 5, norm_b = sqrt(9+1) = sqrt(10)
        // cos = -5 / (5 * sqrt(10)) ≈ -0.3162
        assert_approx(cosine_similarity(&a, &b), -0.3162, 1e-4);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = [3.0_f32, 4.0];
        let zero = [0.0_f32, 0.0];
        // Either vector being zero → return 0.0, not NaN
        let result = cosine_similarity(&a, &zero);
        assert!(!result.is_nan(), "cosine_similarity with zero vector must not be NaN");
        assert_approx(result, 0.0, 1e-4);
    }

    #[test]
    fn test_normalize() {
        let mut v = [3.0_f32, 4.0];
        normalize(&mut v);
        // After normalization, l2_norm should be ≈ 1.0
        assert_approx(l2_norm(&v), 1.0, 1e-4);
        // Individual components: 3/5 = 0.6, 4/5 = 0.8
        assert_approx(v[0], 0.6, 1e-4);
        assert_approx(v[1], 0.8, 1e-4);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_dimension_mismatch_panics() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0];
        dot_product(&a, &b);
    }
}
