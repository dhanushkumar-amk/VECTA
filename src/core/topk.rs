// Heap-based top-k selection.
//
// Reusable by every search algorithm in vecta (brute-force, IVF, HNSW).
// Uses a fixed-size max-heap of size k for O(n log k) selection instead
// of O(n log n) full sort — matters when N is millions and k is 10.

use std::collections::BinaryHeap;

/// A candidate result: an external vector ID paired with its distance/similarity score.
#[derive(Debug, Clone, Copy)]
pub struct ScoredId {
    pub id: u64,
    pub score: f32,
}

/// Newtype wrapper giving `f32` a total ordering for use in `BinaryHeap`.
///
/// **NaN handling**: NaN is treated as greater than all finite values and +∞.
/// This means NaN scores will naturally sit at the top of a max-heap and get
/// evicted first, so they never pollute the final top-k result set.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Max-heap entry: wraps a `ScoredId` so the heap orders by `OrdF32(score)`.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct HeapEntry {
    score: OrdF32,
    id: u64,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Return the `k` candidates with the **smallest** scores, sorted ascending.
///
/// Uses a max-heap of size `k`: for each candidate, if its score is smaller
/// than the current worst (heap root), swap it in. This gives O(n log k)
/// selection instead of O(n log n) full sort.
///
/// Edge cases handled without panic:
/// - `k == 0` → empty vec
/// - `candidates` is empty → empty vec
/// - `k >= candidates.len()` → all candidates returned, sorted ascending
pub fn top_k_smallest(candidates: &[ScoredId], k: usize) -> Vec<ScoredId> {
    if k == 0 || candidates.is_empty() {
        return Vec::new();
    }

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);

    for c in candidates {
        let entry = HeapEntry {
            score: OrdF32(c.score),
            id: c.id,
        };

        if heap.len() < k {
            heap.push(entry);
        } else if let Some(worst) = heap.peek() {
            // Current worst (largest) is at the top of the max-heap.
            // Replace it only if the new candidate is strictly better (smaller).
            if entry.score < worst.score {
                heap.pop();
                heap.push(entry);
            }
        }
    }

    // Extract and sort ascending by score.
    let mut results: Vec<ScoredId> = heap
        .into_iter()
        .map(|e| ScoredId {
            id: e.id,
            score: e.score.0,
        })
        .collect();
    results.sort_by_key(|a| OrdF32(a.score));
    results
}

/// Return the `k` candidates with the **largest** scores, sorted descending.
///
/// Reuses [`top_k_smallest`] with negated scores to avoid duplicating
/// the heap logic.
pub fn top_k_largest(candidates: &[ScoredId], k: usize) -> Vec<ScoredId> {
    if k == 0 || candidates.is_empty() {
        return Vec::new();
    }

    // Negate scores so "largest original" becomes "smallest negated".
    let negated: Vec<ScoredId> = candidates
        .iter()
        .map(|c| ScoredId {
            id: c.id,
            score: -c.score,
        })
        .collect();

    let mut results = top_k_smallest(&negated, k);

    // Restore original scores. The ascending order of negated scores
    // (−9, −8, −7) becomes descending order of real scores (9, 8, 7)
    // — already correct, no reverse needed.
    for r in &mut results {
        r.score = -r.score;
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidates(scores: &[(u64, f32)]) -> Vec<ScoredId> {
        scores
            .iter()
            .map(|&(id, score)| ScoredId { id, score })
            .collect()
    }

    #[test]
    fn test_top_k_smallest_basic() {
        // 10 candidates, want top 3 smallest
        let candidates = make_candidates(&[
            (0, 5.0),
            (1, 2.0),
            (2, 8.0),
            (3, 1.0),
            (4, 9.0),
            (5, 3.0),
            (6, 7.0),
            (7, 4.0),
            (8, 6.0),
            (9, 0.5),
        ]);

        let result = top_k_smallest(&candidates, 3);
        assert_eq!(result.len(), 3);
        // Sorted ascending: 0.5, 1.0, 2.0
        assert_eq!(result[0].id, 9);
        assert!((result[0].score - 0.5).abs() < 1e-6);
        assert_eq!(result[1].id, 3);
        assert!((result[1].score - 1.0).abs() < 1e-6);
        assert_eq!(result[2].id, 1);
        assert!((result[2].score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_largest_basic() {
        let candidates = make_candidates(&[
            (0, 5.0),
            (1, 2.0),
            (2, 8.0),
            (3, 1.0),
            (4, 9.0),
            (5, 3.0),
            (6, 7.0),
            (7, 4.0),
            (8, 6.0),
            (9, 0.5),
        ]);

        let result = top_k_largest(&candidates, 3);
        assert_eq!(result.len(), 3);
        // Sorted descending: 9.0, 8.0, 7.0
        assert_eq!(result[0].id, 4);
        assert!((result[0].score - 9.0).abs() < 1e-6);
        assert_eq!(result[1].id, 2);
        assert!((result[1].score - 8.0).abs() < 1e-6);
        assert_eq!(result[2].id, 6);
        assert!((result[2].score - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_k_greater_than_len() {
        let candidates = make_candidates(&[(0, 3.0), (1, 1.0), (2, 2.0)]);
        let result = top_k_smallest(&candidates, 10);
        assert_eq!(result.len(), 3);
        // Should still be sorted ascending
        assert_eq!(result[0].id, 1);
        assert_eq!(result[1].id, 2);
        assert_eq!(result[2].id, 0);
    }

    #[test]
    fn test_k_zero() {
        let candidates = make_candidates(&[(0, 1.0), (1, 2.0)]);
        let result = top_k_smallest(&candidates, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_empty_candidates() {
        let result = top_k_smallest(&[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_correctness_vs_naive_sort() {
        // Generate 1000 deterministic pseudo-random candidates.
        let n = 1000;
        let k = 10;
        let mut rng: u64 = 123;
        let mut next_f32 = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32) / (u32::MAX as f32) * 1000.0
        };

        let candidates: Vec<ScoredId> = (0..n)
            .map(|i| ScoredId {
                id: i as u64,
                score: next_f32(),
            })
            .collect();

        // Heap-based
        let heap_result = top_k_smallest(&candidates, k);

        // Naive: full sort then take first k
        let mut sorted = candidates.clone();
        sorted.sort_by_key(|a| OrdF32(a.score));
        let naive_result = &sorted[..k];

        // Must match exactly (same IDs, same scores, same order).
        assert_eq!(heap_result.len(), k);
        for (h, n) in heap_result.iter().zip(naive_result.iter()) {
            assert_eq!(h.id, n.id, "ID mismatch: heap={}, naive={}", h.id, n.id);
            assert!(
                (h.score - n.score).abs() < 1e-6,
                "score mismatch for id {}: heap={}, naive={}",
                h.id,
                h.score,
                n.score
            );
        }
    }

    #[test]
    fn bench_heap_vs_naive_100k() {
        use std::time::Instant;

        let n = 100_000;
        let k = 10;
        let mut rng: u64 = 42;
        let mut next_f32 = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32) / (u32::MAX as f32) * 10000.0
        };

        let candidates: Vec<ScoredId> = (0..n)
            .map(|i| ScoredId {
                id: i as u64,
                score: next_f32(),
            })
            .collect();

        // Heap approach
        let start = Instant::now();
        let _heap_result = top_k_smallest(&candidates, k);
        let heap_elapsed = start.elapsed();

        // Naive sort approach
        let start = Instant::now();
        let mut sorted = candidates.clone();
        sorted.sort_by_key(|a| OrdF32(a.score));
        let _naive_result = &sorted[..k];
        let naive_elapsed = start.elapsed();

        println!(
            "\n[BENCH] top_k_smallest (k={k}, n={n}):\n  \
             heap:  {:.3}ms\n  \
             naive: {:.3}ms\n  \
             ratio: {:.1}x",
            heap_elapsed.as_secs_f64() * 1000.0,
            naive_elapsed.as_secs_f64() * 1000.0,
            naive_elapsed.as_secs_f64() / heap_elapsed.as_secs_f64()
        );
    }
}
