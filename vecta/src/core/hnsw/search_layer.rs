//! Greedy single-layer search primitive for HNSW graph traversal.
//!
//! Implements Algorithm 2 (SEARCH-LAYER) from Malkov & Yashunin (2018).
//! Reusable across both insertion (Phase 17) and multi-layer query search (Phase 18).

use std::collections::{BinaryHeap, HashSet};

use crate::core::hnsw::graph::HnswGraph;
use crate::core::topk::ScoredId;
use crate::core::vector::euclidean_distance;

#[derive(Copy, Clone, Debug)]
struct MinCandidate {
    dist: f32,
    idx: usize,
}

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist).is_eq() && self.idx == other.idx
    }
}

impl Eq for MinCandidate {}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering so BinaryHeap acts as a min-heap (smallest dist at root)
        other.dist.total_cmp(&self.dist)
    }
}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Copy, Clone, Debug)]
struct MaxCandidate {
    dist: f32,
    idx: usize,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist).is_eq() && self.idx == other.idx
    }
}

impl Eq for MaxCandidate {}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Natural ordering: largest dist at root for max-heap
        self.dist.total_cmp(&other.dist)
    }
}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Greedy search within a single layer of an HNSW graph.
///
/// Starting from `entry_points`, explores neighbors at `layer` to find and return
/// the `ef` closest nodes to `query`.
///
/// # Arguments
/// * `graph` - The HNSW graph structure.
/// * `query` - Query vector slice.
/// * `entry_points` - Slice of internal node indices to seed the search from.
/// * `layer` - The specific graph layer to traverse.
/// * `ef` - Exploration factor: maximum size of dynamic candidate set to maintain/return.
///
/// # Return
/// A list of [`ScoredId`] sorted ascending by distance (nearest first).
/// The `id` field contains the **internal node index** (`usize` cast to `u64`),
/// not the external user ID.
///
/// # Metric Assumption Note:
/// For v1 correctness, `greedy_search_layer` calculates distances using `euclidean_distance` (L2).
/// In future optimization passes, this should dispatch based on `graph.metric` (Cosine, DotProduct).
/// Using L2 provides consistent Euclidean Voronoi neighborhood properties during graph building.
pub fn greedy_search_layer(
    graph: &HnswGraph,
    query: &[f32],
    entry_points: &[usize],
    layer: usize,
    ef: usize,
) -> Vec<ScoredId> {
    if ef == 0 || entry_points.is_empty() || graph.is_empty() {
        return Vec::new();
    }

    let mut visited: HashSet<usize> = HashSet::with_capacity(ef * 4);
    let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
    let mut best_found: BinaryHeap<MaxCandidate> = BinaryHeap::with_capacity(ef + 1);

    // Initialize with entry points
    for &ep in entry_points {
        if ep < graph.nodes.len() && visited.insert(ep) {
            let dist = euclidean_distance(query, graph.get_vector(ep));
            candidates.push(MinCandidate { dist, idx: ep });
            best_found.push(MaxCandidate { dist, idx: ep });
            if best_found.len() > ef {
                best_found.pop();
            }
        }
    }

    // Graph traversal
    while let Some(curr) = candidates.pop() {
        // Early termination: if the closest candidate is farther than the worst
        // candidate in best_found, and best_found is full, no further improvement is possible.
        if best_found.len() >= ef {
            let worst_dist = best_found.peek().unwrap().dist;
            if curr.dist > worst_dist {
                break;
            }
        }

        // Explore neighbors at the specified layer
        for &neighbor in graph.get_neighbors(curr.idx, layer) {
            let neighbor_idx = neighbor as usize;
            if neighbor_idx < graph.nodes.len() && visited.insert(neighbor_idx) {
                let dist = euclidean_distance(query, graph.get_vector(neighbor_idx));

                let worst_dist = if best_found.len() < ef {
                    f32::INFINITY
                } else {
                    best_found.peek().unwrap().dist
                };

                if dist < worst_dist || best_found.len() < ef {
                    candidates.push(MinCandidate {
                        dist,
                        idx: neighbor_idx,
                    });
                    best_found.push(MaxCandidate {
                        dist,
                        idx: neighbor_idx,
                    });
                    if best_found.len() > ef {
                        best_found.pop();
                    }
                }
            }
        }
    }

    // Convert best_found to sorted Vec<ScoredId> (nearest first)
    let mut results: Vec<ScoredId> = best_found
        .into_iter()
        .map(|c| ScoredId {
            id: c.idx as u64,
            score: c.dist,
        })
        .collect();

    results.sort_by(|a, b| a.score.total_cmp(&b.score));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flat_index::Metric;
    use crate::core::hnsw::graph::HnswNode;

    #[test]
    fn test_greedy_search_layer_on_small_graph() {
        let dim = 2;
        let mut graph = HnswGraph::new(dim, Metric::Euclidean);

        // Vector coordinates:
        // 0: [0.0, 0.0]
        // 1: [1.0, 0.0]
        // 2: [2.0, 0.0]
        // 3: [10.0, 0.0] (far away)
        graph.vectors.push(&[0.0, 0.0]);
        graph.vectors.push(&[1.0, 0.0]);
        graph.vectors.push(&[2.0, 0.0]);
        graph.vectors.push(&[10.0, 0.0]);

        // Connect a line at layer 0: 0 <-> 1 <-> 2 <-> 3
        graph.nodes.push(HnswNode {
            id: 0,
            vector_idx: 0,
            max_layer: 0,
            neighbors: vec![vec![1]],
        });
        graph.nodes.push(HnswNode {
            id: 1,
            vector_idx: 1,
            max_layer: 0,
            neighbors: vec![vec![0, 2]],
        });
        graph.nodes.push(HnswNode {
            id: 2,
            vector_idx: 2,
            max_layer: 0,
            neighbors: vec![vec![1, 3]],
        });
        graph.nodes.push(HnswNode {
            id: 3,
            vector_idx: 3,
            max_layer: 0,
            neighbors: vec![vec![2]],
        });

        // Query near node 2: [2.1, 0.0], start at entry point 0
        let query = [2.1, 0.0];
        let results = greedy_search_layer(&graph, &query, &[0], 0, 2);

        assert_eq!(results.len(), 2);
        // Nearest is node 2, second nearest is node 1
        assert_eq!(results[0].id, 2);
        assert_eq!(results[1].id, 1);
        assert!(results[0].score < results[1].score);
    }
}
