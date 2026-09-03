//! HNSW graph insertion algorithm.
//!
//! Implements Algorithm 1 (INSERT) from Malkov & Yashunin (2018).
//! Connects new nodes via top-down hierarchical descent, greedy search,
//! and bidirectional edge assignment with bounded neighbor pruning.

use rand::Rng;

use crate::core::hnsw::graph::{HnswGraph, HnswNode};
use crate::core::hnsw::layer::{assign_layer, ml_factor, HnswConfig};
use crate::core::hnsw::search_layer::greedy_search_layer;
use crate::core::topk::ScoredId;
use crate::core::vector::euclidean_distance;

/// Select up to `m` neighbors from a list of candidate nodes.
///
/// # Heuristic Note:
/// Implements the **simple selection heuristic** (Malkov & Yashunin 2018),
/// which selects the `m` closest candidates ordered by distance ascending.
/// The more advanced heuristic selection (which balances distance and geometric
/// diversity to avoid redundant parallel edges) is a valid future optimization,
/// but out of scope for v1 correctness.
///
/// Returns the internal node indices of the selected neighbors.
pub fn select_neighbors(candidates: &[ScoredId], m: usize) -> Vec<usize> {
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.score.total_cmp(&b.score));
    sorted.into_iter().take(m).map(|c| c.id as usize).collect()
}

/// Prune a node's neighbor list at `layer` back down to `max_m` closest connections.
fn prune_neighbors(graph: &mut HnswGraph, node_idx: usize, layer: usize, max_m: usize) {
    if graph.nodes[node_idx].neighbors[layer].len() <= max_m {
        return;
    }

    let nbr_indices: Vec<usize> = graph.nodes[node_idx].neighbors[layer]
        .iter()
        .map(|&x| x as usize)
        .collect();

    let node_vec_idx = graph.nodes[node_idx].vector_idx;
    let node_vec = graph.vectors.get(node_vec_idx);

    let candidates: Vec<ScoredId> = nbr_indices
        .into_iter()
        .map(|nbr| {
            let nbr_vec = graph.vectors.get(graph.nodes[nbr].vector_idx);
            let dist = euclidean_distance(node_vec, nbr_vec);
            ScoredId {
                id: nbr as u64,
                score: dist,
            }
        })
        .collect();

    let selected = select_neighbors(&candidates, max_m);
    graph.nodes[node_idx].neighbors[layer] = selected.into_iter().map(|idx| idx as u32).collect();
}

/// Insert a single vector with external `id` into the HNSW graph.
///
/// # Algorithm Steps:
/// 1. Validates that `id` is not duplicate (returns `Err`), and `vector.len() == graph.dim` (panics).
/// 2. Pushes vector into `graph.vectors`.
/// 3. Assigns probabilistic `max_layer` using [`assign_layer`].
/// 4. **Empty graph**: If graph has no nodes yet, initializes `new_node` as `graph.entry_point` and returns.
/// 5. **Normal case**:
///    - Descends greedily (`ef = 1`) from the current entry point down to `new_node.max_layer + 1`.
///    - For layers from `min(top_layer, new_node.max_layer)` down to `0`:
///      - Finds candidate neighbors via [`greedy_search_layer`] with `ef_construction`.
///      - Selects top `m` (or `2*m` at layer 0) neighbors via [`select_neighbors`].
///      - Adds bidirectional edges and prunes neighbors whose lists exceed `max_m`.
/// 6. Updates `graph.entry_point` if `new_node.max_layer` exceeds the current top layer.
///
/// # Errors
/// Returns `Err` if `id` already exists in the graph.
///
/// # Panics
/// Panics if `vector.len() != graph.dim`.
pub fn insert(
    graph: &mut HnswGraph,
    id: u64,
    vector: &[f32],
    config: &HnswConfig,
    rng: &mut impl Rng,
) -> Result<(), String> {
    // 1. Validate external ID
    if graph.id_to_index.contains_key(&id) {
        return Err(format!("HnswGraph::insert: duplicate id {}", id));
    }

    // 2. Validate vector dimension
    assert_eq!(
        vector.len(),
        graph.dim,
        "HnswGraph::insert: vector dimension {} != graph dimension {}",
        vector.len(),
        graph.dim
    );

    // 3. Append vector to flat storage
    let vector_idx = graph.vectors.len();
    graph.vectors.push(vector);

    // 4. Assign max_layer probabilistically
    let ml = ml_factor(config.m);
    let new_node_max_layer = assign_layer(ml, rng);

    // 5. Check if graph is empty
    if graph.is_empty() {
        let node_idx = 0;
        let node = HnswNode {
            id,
            vector_idx,
            max_layer: new_node_max_layer,
            neighbors: vec![Vec::new(); new_node_max_layer + 1],
        };
        graph.nodes.push(node);
        graph.id_to_index.insert(id, node_idx);
        graph.entry_point = Some(node_idx);
        return Ok(());
    }

    // 6. Normal insertion into non-empty graph
    let new_node_idx = graph.nodes.len();
    let new_node = HnswNode {
        id,
        vector_idx,
        max_layer: new_node_max_layer,
        neighbors: vec![Vec::new(); new_node_max_layer + 1],
    };
    graph.nodes.push(new_node);
    graph.id_to_index.insert(id, new_node_idx);

    let curr_ep = graph.entry_point.unwrap();
    let curr_max_layer = graph.nodes[curr_ep].max_layer;

    let mut eps = vec![curr_ep];

    // Phase A: Greedy descent from curr_max_layer down to new_node_max_layer + 1
    if curr_max_layer > new_node_max_layer {
        for l in (new_node_max_layer + 1..=curr_max_layer).rev() {
            let best = greedy_search_layer(graph, vector, &eps, l, 1);
            if let Some(first) = best.first() {
                eps = vec![first.id as usize];
            }
        }
    }

    // Phase B: Connecting layers from min(curr_max_layer, new_node_max_layer) down to layer 0
    let start_layer = curr_max_layer.min(new_node_max_layer);
    for l in (0..=start_layer).rev() {
        let candidates = greedy_search_layer(graph, vector, &eps, l, config.ef_construction);

        // HNSW convention: layer 0 allows 2*m connections, higher layers allow m
        let max_m = if l == 0 { 2 * config.m } else { config.m };

        let selected_neighbors = select_neighbors(&candidates, max_m);

        // Set forward connections
        graph.nodes[new_node_idx].neighbors[l] =
            selected_neighbors.iter().map(|&idx| idx as u32).collect();

        // Set backward connections and prune
        for &nbr in &selected_neighbors {
            if !graph.nodes[nbr].neighbors[l].contains(&(new_node_idx as u32)) {
                graph.nodes[nbr].neighbors[l].push(new_node_idx as u32);
                prune_neighbors(graph, nbr, l, max_m);
            }
        }

        // Candidates become entry points for the next lower layer
        eps = candidates.into_iter().map(|c| c.id as usize).collect();
    }

    // 7. Update global entry point if new node is taller than current entry point
    if new_node_max_layer > curr_max_layer {
        graph.entry_point = Some(new_node_idx);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flat_index::Metric;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashMap;

    /// Test 1: Inserting into an empty graph sets entry_point and creates node.
    #[test]
    fn test_insert_empty_graph() {
        let mut graph = HnswGraph::new(3, Metric::Euclidean);
        let config = HnswConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        let res = insert(&mut graph, 100, &[1.0, 2.0, 3.0], &config, &mut rng);
        assert!(res.is_ok());
        assert_eq!(graph.len(), 1);
        assert!(!graph.is_empty());
        assert_eq!(graph.entry_point, Some(0));
        assert_eq!(graph.internal_index_of(100), Some(0));

        let node = &graph.nodes[0];
        assert_eq!(node.id, 100);
        assert_eq!(node.neighbors.len(), node.max_layer + 1);
        for l in 0..=node.max_layer {
            assert!(node.neighbors[l].is_empty());
        }
    }

    /// Test 2: Inserting duplicate ID returns Err and preserves graph state.
    #[test]
    fn test_insert_duplicate_id_returns_err() {
        let mut graph = HnswGraph::new(2, Metric::Euclidean);
        let config = HnswConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        assert!(insert(&mut graph, 42, &[1.0, 2.0], &config, &mut rng).is_ok());
        assert_eq!(graph.len(), 1);

        let res = insert(&mut graph, 42, &[3.0, 4.0], &config, &mut rng);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "HnswGraph::insert: duplicate id 42");
        assert_eq!(graph.len(), 1);
    }

    /// Test 3: Inserting wrong dimension panics with clear message.
    #[test]
    #[should_panic(expected = "vector dimension 3 != graph dimension 2")]
    fn test_insert_wrong_dimension_panics() {
        let mut graph = HnswGraph::new(2, Metric::Euclidean);
        let config = HnswConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        let _ = insert(&mut graph, 1, &[1.0, 2.0, 3.0], &config, &mut rng);
    }

    /// Test 4: Hand-verifiable small test with 6 well-separated 2D points.
    /// Confirms that layer-0 neighbors connect to geometrically close points.
    #[test]
    fn test_hand_verified_geometric_connections() {
        let mut graph = HnswGraph::new(2, Metric::Euclidean);
        let config = HnswConfig {
            m: 2,
            ef_construction: 50,
            ef_search: 20,
        };
        let mut rng = StdRng::seed_from_u64(1234);

        // Two distinct clusters:
        // Cluster A (near (1,1)): (1, [1.0, 1.0]), (2, [1.0, 1.5]), (3, [1.5, 1.0])
        // Cluster B (near (9,9)): (4, [9.0, 9.0]), (5, [9.0, 8.5]), (6, [8.5, 9.0])
        let points: Vec<(u64, [f32; 2])> = vec![
            (1, [1.0, 1.0]),
            (2, [1.0, 1.5]),
            (3, [1.5, 1.0]),
            (4, [9.0, 9.0]),
            (5, [9.0, 8.5]),
            (6, [8.5, 9.0]),
        ];

        for (id, pt) in &points {
            insert(&mut graph, *id, pt, &config, &mut rng).unwrap();
        }

        assert_eq!(graph.len(), 6);

        println!("\nPhase 17 Test 4: Hand-verified 6-point layer-0 connectivity:");
        for (i, (_, pt)) in points.iter().enumerate() {
            let nbrs: Vec<u64> = graph.nodes[i].neighbors[0]
                .iter()
                .map(|&idx| graph.nodes[idx as usize].id)
                .collect();
            println!(
                "  Node {} (id={}, pt={:?}) -> layer-0 neighbors: {:?}",
                i, graph.nodes[i].id, pt, nbrs
            );

            // Cluster A nodes (id 1, 2, 3) should have neighbors primarily from Cluster A (ids 1, 2, 3)
            if graph.nodes[i].id <= 3 {
                let has_cluster_a = nbrs.iter().any(|&id| id <= 3);
                assert!(
                    has_cluster_a,
                    "Cluster A node should connect to another Cluster A point"
                );
            } else {
                let has_cluster_b = nbrs.iter().any(|&id| id >= 4);
                assert!(
                    has_cluster_b,
                    "Cluster B node should connect to another Cluster B point"
                );
            }
        }
    }

    /// Test 5: Bidirectionality test:
    /// In a small graph without pruning overflow, if A links to B at layer L, B links to A.
    #[test]
    fn test_bidirectionality_edges() {
        let mut graph = HnswGraph::new(2, Metric::Euclidean);
        // Set m large enough (m=8) so no edges are pruned across 4 nodes
        let config = HnswConfig {
            m: 8,
            ef_construction: 50,
            ef_search: 20,
        };
        let mut rng = StdRng::seed_from_u64(999);

        let points = [
            (1, [0.0, 0.0]),
            (2, [1.0, 0.0]),
            (3, [0.0, 1.0]),
            (4, [1.0, 1.0]),
        ];

        for (id, pt) in &points {
            insert(&mut graph, *id, pt, &config, &mut rng).unwrap();
        }

        // Check layer 0 bidirectionality across all connected pairs
        for a_idx in 0..graph.len() {
            for &b_u32 in &graph.nodes[a_idx].neighbors[0] {
                let b_idx = b_u32 as usize;
                let b_nbrs = &graph.nodes[b_idx].neighbors[0];
                assert!(
                    b_nbrs.contains(&(a_idx as u32)),
                    "Edge {} -> {} exists, but {} does not link back to {}",
                    a_idx,
                    b_idx,
                    b_idx,
                    a_idx
                );
            }
        }
    }

    /// Test 6: Neighbor list capacity test:
    /// Ensures that even under heavy insertion with small m, no node exceeds max_m (2m at layer 0, m at layer > 0).
    #[test]
    fn test_neighbor_list_capacity_cap() {
        let m = 3;
        let mut graph = HnswGraph::new(2, Metric::Euclidean);
        let config = HnswConfig {
            m,
            ef_construction: 50,
            ef_search: 20,
        };
        let mut rng = StdRng::seed_from_u64(777);

        // Insert 30 points into small-m graph
        for i in 0..30 {
            let pt = [(i as f32) * 0.5, (i as f32) * 0.3];
            insert(&mut graph, i as u64, &pt, &config, &mut rng).unwrap();
        }

        let max_l0 = 2 * m;
        for node in &graph.nodes {
            for (l, nbrs) in node.neighbors.iter().enumerate() {
                let cap = if l == 0 { max_l0 } else { m };
                assert!(
                    nbrs.len() <= cap,
                    "Node {} at layer {} has {} neighbors, exceeding cap of {}",
                    node.id,
                    l,
                    nbrs.len(),
                    cap
                );
            }
        }
    }

    /// Test 7: entry_point points to the node with the highest max_layer.
    #[test]
    fn test_entry_point_correctness() {
        let mut graph = HnswGraph::new(2, Metric::Euclidean);
        let config = HnswConfig::default();
        let mut rng = StdRng::seed_from_u64(12345);

        for i in 0..50 {
            let pt = [i as f32, (i * 2) as f32];
            insert(&mut graph, i as u64, &pt, &config, &mut rng).unwrap();
        }

        let ep = graph.entry_point.expect("Entry point must be set");
        let ep_max_layer = graph.nodes[ep].max_layer;

        // Entry point must have the maximum layer of any node in the graph
        for (idx, node) in graph.nodes.iter().enumerate() {
            assert!(
                ep_max_layer >= node.max_layer,
                "Node {} has layer {} > entry_point {} (layer {})",
                idx,
                node.max_layer,
                ep,
                ep_max_layer
            );
        }
    }

    /// Test 8: Large-scale smoke test with 500 vectors of 128 dimensions.
    #[test]
    fn test_large_scale_500_vectors_insertion() {
        let n = 500;
        let dim = 128;
        let mut graph = HnswGraph::new(dim, Metric::Euclidean);
        let config = HnswConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        for id in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-10.0..10.0));
            }
            insert(&mut graph, id as u64, &v, &config, &mut rng).unwrap();
        }

        assert_eq!(graph.len(), n);
        assert!(!graph.is_empty());
        assert!(graph.entry_point.is_some());

        // Count max_layer distribution
        let mut layer_counts: HashMap<usize, usize> = HashMap::new();
        for node in &graph.nodes {
            *layer_counts.entry(node.max_layer).or_insert(0) += 1;
        }

        let max_observed = *layer_counts.keys().max().unwrap_or(&0);

        println!("\nPhase 17 Test 8: 500-vector HNSW graph max_layer distribution:");
        for l in 0..=max_observed {
            let count = *layer_counts.get(&l).unwrap_or(&0);
            let pct = (count as f64 / n as f64) * 100.0;
            println!("  Layer {}: {:>4} nodes ({:>5.1}%)", l, count, pct);
        }

        // Layer 0 must be the vast majority (> 80%)
        let l0 = *layer_counts.get(&0).unwrap_or(&0);
        assert!(l0 > (n * 80 / 100));
    }
}
