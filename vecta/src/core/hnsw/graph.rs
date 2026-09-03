//! HNSW (Hierarchical Navigable Small World) graph data structure.
//!
//! Provides the foundational per-layer adjacency graph storage for HNSW indexing.
//!
//! # Architecture & Invariants:
//! 1. **Sparse Multi-Layer Adjacency**: Each node maintains an adjacency list per layer
//!    from `0` up to `max_layer`.
//! 2. **Internal Index vs. External ID**:
//!    - `HnswNode.id`: The caller-provided external `u64` ID (e.g. database primary key).
//!    - `HnswNode.neighbors`: Stores **internal indices** (`u32`) referring to positions
//!      in `HnswGraph.nodes`. This provides $O(1)$ direct array addressing during graph traversal.
//! 3. **Single Vector Storage**: Vector coordinate buffers are never duplicated inside nodes.
//!    All vectors reside in a shared [`VectorBatch`], indexed by `node.vector_idx`.
//! 4. **Layer Memory Invariant**: `neighbors.len() == max_layer + 1`. Nodes allocate
//!    adjacency lists exclusively for the layers in which they participate.

use std::collections::HashMap;

use crate::core::batch::VectorBatch;
use crate::core::flat_index::Metric;

/// A single node in the HNSW hierarchy.
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// External identifier (e.g., entity ID, document ID).
    pub id: u64,
    /// Offset into the graph's contiguous [`VectorBatch`].
    pub vector_idx: usize,
    /// Highest layer in which this node exists (0-indexed).
    pub max_layer: usize,
    /// Neighbor lists per layer: `neighbors[layer]` is a list of internal node indices.
    ///
    /// # Critical Invariants:
    /// - Elements in `neighbors[layer]` are **internal node indices** (`u32`),
    ///   referencing positions within `HnswGraph.nodes` — NOT external `id`s.
    /// - `neighbors.len() == max_layer + 1` (contains layer 0 through `max_layer`).
    pub neighbors: Vec<Vec<u32>>,
}

/// The multi-layer hierarchical graph structure for HNSW.
#[derive(Debug, Clone)]
pub struct HnswGraph {
    /// Internal contiguous node array, indexed directly by internal index (`usize`).
    pub nodes: Vec<HnswNode>,
    /// Map from external ID (`u64`) to internal node index (`usize`).
    pub id_to_index: HashMap<u64, usize>,
    /// Contiguous flat vector store: vector for node `i` is at `vectors.get(nodes[i].vector_idx)`.
    pub vectors: VectorBatch,
    /// Internal index of the current top-layer entry point node, or `None` if graph is empty.
    pub entry_point: Option<usize>,
    /// Dimensionality of vectors stored in the graph.
    pub dim: usize,
    /// Distance or similarity metric used for vector comparisons.
    pub metric: Metric,
}

impl HnswGraph {
    /// Create a new, empty HNSW graph.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            nodes: Vec::new(),
            id_to_index: HashMap::new(),
            vectors: VectorBatch::new(dim),
            entry_point: None,
            dim,
            metric,
        }
    }

    /// Return the total number of nodes in the graph.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return `true` if the graph contains no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return the dimensionality of vectors in this graph.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the distance metric used by this graph.
    #[inline]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Return the internal index of the top-layer entry point, if any.
    #[inline]
    pub fn entry_point(&self) -> Option<usize> {
        self.entry_point
    }

    /// Retrieve the slice of vector coordinates for node at `internal_idx`.
    ///
    /// # Panics
    /// Panics if `internal_idx >= self.nodes.len()`.
    #[inline]
    pub fn get_vector(&self, internal_idx: usize) -> &[f32] {
        let vec_idx = self.nodes[internal_idx].vector_idx;
        self.vectors.get(vec_idx)
    }

    /// Return the list of neighbor internal indices for node at `internal_idx` at `layer`.
    ///
    /// # Out-of-Range Handling
    /// If `layer > nodes[internal_idx].max_layer`, returns an empty slice `&[]` without panicking.
    /// This allows callers during hierarchical search to query layers safely without
    /// pre-checking node maximum layers.
    ///
    /// # Panics
    /// Panics only if `internal_idx >= self.nodes.len()` (invalid node index).
    #[inline]
    pub fn get_neighbors(&self, internal_idx: usize, layer: usize) -> &[u32] {
        let node = &self.nodes[internal_idx];
        if layer <= node.max_layer && layer < node.neighbors.len() {
            &node.neighbors[layer]
        } else {
            &[]
        }
    }

    /// Look up the internal node index for a given external `id`.
    ///
    /// Returns `Some(internal_idx)` if present, or `None` if the ID is not in the graph.
    #[inline]
    pub fn internal_index_of(&self, id: u64) -> Option<usize> {
        self.id_to_index.get(&id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: HnswGraph::new() produces an empty graph with None entry point.
    #[test]
    fn test_new_graph_is_empty() {
        let graph = HnswGraph::new(128, Metric::Euclidean);
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
        assert_eq!(graph.dim(), 128);
        assert_eq!(graph.metric(), Metric::Euclidean);
        assert_eq!(graph.entry_point(), None);
        assert_eq!(graph.vectors.len(), 0);
        assert!(graph.id_to_index.is_empty());
    }

    /// Test 2: Manually construct a small 3-node graph across 2 layers.
    /// Confirms neighbor lookups, vector retrieval, and ID mapping.
    #[test]
    fn test_manual_multi_layer_graph_construction() {
        let dim = 2;
        let mut graph = HnswGraph::new(dim, Metric::Euclidean);

        // Add 3 vectors into flat VectorBatch
        // Node 0: id=100, vector=[1.0, 2.0], max_layer=1
        // Node 1: id=200, vector=[3.0, 4.0], max_layer=1
        // Node 2: id=300, vector=[5.0, 6.0], max_layer=0
        graph.vectors.push(&[1.0, 2.0]);
        graph.vectors.push(&[3.0, 4.0]);
        graph.vectors.push(&[5.0, 6.0]);

        // Manually build nodes with neighbor lists
        let node0 = HnswNode {
            id: 100,
            vector_idx: 0,
            max_layer: 1,
            // layer 0: connected to node 1 and node 2
            // layer 1: connected to node 1
            neighbors: vec![vec![1, 2], vec![1]],
        };

        let node1 = HnswNode {
            id: 200,
            vector_idx: 1,
            max_layer: 1,
            // layer 0: connected to node 0 and node 2
            // layer 1: connected to node 0
            neighbors: vec![vec![0, 2], vec![0]],
        };

        let node2 = HnswNode {
            id: 300,
            vector_idx: 2,
            max_layer: 0,
            // layer 0: connected to node 0 and node 1
            neighbors: vec![vec![0, 1]],
        };

        graph.nodes.push(node0);
        graph.nodes.push(node1);
        graph.nodes.push(node2);

        graph.id_to_index.insert(100, 0);
        graph.id_to_index.insert(200, 1);
        graph.id_to_index.insert(300, 2);

        graph.entry_point = Some(0);

        assert_eq!(graph.len(), 3);
        assert!(!graph.is_empty());
        assert_eq!(graph.entry_point, Some(0));

        // 1. Verify get_vector()
        assert_eq!(graph.get_vector(0), &[1.0, 2.0]);
        assert_eq!(graph.get_vector(1), &[3.0, 4.0]);
        assert_eq!(graph.get_vector(2), &[5.0, 6.0]);

        // 2. Verify get_neighbors() at layer 0
        assert_eq!(graph.get_neighbors(0, 0), &[1, 2]);
        assert_eq!(graph.get_neighbors(1, 0), &[0, 2]);
        assert_eq!(graph.get_neighbors(2, 0), &[0, 1]);

        // 3. Verify get_neighbors() at layer 1
        let empty_neighbors: &[u32] = &[];
        assert_eq!(graph.get_neighbors(0, 1), &[1]);
        assert_eq!(graph.get_neighbors(1, 1), &[0]);
        // Node 2 has max_layer=0; layer 1 should return empty slice
        assert_eq!(graph.get_neighbors(2, 1), empty_neighbors);

        // 4. Verify internal_index_of()
        assert_eq!(graph.internal_index_of(100), Some(0));
        assert_eq!(graph.internal_index_of(200), Some(1));
        assert_eq!(graph.internal_index_of(300), Some(2));
    }

    /// Test 3: get_neighbors() with a layer number higher than any node's max_layer
    /// returns an empty slice, not a panic or out-of-bounds error.
    #[test]
    fn test_get_neighbors_out_of_bounds_layer_returns_empty_slice() {
        let mut graph = HnswGraph::new(2, Metric::Euclidean);
        graph.vectors.push(&[1.0, 2.0]);

        let node = HnswNode {
            id: 42,
            vector_idx: 0,
            max_layer: 1,
            neighbors: vec![vec![], vec![]],
        };
        graph.nodes.push(node);
        graph.id_to_index.insert(42, 0);

        let empty: &[u32] = &[];
        // Querying non-existent higher layers
        assert_eq!(graph.get_neighbors(0, 2), empty);
        assert_eq!(graph.get_neighbors(0, 5), empty);
        assert_eq!(graph.get_neighbors(0, 100), empty);
    }

    /// Test 4: internal_index_of() for a nonexistent ID returns None.
    #[test]
    fn test_internal_index_of_nonexistent_id_returns_none() {
        let mut graph = HnswGraph::new(2, Metric::Euclidean);
        graph.vectors.push(&[1.0, 2.0]);
        graph.nodes.push(HnswNode {
            id: 1,
            vector_idx: 0,
            max_layer: 0,
            neighbors: vec![vec![]],
        });
        graph.id_to_index.insert(1, 0);

        assert_eq!(graph.internal_index_of(1), Some(0));
        assert_eq!(graph.internal_index_of(2), None);
        assert_eq!(graph.internal_index_of(9999), None);
    }
}
