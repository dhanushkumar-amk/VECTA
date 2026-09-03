//! Hierarchical Navigable Small World (HNSW) indexing algorithm.
//!
//! Provides approximate nearest neighbor (ANN) search via multi-layer skip-graph traversal.
//! - `graph`: In-memory graph representation and per-layer adjacency storage.

pub mod graph;
pub mod layer;

pub use graph::{HnswGraph, HnswNode};
pub use layer::{assign_layer, ml_factor, HnswConfig};
