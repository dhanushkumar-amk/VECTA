//! Hierarchical Navigable Small World (HNSW) indexing algorithm.
//!
//! Provides approximate nearest neighbor (ANN) search via multi-layer skip-graph traversal.
//! - `graph`: In-memory graph representation and per-layer adjacency storage.

pub mod graph;
pub mod insert;
pub mod layer;
pub mod search_layer;

pub use graph::{HnswGraph, HnswNode};
pub use insert::{insert, select_neighbors};
pub use layer::{assign_layer, ml_factor, HnswConfig};
pub use search_layer::greedy_search_layer;
