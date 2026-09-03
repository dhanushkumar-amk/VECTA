//! Hierarchical Navigable Small World (HNSW) indexing algorithm.
//!
//! Provides approximate nearest neighbor (ANN) search via multi-layer skip-graph traversal.
//! - `graph`: In-memory graph representation and per-layer adjacency storage.

pub mod graph;

pub use graph::{HnswGraph, HnswNode};
