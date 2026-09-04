//! # Vecta Core Engine
//!
//! Internal vector database primitives, data structures, and algorithms:
//! - [`flat_index`]: Brute-force exact search engine with SIMD-aligned vector batches.
//! - [`ivf_index`]: Inverted file index with k-means coarse quantization.
//! - [`hnsw`]: Multi-layer Hierarchical Navigable Small World graph.
//! - [`ivf_pq_index`]: Compressed inverted index using Product Quantization (ADC).
//! - [`kmeans`]: Lloyd's k-means clustering with k-means++ initialization.
//! - [`pq`]: Product quantization subvector decomposition and codebook training.
//! - [`topk`]: Min-heap and max-heap bounded top-k candidate selection.
//! - [`vector`]: SIMD-friendly vector similarity distance metrics (L2, Cosine, Dot).
//! - [`batch`]: Contiguous flat memory buffer for vector batches.
//! - [`serialize`]: Binary format specification, save/load, and memory mapping.
//! - [`wal`]: Append-only write-ahead log for durable crash recovery.
//! - [`metadata`]: Attribute store and filtered top-k candidate pruning.
//! - [`concurrent_index`]: Thread-safe reader-writer index wrapper (`RwLock`).
//! - [`sharded_index`]: In-process horizontal partitioning across multiple shards.

pub mod batch;
pub mod concurrent_index;
pub mod flat_index;
pub mod hnsw;
pub mod ivf_index;
pub mod ivf_pq_index;
pub mod kmeans;
pub mod metadata;
pub mod pq;
pub mod serialize;
pub mod sharded_index;
pub mod topk;
pub mod vector;
pub mod wal;
