//! # Vecta: A Fast, Production-Grade Vector Database Engine in Rust
//!
//! `vecta` is a high-performance vector search engine featuring four core index architectures,
//! zero-copy memory mapping, write-ahead logging (WAL) crash resilience, metadata filtering,
//! reader-writer concurrency, and horizontal sharding.
//!
//! ## Core Architectures:
//! - **[`core::flat_index::FlatIndex`]**: Exact brute-force vector search (100% recall baseline).
//! - **[`core::ivf_index::IVFIndex`]**: Inverted File index with k-means coarse quantization.
//! - **[`core::hnsw`]**: Hierarchical Navigable Small World graph for sub-millisecond approximate search.
//! - **[`core::ivf_pq_index::IVFPQIndex`]**: Inverted File with Product Quantization for up to 20x memory compression.
//!
//! ## Storage & Scaling:
//! - **Binary Persistence & Mmap**: Fast zero-copy memory-mapped search via [`core::serialize`].
//! - **WAL Crash Recovery**: Durable append-only logging via [`core::wal`].
//! - **Metadata Filtering**: Expressive post-filtered search via [`core::metadata`].
//! - **Concurrent Access**: Thread-safe multi-reader single-writer locking via [`core::concurrent_index`].
//! - **Sharding**: Coordinator fan-out and candidate merging across shards via [`core::sharded_index`].

pub mod core;
mod python;

use pyo3::prelude::*;

/// The top-level Python module for vecta.
///
/// All PyO3 function/class registrations happen here,
/// delegating to `python.rs` for the actual implementations.
#[pymodule]
fn vecta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::FlatIndex>()?;
    m.add_class::<python::ConcurrentFlatIndex>()?;
    m.add_class::<python::ShardedFlatIndex>()?;
    m.add_class::<python::IVFIndex>()?;
    m.add_class::<python::HnswIndex>()?;
    m.add_class::<python::IVFPQIndex>()?;
    m.add_class::<python::MetadataStore>()?;
    python::register(m)?;
    Ok(())
}
