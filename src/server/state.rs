//! Shared application state and collection registry for the Vecta REST server.
//!
//! Reuses the same `RwLock` concurrency pattern from Phase 31 (`ConcurrentFlatIndex`)
//! to allow multiple simultaneous reader threads (searches, metadata queries)
//! while guaranteeing exclusive access for mutating operations (creating collections,
//! inserting points).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use crate::core::flat_index::{FlatIndex, Metric};
use crate::core::hnsw::HnswGraph;
use crate::core::ivf_index::IVFIndex;
use crate::core::ivf_pq_index::IVFPQIndex;

/// An enum wrapping whichever of the four index types a given collection was created as.
///
/// This allows a single registry to hold mixed collection types, dispatching
/// to the appropriate search and insertion logic per variant.
pub enum CollectionIndex {
    Flat(FlatIndex),
    Ivf(IVFIndex),
    Hnsw(HnswGraph),
    IvfPq(IVFPQIndex),
}

impl CollectionIndex {
    /// Return the vector dimensionality of the collection.
    pub fn dim(&self) -> usize {
        match self {
            CollectionIndex::Flat(idx) => idx.dim(),
            CollectionIndex::Ivf(idx) => idx.dim(),
            CollectionIndex::Hnsw(idx) => idx.dim(),
            CollectionIndex::IvfPq(idx) => idx.dim,
        }
    }

    /// Return a static string descriptor of the index type.
    pub fn index_type_str(&self) -> &'static str {
        match self {
            CollectionIndex::Flat(_) => "flat",
            CollectionIndex::Ivf(_) => "ivf",
            CollectionIndex::Hnsw(_) => "hnsw",
            CollectionIndex::IvfPq(_) => "ivfpq",
        }
    }

    /// Return the distance/similarity metric configured for this collection.
    pub fn metric(&self) -> Metric {
        match self {
            CollectionIndex::Flat(idx) => idx.metric,
            CollectionIndex::Ivf(idx) => idx.metric,
            CollectionIndex::Hnsw(idx) => idx.metric,
            CollectionIndex::IvfPq(idx) => idx.metric,
        }
    }

    /// Return the current total number of vectors in the index.
    pub fn len(&self) -> usize {
        match self {
            CollectionIndex::Flat(idx) => idx.len(),
            CollectionIndex::Ivf(idx) => idx.len(),
            CollectionIndex::Hnsw(idx) => idx.len(),
            CollectionIndex::IvfPq(idx) => idx.len(),
        }
    }

    /// Returns `true` if the collection contains zero vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Shared application state managed by Axum and injected into request handlers.
pub struct AppState {
    /// Thread-safe registry mapping collection names to their respective index instances.
    pub collections: RwLock<HashMap<String, CollectionIndex>>,
    /// Path to the data directory for on-disk persistence (wired in Phase 43).
    pub data_dir: PathBuf,
}

impl AppState {
    /// Create a new, empty application state.
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
            data_dir,
        }
    }
}
