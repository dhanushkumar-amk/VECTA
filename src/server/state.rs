//! Shared application state, collection registry, and disk persistence wiring for Vecta.
//!
//! Reuses the same `RwLock` concurrency pattern from Phase 31 (`ConcurrentFlatIndex`)
//! and persistence serialization from Phases 25-28 (`serialize` and `wal`).

use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::core::flat_index::{FlatIndex, Metric};
use crate::core::hnsw::layer::HnswConfig;
use crate::core::hnsw::HnswGraph;
use crate::core::ivf_index::IVFIndex;
use crate::core::ivf_pq_index::IVFPQIndex;
use crate::core::serialize::{
    load_flat_index, load_hnsw_index, load_ivf_index, load_ivf_pq_index, peek_index_type,
    save_flat_index, save_hnsw_index, save_ivf_index, save_ivf_pq_index, INDEX_TYPE_FLAT,
    INDEX_TYPE_HNSW, INDEX_TYPE_IVF, INDEX_TYPE_IVF_PQ,
};
use crate::core::wal::load_with_wal_recovery;

/// An enum wrapping whichever of the four index types a given collection was created as.
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
    /// Path to the data directory for on-disk persistence and write-ahead logs.
    pub data_dir: PathBuf,
}

impl AppState {
    /// Create a new application state, automatically scanning and restoring
    /// any persisted collections and WALs from `data_dir`.
    pub fn new(data_dir: PathBuf) -> Self {
        if !data_dir.exists() {
            let _ = create_dir_all(&data_dir);
        }

        let collections = match Self::load_from_disk(&data_dir) {
            Ok(map) => {
                let count = map.len();
                if count > 0 {
                    println!(
                        "Restored {} collections from {}",
                        count,
                        data_dir.display()
                    );
                }
                map
            }
            Err(err) => {
                eprintln!(
                    "Warning: failed to restore collections from {}: {}",
                    data_dir.display(),
                    err
                );
                HashMap::new()
            }
        };

        Self {
            collections: RwLock::new(collections),
            data_dir,
        }
    }

    /// Path to a collection's binary snapshot file (`<data_dir>/<name>.vcta`).
    pub fn collection_file_path(&self, name: &str) -> PathBuf {
        self.data_dir.join(format!("{}.vcta", name))
    }

    /// Path to a collection's write-ahead log (`<data_dir>/<name>.wal`).
    pub fn wal_file_path(&self, name: &str) -> PathBuf {
        self.data_dir.join(format!("{}.wal", name))
    }

    /// Save a snapshot of a specific collection to disk.
    pub fn save_collection_snapshot(&self, name: &str, index: &CollectionIndex) -> Result<(), String> {
        let path = self.collection_file_path(name);
        match index {
            CollectionIndex::Flat(idx) => {
                save_flat_index(idx, &path).map_err(|e| format!("save_flat_index failed: {}", e))
            }
            CollectionIndex::Ivf(idx) => {
                save_ivf_index(idx, &path).map_err(|e| format!("save_ivf_index failed: {}", e))
            }
            CollectionIndex::Hnsw(idx) => {
                let config = HnswConfig {
                    m: 16,
                    ef_construction: 200,
                    ef_search: 50,
                };
                save_hnsw_index(idx, &config, &path)
                    .map_err(|e| format!("save_hnsw_index failed: {}", e))
            }
            CollectionIndex::IvfPq(idx) => {
                save_ivf_pq_index(idx, &path).map_err(|e| format!("save_ivf_pq_index failed: {}", e))
            }
        }
    }

    /// Create a full snapshot checkpoint of a collection and truncate its WAL.
    pub fn checkpoint_collection(&self, name: &str) -> Result<(), String> {
        let registry = self
            .collections
            .read()
            .map_err(|_| "collection registry lock poisoned".to_string())?;

        let index = registry
            .get(name)
            .ok_or_else(|| format!("collection '{}' not found", name))?;

        // 1. Save base snapshot
        self.save_collection_snapshot(name, index)?;

        // 2. If FlatIndex, clear / truncate the WAL file to 0 bytes
        if let CollectionIndex::Flat(_) = index {
            let wal_path = self.wal_file_path(name);
            if wal_path.exists() {
                File::create(&wal_path)
                    .map_err(|e| format!("failed to truncate WAL file {}: {}", wal_path.display(), e))?;
            }
        }

        Ok(())
    }

    /// Checkpoint all collections currently registered in memory.
    pub fn checkpoint_all(&self) -> Result<usize, String> {
        let registry = self
            .collections
            .read()
            .map_err(|_| "collection registry lock poisoned".to_string())?;

        let mut count = 0;
        for (name, index) in registry.iter() {
            self.save_collection_snapshot(name, index)?;
            if let CollectionIndex::Flat(_) = index {
                let wal_path = self.wal_file_path(name);
                if wal_path.exists() {
                    let _ = File::create(&wal_path);
                }
            }
            count += 1;
        }

        Ok(count)
    }

    /// Scan `data_dir` for `.vcta` index files and restore collections using `peek_index_type`.
    /// For FlatIndex collections, uncheckpointed live mutations in `.wal` are automatically replayed.
    pub fn load_from_disk(data_dir: &Path) -> Result<HashMap<String, CollectionIndex>, String> {
        let mut map = HashMap::new();
        if !data_dir.exists() {
            return Ok(map);
        }

        let read_dir = std::fs::read_dir(data_dir)
            .map_err(|e| format!("failed to read data directory {}: {}", data_dir.display(), e))?;

        for entry_res in read_dir {
            let entry = entry_res.map_err(|e| format!("directory entry error: {}", e))?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "vcta" {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            let name = stem.to_string();
                            let index_type = peek_index_type(&path)?;
                            let index = match index_type {
                                INDEX_TYPE_FLAT => {
                                    let wal_path = data_dir.join(format!("{}.wal", name));
                                    let flat = if wal_path.exists() {
                                        load_with_wal_recovery(&path, &wal_path)?
                                    } else {
                                        load_flat_index(&path)?
                                    };
                                    CollectionIndex::Flat(flat)
                                }
                                INDEX_TYPE_IVF => {
                                    let ivf = load_ivf_index(&path)?;
                                    CollectionIndex::Ivf(ivf)
                                }
                                INDEX_TYPE_HNSW => {
                                    let (hnsw, _cfg) = load_hnsw_index(&path)?;
                                    CollectionIndex::Hnsw(hnsw)
                                }
                                INDEX_TYPE_IVF_PQ => {
                                    let ivf_pq = load_ivf_pq_index(&path)?;
                                    CollectionIndex::IvfPq(ivf_pq)
                                }
                                other => {
                                    return Err(format!(
                                        "unknown index type {} in file {}",
                                        other,
                                        path.display()
                                    ));
                                }
                            };
                            map.insert(name, index);
                        }
                    }
                }
            }
        }

        Ok(map)
    }
}
