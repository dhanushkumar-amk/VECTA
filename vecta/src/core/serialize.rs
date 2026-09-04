//! Binary serialization format for persisting vecta indexes to disk.
//!
//! # On-Disk File Layout Specification
//!
//! All four index types (`FlatIndex`, `IVFIndex`, `HnswGraph`, and `IVFPQIndex`)
//! share a unified 22-byte header convention. Every multi-byte numerical primitive
//! is encoded using **explicit little-endian** byte ordering (`to_le_bytes` / `from_le_bytes`)
//! to guarantee full cross-platform portability across CPU architectures.
//!
//! ```text
//! [HEADER] (22 bytes total, shared across all 4 index families)
//!   - magic_bytes:    [u8; 4] = b"VCTA"
//!   - format_version: u32 (LE) (current = 1)
//!   - index_type:     u8       (0 = FlatIndex, 1 = IVFIndex, 2 = HnswGraph, 3 = IVFPQIndex)
//!   - metric:         u8       (0 = Euclidean, 1 = Cosine, 2 = DotProduct)
//!   - dim:            u32 (LE) (vector dimensionality)
//!   - num_vectors:    u64 (LE) (total indexed vectors / graph nodes)
//! ```
//!
//! ### Layout by Index Type:
//!
//! 1. **FlatIndex (`index_type = 0`)**:
//!    - `[HEADER]` (22 bytes)
//!    - `[IDS SECTION]`: `num_vectors * 8` bytes (contiguous little-endian `u64` IDs)
//!    - `[VECTOR DATA SECTION]`: `num_vectors * dim * 4` bytes (contiguous little-endian `f32` coordinates, row-major)
//!
//! 2. **IVFIndex (`index_type = 1`)**:
//!    - `[HEADER]` (22 bytes)
//!    - `[num_clusters: u32]` (4 bytes)
//!    - `[centroids: num_clusters * dim * 4 bytes]` (contiguous `f32` centroid coordinates)
//!    - `[per-cluster sections]`: For each of the `num_clusters` inverted lists:
//!      - `cluster_count: u32`
//!      - `cluster_count * 8` bytes of IDs
//!      - `cluster_count * dim * 4` bytes of vector coordinates
//!    - `[is_trained: u8]` (1 byte boolean flag: 1 = trained, 0 = untrained)
//!
//! 3. **HnswGraph (`index_type = 2`)**:
//!    - `[HEADER]` (22 bytes)
//!    - `[hnsw config: m: u32, ef_construction: u32, ef_search: u32]` (12 bytes)
//!    - `[entry_point: i64]` (8 bytes, with `-1` representing `None`)
//!    - `[num_nodes: u64]` (8 bytes, node count matching `num_vectors`)
//!    - `[vector data: num_nodes * dim * 4 bytes]` (contiguous `f32` coordinates)
//!    - `[per-node variable-length section]`: For each node:
//!      - `id: u64`
//!      - `max_layer: u32`
//!      - For each layer in `0..=max_layer`:
//!        - `neighbor_count: u32`
//!        - `neighbor_count * 4` bytes of `u32` internal neighbor indices
//!
//!    *(NOTE: This per-node section is the one exception in vecta's serialization that is NOT*
//!    *fixed-size records, because graph adjacency in HNSW is inherently variable-length).*
//!
//! 4. **IVFPQIndex (`index_type = 3`)**:
//!    - `[HEADER]` (22 bytes)
//!    - `[pq config: m: u32, k_per_subvector: u32]` (8 bytes)
//!    - `[num_clusters: u32]` (4 bytes)
//!    - `[centroids: num_clusters * dim * 4 bytes]` (full-precision coarse centroids)
//!    - `[pq codebooks: m * k_per_subvector * sub_dim * 4 bytes]`
//!    - `[per-cluster sections]`: For each coarse cluster:
//!      - `vector_count: u32`
//!      - `vector_count` pairs of `(id: u64, code: [u8; m])`
//!    - `[is_trained: u8]` (1 byte boolean flag)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::core::batch::VectorBatch;
use crate::core::flat_index::{FlatIndex, Metric};
use crate::core::hnsw::{HnswConfig, HnswGraph, HnswNode};
use crate::core::ivf_index::IVFIndex;
use crate::core::ivf_pq_index::IVFPQIndex;
use crate::core::pq::{PQCodebooks, PQConfig};

/// Magic identification bytes at the beginning of every vecta index file (`b"VCTA"`).
pub const MAGIC_BYTES: &[u8; 4] = b"VCTA";

/// Current serialization format version.
pub const FORMAT_VERSION: u32 = 1;

/// Index type identifiers stored in the header byte.
pub const INDEX_TYPE_FLAT: u8 = 0;
/// IVFIndex persistence identifier.
pub const INDEX_TYPE_IVF: u8 = 1;
/// HnswGraph persistence identifier.
pub const INDEX_TYPE_HNSW: u8 = 2;
/// IVFPQIndex persistence identifier.
pub const INDEX_TYPE_IVF_PQ: u8 = 3;

/// Metric encoding values stored in the header byte.
pub const METRIC_EUCLIDEAN: u8 = 0;
pub const METRIC_COSINE: u8 = 1;
pub const METRIC_DOT_PRODUCT: u8 = 2;

/// Convert a runtime [`Metric`] into its serialized binary byte code.
#[inline]
pub fn metric_to_code(metric: Metric) -> u8 {
    match metric {
        Metric::Euclidean => METRIC_EUCLIDEAN,
        Metric::Cosine => METRIC_COSINE,
        Metric::DotProduct => METRIC_DOT_PRODUCT,
    }
}

/// Parse a serialized binary byte code into a runtime [`Metric`].
#[inline]
pub fn code_to_metric(code: u8) -> Result<Metric, String> {
    match code {
        METRIC_EUCLIDEAN => Ok(Metric::Euclidean),
        METRIC_COSINE => Ok(Metric::Cosine),
        METRIC_DOT_PRODUCT => Ok(Metric::DotProduct),
        other => Err(format!("unknown metric code: {}", other)),
    }
}

// ============================================================================
// Shared I/O Streaming Helpers
// ============================================================================

/// Parsed metadata from the standard 22-byte vecta header.
#[derive(Debug, Clone, PartialEq)]
pub struct HeaderMetadata {
    pub index_type: u8,
    pub metric: Metric,
    pub dim: usize,
    pub num_vectors: usize,
}

/// Write the standard 22-byte vecta index header to a writer.
pub fn write_header<W: Write>(
    writer: &mut W,
    index_type: u8,
    metric: Metric,
    dim: usize,
    num_vectors: usize,
) -> std::io::Result<()> {
    writer.write_all(MAGIC_BYTES)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
    writer.write_all(&[index_type])?;
    writer.write_all(&[metric_to_code(metric)])?;
    writer.write_all(&(dim as u32).to_le_bytes())?;
    writer.write_all(&(num_vectors as u64).to_le_bytes())?;
    Ok(())
}

/// Read and validate the standard 22-byte vecta index header from a reader.
pub fn read_header<R: Read>(
    reader: &mut R,
    expected_index_type: u8,
    expected_type_name: &str,
) -> Result<HeaderMetadata, String> {
    // 1. Magic bytes
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("failed to read magic bytes: {}", e))?;
    if &magic != MAGIC_BYTES {
        return Err("not a valid vecta index file".to_string());
    }

    // 2. Format version
    let mut version_buf = [0u8; 4];
    reader
        .read_exact(&mut version_buf)
        .map_err(|e| format!("failed to read format version: {}", e))?;
    let version = u32::from_le_bytes(version_buf);
    if version != FORMAT_VERSION {
        return Err(format!(
            "unsupported format version {}: current supported version is {}",
            version, FORMAT_VERSION
        ));
    }

    // 3. Index type
    let mut index_type_buf = [0u8; 1];
    reader
        .read_exact(&mut index_type_buf)
        .map_err(|e| format!("failed to read index type: {}", e))?;
    let index_type = index_type_buf[0];
    if index_type != expected_index_type {
        return Err(format!(
            "file contains a different index type ({}), expected {} ({})",
            index_type, expected_type_name, expected_index_type
        ));
    }

    // 4. Metric
    let mut metric_buf = [0u8; 1];
    reader
        .read_exact(&mut metric_buf)
        .map_err(|e| format!("failed to read metric code: {}", e))?;
    let metric = code_to_metric(metric_buf[0])?;

    // 5. Dimension
    let mut dim_buf = [0u8; 4];
    reader
        .read_exact(&mut dim_buf)
        .map_err(|e| format!("failed to read dimension: {}", e))?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    if dim == 0 {
        return Err("invalid serialized index dimension 0".to_string());
    }

    // 6. Vector count
    let mut count_buf = [0u8; 8];
    reader
        .read_exact(&mut count_buf)
        .map_err(|e| format!("failed to read vector count: {}", e))?;
    let num_vectors = u64::from_le_bytes(count_buf) as usize;

    Ok(HeaderMetadata {
        index_type,
        metric,
        dim,
        num_vectors,
    })
}

/// Write a slice of external `u64` IDs in contiguous little-endian chunks.
pub fn write_ids<W: Write>(writer: &mut W, ids: &[u64]) -> std::io::Result<()> {
    let mut id_buf = Vec::with_capacity(8192);
    for &id in ids {
        id_buf.extend_from_slice(&id.to_le_bytes());
        if id_buf.len() >= 8192 {
            writer.write_all(&id_buf)?;
            id_buf.clear();
        }
    }
    if !id_buf.is_empty() {
        writer.write_all(&id_buf)?;
    }
    Ok(())
}

/// Read a sequence of `count` `u64` IDs from a reader in little-endian order.
pub fn read_ids<R: Read>(reader: &mut R, count: usize) -> Result<Vec<u64>, String> {
    let mut ids = Vec::with_capacity(count);
    let mut chunk_bytes = vec![0u8; 8192];
    let mut ids_read = 0;
    while ids_read < count {
        let ids_to_read = (count - ids_read).min(chunk_bytes.len() / 8);
        let bytes_to_read = ids_to_read * 8;
        reader
            .read_exact(&mut chunk_bytes[..bytes_to_read])
            .map_err(|e| format!("failed to read IDs (corrupted/truncated file): {}", e))?;

        let (chunks, _) = chunk_bytes[..bytes_to_read].as_chunks::<8>();
        for &chunk in chunks {
            ids.push(u64::from_le_bytes(chunk));
        }
        ids_read += ids_to_read;
    }
    Ok(ids)
}

/// Write a slice of `f32` values in contiguous little-endian chunks.
pub fn write_floats<W: Write>(writer: &mut W, data: &[f32]) -> std::io::Result<()> {
    let mut float_buf = Vec::with_capacity(32768);
    for &val in data {
        float_buf.extend_from_slice(&val.to_le_bytes());
        if float_buf.len() >= 32768 {
            writer.write_all(&float_buf)?;
            float_buf.clear();
        }
    }
    if !float_buf.is_empty() {
        writer.write_all(&float_buf)?;
    }
    Ok(())
}

/// Read a sequence of `total_floats` `f32` values from a reader in little-endian order.
pub fn read_floats<R: Read>(reader: &mut R, total_floats: usize) -> Result<Vec<f32>, String> {
    let mut data = Vec::with_capacity(total_floats);
    let mut chunk_bytes = vec![0u8; 32768];
    let mut floats_read = 0;
    while floats_read < total_floats {
        let floats_to_read = (total_floats - floats_read).min(chunk_bytes.len() / 4);
        let bytes_to_read = floats_to_read * 4;
        reader
            .read_exact(&mut chunk_bytes[..bytes_to_read])
            .map_err(|e| {
                format!(
                    "failed to read vector data (corrupted/truncated file): {}",
                    e
                )
            })?;

        let (chunks, _) = chunk_bytes[..bytes_to_read].as_chunks::<4>();
        for &chunk in chunks {
            data.push(f32::from_le_bytes(chunk));
        }
        floats_read += floats_to_read;
    }
    Ok(data)
}

/// Peek at the header of a vecta index file to determine its [`index_type`]
/// without reading the rest of the index.
///
/// # Returns
/// - `Ok(0)` for [`FlatIndex`] (`INDEX_TYPE_FLAT`)
/// - `Ok(1)` for [`IVFIndex`] (`INDEX_TYPE_IVF`)
/// - `Ok(2)` for [`HnswGraph`] (`INDEX_TYPE_HNSW`)
/// - `Ok(3)` for [`IVFPQIndex`] (`INDEX_TYPE_IVF_PQ`)
///
/// # Errors
/// Returns an `Err(String)` if the file cannot be opened, has fewer than 9 bytes,
/// does not match [`MAGIC_BYTES`], or has an unsupported [`FORMAT_VERSION`].
pub fn peek_index_type(path: &Path) -> Result<u8, String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open file {}: {}", path.display(), e))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("failed to read magic bytes: {}", e))?;
    if &magic != MAGIC_BYTES {
        return Err("not a valid vecta index file".to_string());
    }

    let mut version_buf = [0u8; 4];
    reader
        .read_exact(&mut version_buf)
        .map_err(|e| format!("failed to read format version: {}", e))?;
    let version = u32::from_le_bytes(version_buf);
    if version != FORMAT_VERSION {
        return Err(format!(
            "unsupported format version {}: current supported version is {}",
            version, FORMAT_VERSION
        ));
    }

    let mut type_buf = [0u8; 1];
    reader
        .read_exact(&mut type_buf)
        .map_err(|e| format!("failed to read index type: {}", e))?;
    Ok(type_buf[0])
}

// ============================================================================
// FlatIndex Persistence
// ============================================================================

/// Save a [`FlatIndex`] to disk in the vecta binary serialization format.
///
/// Uses [`BufWriter`] to minimize syscall overhead on large indexes.
///
/// # Errors
/// Returns an [`std::io::Error`] if file creation or any write operation fails.
pub fn save_flat_index(index: &FlatIndex, path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // 1. Header (22 bytes)
    write_header(
        &mut writer,
        INDEX_TYPE_FLAT,
        index.metric,
        index.dim(),
        index.len(),
    )?;

    // 2. IDs Section (num_vectors * 8 bytes)
    write_ids(&mut writer, &index.ids)?;

    // 3. Vector Data Section (num_vectors * dim * 4 bytes)
    write_floats(&mut writer, &index.batch.data)?;

    writer.flush()?;
    Ok(())
}

/// Load a [`FlatIndex`] from disk from a vecta binary serialization file.
///
/// Reads all IDs and vector data using buffered I/O ([`BufReader`]).
///
/// # Errors
/// Returns an informative `Err(String)` if header validation fails, or if the file
/// is truncated or corrupted mid-stream.
pub fn load_flat_index(path: &Path) -> Result<FlatIndex, String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open file {}: {}", path.display(), e))?;
    let mut reader = BufReader::new(file);

    let header = read_header(&mut reader, INDEX_TYPE_FLAT, "FlatIndex")?;
    let ids = read_ids(&mut reader, header.num_vectors)?;

    let total_floats = header
        .num_vectors
        .checked_mul(header.dim)
        .ok_or_else(|| "total float count overflows usize".to_string())?;
    let data = read_floats(&mut reader, total_floats)?;

    let batch = VectorBatch::from_parts(data, header.dim, header.num_vectors)?;
    let index = FlatIndex::from_parts(batch, ids, header.metric)?;

    Ok(index)
}

// ============================================================================
// IVFIndex Persistence
// ============================================================================

/// Save an [`IVFIndex`] to disk in the vecta binary serialization format.
///
/// # Layout:
/// - 22-byte standard header (`index_type = 1`).
/// - `num_clusters: u32` (little-endian).
/// - `centroids: num_clusters * dim * 4 bytes` (full-precision coordinates).
/// - For each inverted list: `cluster_count: u32`, IDs, vector coordinates.
/// - `is_trained: u8` (1 = trained, 0 = untrained).
pub fn save_ivf_index(index: &IVFIndex, path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let dim = index.dim;
    let num_vectors = index.len();
    let num_clusters = index.inverted_lists.len();

    // 1. Standard header (22 bytes)
    write_header(&mut writer, INDEX_TYPE_IVF, index.metric, dim, num_vectors)?;

    // 2. Number of clusters (4 bytes)
    writer.write_all(&(num_clusters as u32).to_le_bytes())?;

    // 3. Centroids (num_clusters * dim * 4 bytes)
    if index.is_trained {
        write_floats(&mut writer, &index.centroids.data)?;
    } else {
        let dummy = vec![0.0f32; num_clusters * dim];
        write_floats(&mut writer, &dummy)?;
    }

    // 4. Per-cluster inverted lists
    for cluster in &index.inverted_lists {
        let count = cluster.len() as u32;
        writer.write_all(&count.to_le_bytes())?;
        write_ids(&mut writer, &cluster.ids)?;
        write_floats(&mut writer, &cluster.batch.data)?;
    }

    // 5. is_trained flag (1 byte)
    writer.write_all(&[if index.is_trained { 1 } else { 0 }])?;

    writer.flush()?;
    Ok(())
}

/// Load an [`IVFIndex`] from disk from a vecta binary serialization file.
pub fn load_ivf_index(path: &Path) -> Result<IVFIndex, String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open file {}: {}", path.display(), e))?;
    let mut reader = BufReader::new(file);

    // 1. Standard header
    let header = read_header(&mut reader, INDEX_TYPE_IVF, "IVFIndex")?;
    let dim = header.dim;
    let metric = header.metric;

    // 2. Number of clusters
    let mut clusters_buf = [0u8; 4];
    reader
        .read_exact(&mut clusters_buf)
        .map_err(|e| format!("failed to read num_clusters: {}", e))?;
    let num_clusters = u32::from_le_bytes(clusters_buf) as usize;

    // 3. Centroids
    let total_centroid_floats = num_clusters
        .checked_mul(dim)
        .ok_or_else(|| "centroid float count overflows usize".to_string())?;
    let centroid_data = read_floats(&mut reader, total_centroid_floats)?;

    // 4. Per-cluster inverted lists
    let mut inverted_lists = Vec::with_capacity(num_clusters);
    let mut total_loaded_vectors = 0;
    for _ in 0..num_clusters {
        let mut count_buf = [0u8; 4];
        reader
            .read_exact(&mut count_buf)
            .map_err(|e| format!("failed to read cluster vector count: {}", e))?;
        let count = u32::from_le_bytes(count_buf) as usize;
        total_loaded_vectors += count;

        let ids = read_ids(&mut reader, count)?;
        let cluster_floats = count
            .checked_mul(dim)
            .ok_or_else(|| "cluster float count overflows usize".to_string())?;
        let data = read_floats(&mut reader, cluster_floats)?;
        let batch = VectorBatch::from_parts(data, dim, count)?;
        let cluster_index = FlatIndex::from_parts(batch, ids, metric)?;
        inverted_lists.push(cluster_index);
    }

    if total_loaded_vectors != header.num_vectors {
        return Err(format!(
            "mismatch between header vector count ({}) and sum of cluster counts ({})",
            header.num_vectors, total_loaded_vectors
        ));
    }

    // 5. is_trained flag
    let mut trained_buf = [0u8; 1];
    reader
        .read_exact(&mut trained_buf)
        .map_err(|e| format!("failed to read is_trained flag: {}", e))?;
    let is_trained = trained_buf[0] != 0;

    let centroids = if is_trained {
        VectorBatch::from_parts(centroid_data, dim, num_clusters)?
    } else {
        VectorBatch::new(dim)
    };

    Ok(IVFIndex {
        centroids,
        inverted_lists,
        dim,
        metric,
        is_trained,
    })
}

// ============================================================================
// HnswGraph Persistence
// ============================================================================

/// Save an [`HnswGraph`] and its construction [`HnswConfig`] to disk.
///
/// # Layout:
/// - 22-byte standard header (`index_type = 2`).
/// - `hnsw config`: `m: u32`, `ef_construction: u32`, `ef_search: u32`.
/// - `entry_point: i64` (`-1` indicates `None`).
/// - `num_nodes: u64`.
/// - `vector data: num_nodes * dim * 4 bytes` (contiguous coordinates).
/// - `per-node variable-length section`: for each node, `id: u64`, `max_layer: u32`,
///   and for each layer `0..=max_layer`, `neighbor_count: u32` followed by `u32` neighbor indices.
///
/// *(NOTE: Graph adjacency is inherently variable-length; this is the sole non-fixed-size*
/// *section across vecta's index serialization formats).*
pub fn save_hnsw_index(graph: &HnswGraph, config: &HnswConfig, path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let dim = graph.dim;
    let num_nodes = graph.nodes.len();

    // 1. Standard header (22 bytes)
    write_header(&mut writer, INDEX_TYPE_HNSW, graph.metric, dim, num_nodes)?;

    // 2. HnswConfig parameters (12 bytes)
    writer.write_all(&(config.m as u32).to_le_bytes())?;
    writer.write_all(&(config.ef_construction as u32).to_le_bytes())?;
    writer.write_all(&(config.ef_search as u32).to_le_bytes())?;

    // 3. Entry point index (8 bytes, -1 for None)
    let ep_raw: i64 = match graph.entry_point {
        Some(idx) => idx as i64,
        None => -1,
    };
    writer.write_all(&ep_raw.to_le_bytes())?;

    // 4. Number of nodes (8 bytes)
    writer.write_all(&(num_nodes as u64).to_le_bytes())?;

    // 5. Contiguous vector coordinate store
    write_floats(&mut writer, &graph.vectors.data)?;

    // 6. Per-node variable-length adjacency section
    for node in &graph.nodes {
        writer.write_all(&node.id.to_le_bytes())?;
        writer.write_all(&(node.max_layer as u32).to_le_bytes())?;
        for layer in 0..=node.max_layer {
            let nbrs = &node.neighbors[layer];
            writer.write_all(&(nbrs.len() as u32).to_le_bytes())?;
            for &nbr in nbrs {
                writer.write_all(&nbr.to_le_bytes())?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

/// Load an [`HnswGraph`] and its [`HnswConfig`] from disk.
pub fn load_hnsw_index(path: &Path) -> Result<(HnswGraph, HnswConfig), String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open file {}: {}", path.display(), e))?;
    let mut reader = BufReader::new(file);

    // 1. Standard header
    let header = read_header(&mut reader, INDEX_TYPE_HNSW, "HnswGraph")?;
    let dim = header.dim;
    let metric = header.metric;

    // 2. HnswConfig
    let mut cfg_buf = [0u8; 12];
    reader
        .read_exact(&mut cfg_buf)
        .map_err(|e| format!("failed to read HNSW config: {}", e))?;
    let m = u32::from_le_bytes(cfg_buf[0..4].try_into().unwrap()) as usize;
    let ef_construction = u32::from_le_bytes(cfg_buf[4..8].try_into().unwrap()) as usize;
    let ef_search = u32::from_le_bytes(cfg_buf[8..12].try_into().unwrap()) as usize;
    let config = HnswConfig {
        m,
        ef_construction,
        ef_search,
    };

    // 3. Entry point
    let mut ep_buf = [0u8; 8];
    reader
        .read_exact(&mut ep_buf)
        .map_err(|e| format!("failed to read entry point: {}", e))?;
    let ep_raw = i64::from_le_bytes(ep_buf);
    let entry_point = if ep_raw == -1 {
        None
    } else if ep_raw < -1 {
        return Err(format!("invalid negative entry point index: {}", ep_raw));
    } else {
        Some(ep_raw as usize)
    };

    // 4. Number of nodes
    let mut count_buf = [0u8; 8];
    reader
        .read_exact(&mut count_buf)
        .map_err(|e| format!("failed to read node count: {}", e))?;
    let num_nodes = u64::from_le_bytes(count_buf) as usize;
    if num_nodes != header.num_vectors {
        return Err(format!(
            "mismatch between header vector count ({}) and node count ({})",
            header.num_vectors, num_nodes
        ));
    }

    if let Some(ep) = entry_point {
        if ep >= num_nodes {
            return Err(format!(
                "entry point index {} out of bounds for graph with {} nodes",
                ep, num_nodes
            ));
        }
    }

    // 5. Contiguous vector coordinate store
    let total_floats = num_nodes
        .checked_mul(dim)
        .ok_or_else(|| "total float count overflows usize".to_string())?;
    let vector_data = read_floats(&mut reader, total_floats)?;
    let vectors = VectorBatch::from_parts(vector_data, dim, num_nodes)?;

    // 6. Per-node variable-length adjacency section
    let mut nodes = Vec::with_capacity(num_nodes);
    let mut id_to_index = HashMap::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let mut id_buf = [0u8; 8];
        reader
            .read_exact(&mut id_buf)
            .map_err(|e| format!("failed to read node ID (node {}): {}", i, e))?;
        let id = u64::from_le_bytes(id_buf);

        let mut layer_buf = [0u8; 4];
        reader
            .read_exact(&mut layer_buf)
            .map_err(|e| format!("failed to read max_layer (node {}): {}", i, e))?;
        let max_layer = u32::from_le_bytes(layer_buf) as usize;

        let mut neighbors = Vec::with_capacity(max_layer + 1);
        for layer in 0..=max_layer {
            let mut nbr_count_buf = [0u8; 4];
            reader.read_exact(&mut nbr_count_buf).map_err(|e| {
                format!(
                    "failed to read neighbor count (node {}, layer {}): {}",
                    i, layer, e
                )
            })?;
            let nbr_count = u32::from_le_bytes(nbr_count_buf) as usize;

            let mut nbr_raw = vec![0u8; nbr_count * 4];
            reader.read_exact(&mut nbr_raw).map_err(|e| {
                format!(
                    "failed to read neighbor indices (node {}, layer {}): {}",
                    i, layer, e
                )
            })?;

            let mut nbr_list = Vec::with_capacity(nbr_count);
            let (chunks, _) = nbr_raw.as_chunks::<4>();
            for &chunk in chunks {
                let nbr_idx = u32::from_le_bytes(chunk);
                if (nbr_idx as usize) >= num_nodes {
                    return Err(format!(
                        "neighbor index {} out of bounds for graph with {} nodes",
                        nbr_idx, num_nodes
                    ));
                }
                nbr_list.push(nbr_idx);
            }
            neighbors.push(nbr_list);
        }

        nodes.push(HnswNode {
            id,
            vector_idx: i,
            max_layer,
            neighbors,
        });
        id_to_index.insert(id, i);
    }

    Ok((
        HnswGraph {
            nodes,
            id_to_index,
            vectors,
            entry_point,
            dim,
            metric,
        },
        config,
    ))
}

// ============================================================================
// IVFPQIndex Persistence
// ============================================================================

/// Save an [`IVFPQIndex`] to disk in the vecta binary serialization format.
///
/// # Layout:
/// - 22-byte standard header (`index_type = 3`).
/// - `pq config: m: u32, k_per_subvector: u32` (8 bytes).
/// - `num_clusters: u32` (4 bytes).
/// - `centroids: num_clusters * dim * 4 bytes` (full-precision coarse centroids).
/// - `pq codebooks: m * k_per_subvector * sub_dim * 4 bytes`.
/// - `per-cluster sections`: for each coarse cluster, `vector_count: u32`, followed by
///   `vector_count` pairs of `(id: u64, code: [u8; m])`.
/// - `is_trained: u8` (1 byte boolean flag).
pub fn save_ivf_pq_index(index: &IVFPQIndex, path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let dim = index.dim;
    let num_vectors = index.len();
    let num_clusters = index.num_clusters();
    let m = index.pq_config.m;
    let k_per_subvector = index.pq_config.k_per_subvector;
    let sub_dim = dim / m;

    // 1. Standard header (22 bytes)
    write_header(
        &mut writer,
        INDEX_TYPE_IVF_PQ,
        index.metric,
        dim,
        num_vectors,
    )?;

    // 2. PQConfig parameters (8 bytes)
    writer.write_all(&(m as u32).to_le_bytes())?;
    writer.write_all(&(k_per_subvector as u32).to_le_bytes())?;

    // 3. Number of coarse clusters (4 bytes)
    writer.write_all(&(num_clusters as u32).to_le_bytes())?;

    // 4. Coarse Centroids (num_clusters * dim * 4 bytes)
    if index.is_trained {
        write_floats(&mut writer, &index.centroids.data)?;
    } else {
        let dummy = vec![0.0f32; num_clusters * dim];
        write_floats(&mut writer, &dummy)?;
    }

    // 5. PQ Codebooks (m * k_per_subvector * sub_dim * 4 bytes)
    if let Some(ref cb) = index.pq_codebooks {
        for book in &cb.codebooks {
            write_floats(&mut writer, &book.data)?;
        }
    } else {
        let dummy_cb = vec![0.0f32; m * k_per_subvector * sub_dim];
        write_floats(&mut writer, &dummy_cb)?;
    }

    // 6. Per-cluster inverted lists
    for cluster in &index.inverted_lists {
        let count = cluster.len() as u32;
        writer.write_all(&count.to_le_bytes())?;
        for &(id, ref code) in cluster {
            writer.write_all(&id.to_le_bytes())?;
            writer.write_all(code)?;
        }
    }

    // 7. is_trained flag (1 byte)
    writer.write_all(&[if index.is_trained { 1 } else { 0 }])?;

    writer.flush()?;
    Ok(())
}

/// Load an [`IVFPQIndex`] from disk from a vecta binary serialization file.
pub fn load_ivf_pq_index(path: &Path) -> Result<IVFPQIndex, String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open file {}: {}", path.display(), e))?;
    let mut reader = BufReader::new(file);

    // 1. Standard header
    let header = read_header(&mut reader, INDEX_TYPE_IVF_PQ, "IVFPQIndex")?;
    let dim = header.dim;
    let metric = header.metric;

    // 2. PQConfig
    let mut cfg_buf = [0u8; 8];
    reader
        .read_exact(&mut cfg_buf)
        .map_err(|e| format!("failed to read PQ config: {}", e))?;
    let m = u32::from_le_bytes(cfg_buf[0..4].try_into().unwrap()) as usize;
    let k_per_subvector = u32::from_le_bytes(cfg_buf[4..8].try_into().unwrap()) as usize;

    if m == 0 || dim % m != 0 {
        return Err(format!(
            "invalid PQ config: dim {} not evenly divisible by m {}",
            dim, m
        ));
    }
    if k_per_subvector == 0 || k_per_subvector > 256 {
        return Err(format!(
            "invalid PQ config: k_per_subvector {} must be in 1..=256",
            k_per_subvector
        ));
    }
    let sub_dim = dim / m;
    let pq_config = PQConfig {
        m,
        k_per_subvector,
        max_iterations: 100,
    };

    // 3. Number of coarse clusters
    let mut clusters_buf = [0u8; 4];
    reader
        .read_exact(&mut clusters_buf)
        .map_err(|e| format!("failed to read num_clusters: {}", e))?;
    let num_clusters = u32::from_le_bytes(clusters_buf) as usize;

    // 4. Coarse Centroids
    let total_centroid_floats = num_clusters
        .checked_mul(dim)
        .ok_or_else(|| "centroid float count overflows usize".to_string())?;
    let centroid_data = read_floats(&mut reader, total_centroid_floats)?;

    // 5. PQ Codebooks
    let total_cb_floats = m
        .checked_mul(k_per_subvector)
        .and_then(|v| v.checked_mul(sub_dim))
        .ok_or_else(|| "codebook float count overflows usize".to_string())?;
    let cb_data = read_floats(&mut reader, total_cb_floats)?;

    // 6. Per-cluster inverted lists
    let mut inverted_lists = Vec::with_capacity(num_clusters);
    let mut total_loaded_vectors = 0;
    for c in 0..num_clusters {
        let mut count_buf = [0u8; 4];
        reader
            .read_exact(&mut count_buf)
            .map_err(|e| format!("failed to read vector count for cluster {}: {}", c, e))?;
        let count = u32::from_le_bytes(count_buf) as usize;
        total_loaded_vectors += count;

        let mut list = Vec::with_capacity(count);
        let entry_len = 8 + m;
        let mut chunk_raw = vec![0u8; count * entry_len];
        reader
            .read_exact(&mut chunk_raw)
            .map_err(|e| format!("failed to read inverted list {}: {}", c, e))?;

        for chunk in chunk_raw.chunks_exact(entry_len) {
            let id = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let code = chunk[8..entry_len].to_vec();
            list.push((id, code));
        }
        inverted_lists.push(list);
    }

    if total_loaded_vectors != header.num_vectors {
        return Err(format!(
            "mismatch between header vector count ({}) and sum of cluster counts ({})",
            header.num_vectors, total_loaded_vectors
        ));
    }

    // 7. is_trained flag
    let mut trained_buf = [0u8; 1];
    reader
        .read_exact(&mut trained_buf)
        .map_err(|e| format!("failed to read is_trained flag: {}", e))?;
    let is_trained = trained_buf[0] != 0;

    let (centroids, pq_codebooks) = if is_trained {
        let centroids = VectorBatch::from_parts(centroid_data, dim, num_clusters)?;
        let mut codebooks = Vec::with_capacity(m);
        let floats_per_book = k_per_subvector * sub_dim;
        for chunk in cb_data.chunks_exact(floats_per_book) {
            codebooks.push(VectorBatch::from_parts(
                chunk.to_vec(),
                sub_dim,
                k_per_subvector,
            )?);
        }
        let pq_cbs = PQCodebooks {
            m,
            sub_dim,
            k_per_subvector,
            codebooks,
        };
        (centroids, Some(pq_cbs))
    } else {
        (VectorBatch::new(dim), None)
    };

    Ok(IVFPQIndex {
        centroids,
        pq_codebooks,
        inverted_lists,
        dim,
        metric,
        is_trained,
        pq_config,
    })
}

// ============================================================================
// Unit and Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::hnsw::insert::insert;
    use crate::core::kmeans::KMeansConfig;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::time::Instant;

    /// Helper to generate a unique temporary file path for isolated testing.
    fn temp_file_path(prefix: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let rand_val: u64 = rand::thread_rng().gen();
        path.push(format!("{}_{}_{}.vcta", prefix, timestamp, rand_val));
        path
    }

    // ========================================================================
    // Phase 25 Tests (FlatIndex Preservation)
    // ========================================================================

    /// Test 1: Round-trip test:
    /// Build a FlatIndex with 5 hand-crafted vectors, save it, load it back,
    /// confirm the loaded index has IDENTICAL ids, vectors, dim, and metric.
    #[test]
    fn test_round_trip_hand_crafted() {
        let dim = 3;
        let mut index = FlatIndex::new(dim, Metric::Euclidean);

        let vectors: Vec<(u64, [f32; 3])> = vec![
            (10, [1.0, 2.0, 3.0]),
            (20, [4.0, 5.0, 6.0]),
            (30, [7.0, 8.0, 9.0]),
            (40, [0.1, 0.2, 0.3]),
            (50, [10.0, 20.0, 30.0]),
        ];

        for (id, vec) in &vectors {
            index.add(*id, vec);
        }

        let file_path = temp_file_path("test_round_trip");
        save_flat_index(&index, &file_path).unwrap();

        let loaded = load_flat_index(&file_path).unwrap();
        let _ = std::fs::remove_file(&file_path);

        assert_eq!(loaded.dim(), index.dim());
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.metric, index.metric);
        assert_eq!(loaded.ids, index.ids);
        assert_eq!(loaded.batch.data, index.batch.data);
    }

    /// Test 2: Round-trip search consistency:
    /// After save/load, run the SAME query through both the original and reloaded index,
    /// confirm IDENTICAL search() results (id and score).
    #[test]
    fn test_round_trip_search_consistency() {
        let dim = 4;
        let mut original = FlatIndex::new(dim, Metric::Cosine);

        let points: Vec<(u64, [f32; 4])> = vec![
            (1, [1.0, 0.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0, 0.0]),
            (3, [0.5, 0.5, 0.0, 0.0]),
            (4, [0.0, 0.0, 1.0, 1.0]),
            (5, [0.2, 0.8, 0.1, 0.0]),
        ];

        for (id, pt) in &points {
            original.add(*id, pt);
        }

        let file_path = temp_file_path("test_search_consistency");
        save_flat_index(&original, &file_path).unwrap();

        let loaded = load_flat_index(&file_path).unwrap();
        let _ = std::fs::remove_file(&file_path);

        let query = [0.1, 0.9, 0.0, 0.0];
        let original_results = original.search(&query, 3);
        let loaded_results = loaded.search(&query, 3);

        println!("\nPhase 25 Test 2: Search consistency confirmation:");
        for i in 0..original_results.len() {
            println!(
                "  Rank #{}: Original=(id={}, score={:.6}) | Loaded=(id={}, score={:.6})",
                i + 1,
                original_results[i].id,
                original_results[i].score,
                loaded_results[i].id,
                loaded_results[i].score
            );
        }

        assert_eq!(original_results.len(), loaded_results.len());
        for i in 0..original_results.len() {
            assert_eq!(
                original_results[i].id, loaded_results[i].id,
                "ID mismatch at rank {}",
                i
            );
            assert_eq!(
                original_results[i].score, loaded_results[i].score,
                "Score mismatch at rank {}",
                i
            );
        }
    }

    /// Test 3: Loading a file with wrong magic bytes returns clear Err, does not panic.
    #[test]
    fn test_load_invalid_magic_bytes_returns_err() {
        let file_path = temp_file_path("test_bad_magic");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"NOPE").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        drop(file);

        let res = load_flat_index(&file_path);
        let _ = std::fs::remove_file(&file_path);

        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("not a valid vecta index file"),
            "Unexpected error message: {}",
            err
        );
    }

    /// Test 4: Loading a file with a mismatched format_version returns clear Err.
    #[test]
    fn test_load_mismatched_format_version_returns_err() {
        let file_path = temp_file_path("test_bad_version");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(MAGIC_BYTES).unwrap();
        file.write_all(&999u32.to_le_bytes()).unwrap();
        file.write_all(&[INDEX_TYPE_FLAT]).unwrap();
        file.write_all(&[METRIC_EUCLIDEAN]).unwrap();
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        drop(file);

        let res = load_flat_index(&file_path);
        let _ = std::fs::remove_file(&file_path);

        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("unsupported format version 999"),
            "Unexpected error message: {}",
            err
        );
        assert!(
            err.contains("current supported version is 1"),
            "Error message should mention supported version 1: {}",
            err
        );
    }

    /// Test 5: Loading a file with wrong index_type byte returns clear Err.
    #[test]
    fn test_load_wrong_index_type_returns_err() {
        let file_path = temp_file_path("test_wrong_index_type");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(MAGIC_BYTES).unwrap();
        file.write_all(&FORMAT_VERSION.to_le_bytes()).unwrap();
        file.write_all(&[INDEX_TYPE_IVF]).unwrap();
        file.write_all(&[METRIC_EUCLIDEAN]).unwrap();
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        drop(file);

        let res = load_flat_index(&file_path);
        let _ = std::fs::remove_file(&file_path);

        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("file contains a different index type (1)"),
            "Unexpected error message: {}",
            err
        );
    }

    /// Test 6: Loading a genuinely truncated/corrupted file returns Err gracefully.
    #[test]
    fn test_load_truncated_corrupted_file_returns_err() {
        let file_path = temp_file_path("test_truncated");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(MAGIC_BYTES).unwrap();
        file.write_all(&FORMAT_VERSION.to_le_bytes()).unwrap();
        file.write_all(&[INDEX_TYPE_FLAT]).unwrap();
        file.write_all(&[METRIC_EUCLIDEAN]).unwrap();
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&10u64.to_le_bytes()).unwrap();

        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        drop(file);

        let res = load_flat_index(&file_path);
        let _ = std::fs::remove_file(&file_path);

        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("corrupted/truncated file"),
            "Expected truncated file error, got: {}",
            err
        );
    }

    /// Test 7: Realistic-scale round-trip test for FlatIndex.
    #[test]
    fn test_realistic_scale_round_trip_and_file_size() {
        let n = 10000;
        let dim = 128;
        let mut index = FlatIndex::new(dim, Metric::Euclidean);

        let mut rng = StdRng::seed_from_u64(42);
        let mut ids = Vec::with_capacity(n);
        let mut vectors = VectorBatch::new(dim);

        for id in 0..n as u64 {
            ids.push(id);
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-1.0..1.0));
            }
            vectors.push(&v);
        }

        index.add_batch(&ids, &vectors);
        assert_eq!(index.len(), n);

        let file_path = temp_file_path("test_realistic_scale");
        save_flat_index(&index, &file_path).unwrap();

        let metadata = std::fs::metadata(&file_path).unwrap();
        let actual_size = metadata.len();
        let expected_size = 22 + (n as u64 * 8) + (n as u64 * dim as u64 * 4);

        assert_eq!(
            actual_size, expected_size,
            "Written file size does not match theoretical calculation"
        );

        let loaded = load_flat_index(&file_path).unwrap();
        let _ = std::fs::remove_file(&file_path);

        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.dim(), dim);
        assert_eq!(loaded.metric, Metric::Euclidean);

        for &check_idx in &[0, 500, 2500, 7777, 9999] {
            assert_eq!(loaded.ids[check_idx], ids[check_idx]);
            assert_eq!(
                loaded.get_vector(ids[check_idx]).unwrap(),
                vectors.get(check_idx)
            );
        }
    }

    /// Test 8: Timing test for FlatIndex.
    #[test]
    fn test_save_load_timing_benchmark() {
        let n = 10000;
        let dim = 128;
        let mut index = FlatIndex::new(dim, Metric::Euclidean);

        let mut rng = StdRng::seed_from_u64(12345);
        let mut ids = Vec::with_capacity(n);
        let mut vectors = VectorBatch::new(dim);

        for id in 0..n as u64 {
            ids.push(id);
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-1.0..1.0));
            }
            vectors.push(&v);
        }

        index.add_batch(&ids, &vectors);

        let file_path = temp_file_path("test_timing_benchmark");

        let start_save = Instant::now();
        save_flat_index(&index, &file_path).unwrap();
        let elapsed_save = start_save.elapsed();

        let start_load = Instant::now();
        let loaded = load_flat_index(&file_path).unwrap();
        let elapsed_load = start_load.elapsed();

        let _ = std::fs::remove_file(&file_path);

        assert_eq!(loaded.len(), n);

        let file_mb = (22 + n * 8 + n * dim * 4) as f64 / (1024.0 * 1024.0);

        println!(
            "\nPhase 25 Test 8: Save/Load timing benchmark (N={}, dim={}, file={:.2} MB):",
            n, dim, file_mb
        );
        println!(
            "  Save time: {:.2?} ({:.1} MB/s)",
            elapsed_save,
            file_mb / elapsed_save.as_secs_f64()
        );
        println!(
            "  Load time: {:.2?} ({:.1} MB/s)",
            elapsed_load,
            file_mb / elapsed_load.as_secs_f64()
        );
    }

    // ========================================================================
    // Phase 26 Tests (IVFIndex, HnswGraph, IVFPQIndex, Peek Dispatch)
    // ========================================================================

    /// IVF Test 1 & 2: Round-trip structural equality and search consistency.
    #[test]
    fn test_ivf_round_trip_and_search_consistency() {
        let dim = 4;
        let num_clusters = 3;
        let mut index = IVFIndex::new(dim, num_clusters, Metric::Euclidean);

        let train_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 1.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 1.1, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.1, 1.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1.1,
        ];
        let training_batch = VectorBatch::from_parts(train_data, dim, 8).unwrap();
        let kmeans_cfg = KMeansConfig {
            k: num_clusters,
            max_iterations: 20,
            tolerance: 1e-4,
        };
        index.train(&training_batch, &kmeans_cfg, 42);

        let doc_ids = vec![101, 102, 103, 104, 105];
        let add_data: Vec<f32> = vec![
            1.05, 0.05, 0.0, 0.0, 0.05, 1.05, 0.0, 0.0, 0.0, 0.05, 1.05, 0.0, 0.05, 0.0, 0.0, 1.05,
            1.0, 0.2, 0.0, 0.0,
        ];
        let add_batch = VectorBatch::from_parts(add_data, dim, 5).unwrap();
        index.add_batch(&doc_ids, &add_batch).unwrap();

        let file_path = temp_file_path("test_ivf_round_trip");
        save_ivf_index(&index, &file_path).unwrap();

        let loaded = load_ivf_index(&file_path).unwrap();
        let _ = std::fs::remove_file(&file_path);

        // Structural assertions
        assert_eq!(loaded.dim, index.dim);
        assert_eq!(loaded.metric, index.metric);
        assert_eq!(loaded.is_trained, index.is_trained);
        assert_eq!(loaded.num_clusters(), index.num_clusters());
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.centroids.data, index.centroids.data);
        for c in 0..num_clusters {
            assert_eq!(loaded.inverted_lists[c].ids, index.inverted_lists[c].ids);
            assert_eq!(
                loaded.inverted_lists[c].batch.data,
                index.inverted_lists[c].batch.data
            );
        }

        // Search consistency confirmation
        let query = [1.0, 0.0, 0.0, 0.0];
        let orig_res = index.search(&query, 3, 2);
        let loaded_res = loaded.search(&query, 3, 2);

        println!("\nPhase 26 Confirmation (IVF Search Consistency):");
        for i in 0..orig_res.len() {
            println!(
                "  Rank #{}: Original=(id={}, score={:.6}) | Loaded=(id={}, score={:.6})",
                i + 1,
                orig_res[i].id,
                orig_res[i].score,
                loaded_res[i].id,
                loaded_res[i].score
            );
        }

        assert_eq!(orig_res.len(), loaded_res.len());
        for i in 0..orig_res.len() {
            assert_eq!(orig_res[i].id, loaded_res[i].id);
            assert_eq!(orig_res[i].score, loaded_res[i].score);
        }
    }

    /// HNSW Test 1 & 2: Round-trip structural equality and search consistency.
    #[test]
    fn test_hnsw_round_trip_and_search_consistency() {
        let dim = 4;
        let mut graph = HnswGraph::new(dim, Metric::Euclidean);
        let config = HnswConfig {
            m: 4,
            ef_construction: 32,
            ef_search: 16,
        };
        let mut rng = StdRng::seed_from_u64(100);

        let vectors: Vec<(u64, [f32; 4])> = vec![
            (1, [1.0, 0.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0, 0.0]),
            (3, [0.0, 0.0, 1.0, 0.0]),
            (4, [0.0, 0.0, 0.0, 1.0]),
            (5, [0.5, 0.5, 0.0, 0.0]),
            (6, [0.0, 0.5, 0.5, 0.0]),
            (7, [0.1, 0.1, 0.1, 0.1]),
        ];

        for (id, vec) in &vectors {
            insert(&mut graph, *id, vec, &config, &mut rng).unwrap();
        }

        let file_path = temp_file_path("test_hnsw_round_trip");
        save_hnsw_index(&graph, &config, &file_path).unwrap();

        let (loaded_graph, loaded_config) = load_hnsw_index(&file_path).unwrap();
        let _ = std::fs::remove_file(&file_path);

        // Structural assertions
        assert_eq!(loaded_graph.dim, graph.dim);
        assert_eq!(loaded_graph.metric, graph.metric);
        assert_eq!(loaded_graph.entry_point, graph.entry_point);
        assert_eq!(loaded_graph.len(), graph.len());
        assert_eq!(loaded_config.m, config.m);
        assert_eq!(loaded_config.ef_construction, config.ef_construction);
        assert_eq!(loaded_config.ef_search, config.ef_search);
        assert_eq!(loaded_graph.vectors.data, graph.vectors.data);

        for i in 0..graph.nodes.len() {
            assert_eq!(loaded_graph.nodes[i].id, graph.nodes[i].id);
            assert_eq!(loaded_graph.nodes[i].vector_idx, graph.nodes[i].vector_idx);
            assert_eq!(loaded_graph.nodes[i].max_layer, graph.nodes[i].max_layer);
            assert_eq!(loaded_graph.nodes[i].neighbors, graph.nodes[i].neighbors);
        }

        // Search consistency confirmation
        let query = [0.1, 0.8, 0.1, 0.0];
        let orig_res = graph.search(&query, 3, config.ef_search);
        let loaded_res = loaded_graph.search(&query, 3, loaded_config.ef_search);

        println!("\nPhase 26 Confirmation (HNSW Search Consistency):");
        for i in 0..orig_res.len() {
            println!(
                "  Rank #{}: Original=(id={}, score={:.6}) | Loaded=(id={}, score={:.6})",
                i + 1,
                orig_res[i].id,
                orig_res[i].score,
                loaded_res[i].id,
                loaded_res[i].score
            );
        }

        assert_eq!(orig_res.len(), loaded_res.len());
        for i in 0..orig_res.len() {
            assert_eq!(orig_res[i].id, loaded_res[i].id);
            assert_eq!(orig_res[i].score, loaded_res[i].score);
        }
    }

    /// HNSW Test 5: Spot-check entry point and neighbor lists across multiple layers.
    #[test]
    fn test_hnsw_entry_point_and_neighbor_spot_check() {
        let dim = 8;
        let mut graph = HnswGraph::new(dim, Metric::Euclidean);
        let config = HnswConfig {
            m: 8,
            ef_construction: 64,
            ef_search: 32,
        };
        let mut rng = StdRng::seed_from_u64(999);

        for id in 0..50u64 {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-1.0..1.0));
            }
            insert(&mut graph, id, &v, &config, &mut rng).unwrap();
        }

        let file_path = temp_file_path("test_hnsw_spot_check");
        save_hnsw_index(&graph, &config, &file_path).unwrap();

        let (loaded, _) = load_hnsw_index(&file_path).unwrap();
        let _ = std::fs::remove_file(&file_path);

        assert_eq!(loaded.entry_point, graph.entry_point);

        println!("\nPhase 26 Test 5: HNSW Entry Point & Neighbor List Spot-Check:");
        println!("  Graph Entry Point: {:?}", loaded.entry_point);

        // Spot check 3 different nodes
        for &node_idx in &[0, 15, 35] {
            let orig_node = &graph.nodes[node_idx];
            let loaded_node = &loaded.nodes[node_idx];

            println!(
                "  Node index={}, id={}, max_layer={}:",
                node_idx, orig_node.id, orig_node.max_layer
            );
            for layer in 0..=orig_node.max_layer {
                println!(
                    "    Layer {}: Original={:?} | Loaded={:?}",
                    layer, orig_node.neighbors[layer], loaded_node.neighbors[layer]
                );
                assert_eq!(
                    orig_node.neighbors[layer], loaded_node.neighbors[layer],
                    "Neighbor mismatch at node {}, layer {}",
                    node_idx, layer
                );
            }
        }
    }

    /// IVFPQ Test 1 & 2: Round-trip structural equality and search consistency.
    #[test]
    fn test_ivf_pq_round_trip_and_search_consistency() {
        let dim = 8;
        let num_clusters = 4;
        let pq_config = PQConfig {
            m: 2,
            k_per_subvector: 4,
            max_iterations: 20,
        };

        let mut index = IVFPQIndex::new(dim, num_clusters, pq_config).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let n_train = 60;
        let mut train_data = Vec::with_capacity(n_train * dim);
        for _ in 0..(n_train * dim) {
            train_data.push(rng.gen_range(-2.0..2.0));
        }
        let train_batch = VectorBatch::from_parts(train_data, dim, n_train).unwrap();
        let kmeans_cfg = KMeansConfig {
            k: num_clusters,
            max_iterations: 15,
            tolerance: 1e-4,
        };
        index.train(&train_batch, &kmeans_cfg, 10, 20).unwrap();

        let mut ids = Vec::new();
        let mut add_batch = VectorBatch::new(dim);
        for id in 100..120u64 {
            ids.push(id);
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-2.0..2.0));
            }
            add_batch.push(&v);
        }
        index.add_batch(&ids, &add_batch).unwrap();

        let file_path = temp_file_path("test_ivf_pq_round_trip");
        save_ivf_pq_index(&index, &file_path).unwrap();

        let loaded = load_ivf_pq_index(&file_path).unwrap();
        let _ = std::fs::remove_file(&file_path);

        // Structural assertions
        assert_eq!(loaded.dim, index.dim);
        assert_eq!(loaded.is_trained, index.is_trained);
        assert_eq!(loaded.num_clusters(), index.num_clusters());
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.pq_config.m, index.pq_config.m);
        assert_eq!(
            loaded.pq_config.k_per_subvector,
            index.pq_config.k_per_subvector
        );
        assert_eq!(loaded.centroids.data, index.centroids.data);

        for c in 0..num_clusters {
            assert_eq!(loaded.inverted_lists[c], index.inverted_lists[c]);
        }

        // Search consistency confirmation
        let mut query = vec![0.0f32; dim];
        for val in &mut query {
            *val = rng.gen_range(-1.0..1.0);
        }

        let orig_res = index.search(&query, 3, 2).unwrap();
        let loaded_res = loaded.search(&query, 3, 2).unwrap();

        println!("\nPhase 26 Confirmation (IVFPQ Search Consistency):");
        for i in 0..orig_res.len() {
            println!(
                "  Rank #{}: Original=(id={}, score={:.6}) | Loaded=(id={}, score={:.6})",
                i + 1,
                orig_res[i].id,
                orig_res[i].score,
                loaded_res[i].id,
                loaded_res[i].score
            );
        }

        assert_eq!(orig_res.len(), loaded_res.len());
        for i in 0..orig_res.len() {
            assert_eq!(orig_res[i].id, loaded_res[i].id);
            assert_eq!(orig_res[i].score, loaded_res[i].score);
        }
    }

    /// Cross-cutting Test 3: Wrong index type returns clear Err, no panic.
    #[test]
    fn test_mismatched_index_type_loading_returns_err() {
        let dim = 4;
        let mut ivf = IVFIndex::new(dim, 2, Metric::Euclidean);
        let train_data = vec![0.0f32; 8];
        let batch = VectorBatch::from_parts(train_data, dim, 2).unwrap();
        let cfg = KMeansConfig {
            k: 2,
            max_iterations: 1,
            tolerance: 1e-4,
        };
        ivf.train(&batch, &cfg, 1);

        let file_path = temp_file_path("test_mismatch");
        save_ivf_index(&ivf, &file_path).unwrap();

        // Try loading IVF file using Flat loader
        let flat_res = load_flat_index(&file_path);
        assert!(flat_res.is_err());
        let err_flat = flat_res.unwrap_err();
        assert!(
            err_flat.contains("file contains a different index type (1), expected FlatIndex (0)")
        );

        // Try loading IVF file using HNSW loader
        let hnsw_res = load_hnsw_index(&file_path);
        assert!(hnsw_res.is_err());
        let err_hnsw = hnsw_res.unwrap_err();
        assert!(
            err_hnsw.contains("file contains a different index type (1), expected HnswGraph (2)")
        );

        // Try loading IVF file using IVFPQ loader
        let pq_res = load_ivf_pq_index(&file_path);
        assert!(pq_res.is_err());
        let err_pq = pq_res.unwrap_err();
        assert!(
            err_pq.contains("file contains a different index type (1), expected IVFPQIndex (3)")
        );

        let _ = std::fs::remove_file(&file_path);
    }

    /// Cross-cutting Test 6: peek_index_type correctly identifies all 4 saved files.
    #[test]
    fn test_peek_index_type_all_four_types() {
        let dim = 4;

        // 1. Flat
        let mut flat = FlatIndex::new(dim, Metric::Euclidean);
        flat.add(1, &[1.0, 2.0, 3.0, 4.0]);
        let p_flat = temp_file_path("peek_flat");
        save_flat_index(&flat, &p_flat).unwrap();
        assert_eq!(peek_index_type(&p_flat).unwrap(), INDEX_TYPE_FLAT);
        let _ = std::fs::remove_file(&p_flat);

        // 2. IVF
        let mut ivf = IVFIndex::new(dim, 1, Metric::Euclidean);
        let batch = VectorBatch::from_parts(vec![1.0, 2.0, 3.0, 4.0], dim, 1).unwrap();
        let cfg = KMeansConfig {
            k: 1,
            max_iterations: 1,
            tolerance: 1e-4,
        };
        ivf.train(&batch, &cfg, 1);
        let p_ivf = temp_file_path("peek_ivf");
        save_ivf_index(&ivf, &p_ivf).unwrap();
        assert_eq!(peek_index_type(&p_ivf).unwrap(), INDEX_TYPE_IVF);
        let _ = std::fs::remove_file(&p_ivf);

        // 3. HNSW
        let mut hnsw = HnswGraph::new(dim, Metric::Euclidean);
        let hcfg = HnswConfig::default();
        let mut rng = StdRng::seed_from_u64(1);
        insert(&mut hnsw, 1, &[1.0, 2.0, 3.0, 4.0], &hcfg, &mut rng).unwrap();
        let p_hnsw = temp_file_path("peek_hnsw");
        save_hnsw_index(&hnsw, &hcfg, &p_hnsw).unwrap();
        assert_eq!(peek_index_type(&p_hnsw).unwrap(), INDEX_TYPE_HNSW);
        let _ = std::fs::remove_file(&p_hnsw);

        // 4. IVFPQ
        let pq_cfg = PQConfig {
            m: 2,
            k_per_subvector: 2,
            max_iterations: 1,
        };
        let mut pq_idx = IVFPQIndex::new(dim, 1, pq_cfg).unwrap();
        let pq_train =
            VectorBatch::from_parts(vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], dim, 2).unwrap();
        let k_cfg = KMeansConfig {
            k: 1,
            max_iterations: 1,
            tolerance: 1e-4,
        };
        pq_idx.train(&pq_train, &k_cfg, 1, 1).unwrap();
        let p_pq = temp_file_path("peek_pq");
        save_ivf_pq_index(&pq_idx, &p_pq).unwrap();
        assert_eq!(peek_index_type(&p_pq).unwrap(), INDEX_TYPE_IVF_PQ);
        let _ = std::fs::remove_file(&p_pq);
    }

    /// Comparison Test: Side-by-side file size and save/load benchmark for all 4 index types.
    #[test]
    fn test_side_by_side_all_four_index_types() {
        let n = 1000;
        let dim = 128;
        let mut rng = StdRng::seed_from_u64(42);

        let mut ids = Vec::with_capacity(n);
        let mut dataset = VectorBatch::new(dim);
        for id in 0..n as u64 {
            ids.push(id);
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(rng.gen_range(-1.0..1.0));
            }
            dataset.push(&v);
        }

        println!(
            "\n================================================================================"
        );
        println!(
            "Phase 26: Side-by-Side Index Persistence Benchmark (N={}, dim={})",
            n, dim
        );
        println!(
            "================================================================================"
        );

        // 1. FlatIndex
        let mut flat = FlatIndex::new(dim, Metric::Euclidean);
        flat.add_batch(&ids, &dataset);
        let p_flat = temp_file_path("bench_flat");

        let t0 = Instant::now();
        save_flat_index(&flat, &p_flat).unwrap();
        let flat_save_dur = t0.elapsed();
        let flat_sz = std::fs::metadata(&p_flat).unwrap().len();

        let t0 = Instant::now();
        let _ = load_flat_index(&p_flat).unwrap();
        let flat_load_dur = t0.elapsed();
        let _ = std::fs::remove_file(&p_flat);

        // 2. IVFIndex (num_clusters = 16)
        let num_clusters = 16;
        let mut ivf = IVFIndex::new(dim, num_clusters, Metric::Euclidean);
        let km_cfg = KMeansConfig {
            k: num_clusters,
            max_iterations: 10,
            tolerance: 1e-4,
        };
        ivf.train(&dataset, &km_cfg, 1);
        ivf.add_batch(&ids, &dataset).unwrap();
        let p_ivf = temp_file_path("bench_ivf");

        let t0 = Instant::now();
        save_ivf_index(&ivf, &p_ivf).unwrap();
        let ivf_save_dur = t0.elapsed();
        let ivf_sz = std::fs::metadata(&p_ivf).unwrap().len();

        let t0 = Instant::now();
        let _ = load_ivf_index(&p_ivf).unwrap();
        let ivf_load_dur = t0.elapsed();
        let _ = std::fs::remove_file(&p_ivf);

        // 3. HnswGraph (M = 16, ef_c = 64, ef_s = 32)
        let hnsw_cfg = HnswConfig {
            m: 16,
            ef_construction: 64,
            ef_search: 32,
        };
        let mut hnsw = HnswGraph::new(dim, Metric::Euclidean);
        let mut hrng = StdRng::seed_from_u64(123);
        for (i, &id) in ids.iter().enumerate() {
            insert(&mut hnsw, id, dataset.get(i), &hnsw_cfg, &mut hrng).unwrap();
        }
        let p_hnsw = temp_file_path("bench_hnsw");

        let t0 = Instant::now();
        save_hnsw_index(&hnsw, &hnsw_cfg, &p_hnsw).unwrap();
        let hnsw_save_dur = t0.elapsed();
        let hnsw_sz = std::fs::metadata(&p_hnsw).unwrap().len();

        let t0 = Instant::now();
        let _ = load_hnsw_index(&p_hnsw).unwrap();
        let hnsw_load_dur = t0.elapsed();
        let _ = std::fs::remove_file(&p_hnsw);

        // 4. IVFPQIndex (m = 8, k = 256, clusters = 16)
        let pq_cfg = PQConfig {
            m: 8,
            k_per_subvector: 256,
            max_iterations: 10,
        };
        let mut ivf_pq = IVFPQIndex::new(dim, num_clusters, pq_cfg).unwrap();
        let km_pq_cfg = KMeansConfig {
            k: num_clusters,
            max_iterations: 10,
            tolerance: 1e-4,
        };
        ivf_pq.train(&dataset, &km_pq_cfg, 1, 2).unwrap();
        ivf_pq.add_batch(&ids, &dataset).unwrap();
        let p_pq = temp_file_path("bench_pq");

        let t0 = Instant::now();
        save_ivf_pq_index(&ivf_pq, &p_pq).unwrap();
        let pq_save_dur = t0.elapsed();
        let pq_sz = std::fs::metadata(&p_pq).unwrap().len();

        let t0 = Instant::now();
        let _ = load_ivf_pq_index(&p_pq).unwrap();
        let pq_load_dur = t0.elapsed();
        let _ = std::fs::remove_file(&p_pq);

        println!(
            "{:<14} | {:>14} | {:>12} | {:>12}",
            "Index Type", "File Size", "Save Time", "Load Time"
        );
        println!("{:-<14}-|-{:-<14}-|-{:-<12}-|-{:-<12}", "", "", "", "");
        println!(
            "{:<14} | {:>11.2} KB | {:>12.2?} | {:>12.2?}",
            "FlatIndex",
            flat_sz as f64 / 1024.0,
            flat_save_dur,
            flat_load_dur
        );
        println!(
            "{:<14} | {:>11.2} KB | {:>12.2?} | {:>12.2?}",
            "IVFIndex",
            ivf_sz as f64 / 1024.0,
            ivf_save_dur,
            ivf_load_dur
        );
        println!(
            "{:<14} | {:>11.2} KB | {:>12.2?} | {:>12.2?}",
            "HnswGraph",
            hnsw_sz as f64 / 1024.0,
            hnsw_save_dur,
            hnsw_load_dur
        );
        println!(
            "{:<14} | {:>11.2} KB | {:>12.2?} | {:>12.2?}",
            "IVFPQIndex",
            pq_sz as f64 / 1024.0,
            pq_save_dur,
            pq_load_dur
        );
        println!(
            "================================================================================"
        );
    }
}
