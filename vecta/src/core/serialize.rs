//! Binary serialization format for persisting vecta indexes to disk.
//!
//! # On-Disk File Layout (FlatIndex)
//!
//! Indexes are serialized with a fixed-size header followed by contiguous data arrays.
//! Every multi-byte numerical primitive is encoded using **explicit little-endian**
//! byte ordering (`to_le_bytes` / `from_le_bytes`) to guarantee full cross-platform portability.
//!
//! ```text
//! [HEADER] (22 bytes total)
//!   - magic_bytes:    [u8; 4] = b"VCTA"
//!   - format_version: u32     (current = 1)
//!   - index_type:     u8      (0 = FlatIndex, 1 = IVFIndex, 2 = HnswIndex, 3 = IVFPQIndex)
//!   - metric:         u8      (0 = Euclidean, 1 = Cosine, 2 = DotProduct)
//!   - dim:            u32     (vector dimensionality)
//!   - num_vectors:    u64     (total indexed vectors)
//!
//! [IDS SECTION]
//!   - num_vectors * 8 bytes (each u64 in little-endian, contiguous)
//!
//! [VECTOR DATA SECTION]
//!   - num_vectors * dim * 4 bytes (each f32 in little-endian, row-major, contiguous)
//! ```
//!
//! This layout matches [`VectorBatch`](crate::core::batch::VectorBatch)'s in-memory contiguous
//! flat buffer layout, making disk writing and loading close to a direct memory stream.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::core::batch::VectorBatch;
use crate::core::flat_index::{FlatIndex, Metric};

/// Magic identification bytes at the beginning of every vecta index file (`b"VCTA"`).
pub const MAGIC_BYTES: &[u8; 4] = b"VCTA";

/// Current serialization format version.
pub const FORMAT_VERSION: u32 = 1;

/// Index type identifiers stored in the header byte.
pub const INDEX_TYPE_FLAT: u8 = 0;
/// Reserved for future IVFIndex persistence.
pub const INDEX_TYPE_IVF: u8 = 1;
/// Reserved for future HnswIndex persistence.
pub const INDEX_TYPE_HNSW: u8 = 2;
/// Reserved for future IVFPQIndex persistence.
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

/// Save a [`FlatIndex`] to disk in the vecta binary serialization format.
///
/// # Layout:
/// - Writes 22-byte header: magic bytes, version, index type, metric, dim, and vector count.
/// - Writes all external vector IDs sequentially as little-endian `u64`s.
/// - Writes all vector float values sequentially in row-major order as little-endian `f32`s.
///
/// Uses [`BufWriter`] to minimize syscall overhead on large indexes.
///
/// # Errors
/// Returns an [`std::io::Error`] if file creation or any write operation fails.
pub fn save_flat_index(index: &FlatIndex, path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let dim = index.dim();
    let num_vectors = index.len();

    // 1. Write Header (22 bytes)
    writer.write_all(MAGIC_BYTES)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
    writer.write_all(&[INDEX_TYPE_FLAT])?;
    writer.write_all(&[metric_to_code(index.metric)])?;
    writer.write_all(&(dim as u32).to_le_bytes())?;
    writer.write_all(&(num_vectors as u64).to_le_bytes())?;

    // 2. Write IDs Section (num_vectors * 8 bytes)
    // Buffer IDs in 8KB chunks to balance memory and syscalls
    let mut id_buf = Vec::with_capacity(8192);
    for &id in &index.ids {
        id_buf.extend_from_slice(&id.to_le_bytes());
        if id_buf.len() >= 8192 {
            writer.write_all(&id_buf)?;
            id_buf.clear();
        }
    }
    if !id_buf.is_empty() {
        writer.write_all(&id_buf)?;
    }

    // 3. Write Vector Data Section (num_vectors * dim * 4 bytes)
    // Buffer float bytes in 32KB chunks
    let mut float_buf = Vec::with_capacity(32768);
    for &val in &index.batch.data {
        float_buf.extend_from_slice(&val.to_le_bytes());
        if float_buf.len() >= 32768 {
            writer.write_all(&float_buf)?;
            float_buf.clear();
        }
    }
    if !float_buf.is_empty() {
        writer.write_all(&float_buf)?;
    }

    writer.flush()?;
    Ok(())
}

/// Load a [`FlatIndex`] from disk from a vecta binary serialization file.
///
/// Validates:
/// - Magic bytes (`b"VCTA"`)
/// - Format version matching [`FORMAT_VERSION`]
/// - Index type matching [`INDEX_TYPE_FLAT`]
/// - Valid metric code
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

    // 1. Validate Magic Bytes (4 bytes)
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("failed to read magic bytes: {}", e))?;
    if &magic != MAGIC_BYTES {
        return Err("not a valid vecta index file".to_string());
    }

    // 2. Validate Format Version (4 bytes, little-endian u32)
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

    // 3. Validate Index Type (1 byte)
    let mut index_type_buf = [0u8; 1];
    reader
        .read_exact(&mut index_type_buf)
        .map_err(|e| format!("failed to read index type: {}", e))?;
    let index_type = index_type_buf[0];
    if index_type != INDEX_TYPE_FLAT {
        return Err(format!(
            "file contains a different index type ({}), expected FlatIndex ({})",
            index_type, INDEX_TYPE_FLAT
        ));
    }

    // 4. Validate Metric (1 byte)
    let mut metric_buf = [0u8; 1];
    reader
        .read_exact(&mut metric_buf)
        .map_err(|e| format!("failed to read metric code: {}", e))?;
    let metric = code_to_metric(metric_buf[0])?;

    // 5. Read Dimension (4 bytes, little-endian u32)
    let mut dim_buf = [0u8; 4];
    reader
        .read_exact(&mut dim_buf)
        .map_err(|e| format!("failed to read dimension: {}", e))?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    if dim == 0 {
        return Err("invalid serialized index dimension 0".to_string());
    }

    // 6. Read Number of Vectors (8 bytes, little-endian u64)
    let mut count_buf = [0u8; 8];
    reader
        .read_exact(&mut count_buf)
        .map_err(|e| format!("failed to read vector count: {}", e))?;
    let num_vectors = u64::from_le_bytes(count_buf) as usize;

    // 7. Read IDs Section (num_vectors * 8 bytes)
    let mut ids = Vec::with_capacity(num_vectors);
    let mut id_raw = [0u8; 8];
    for _ in 0..num_vectors {
        reader
            .read_exact(&mut id_raw)
            .map_err(|e| format!("failed to read ID (corrupted/truncated file): {}", e))?;
        ids.push(u64::from_le_bytes(id_raw));
    }

    // 8. Read Vector Data Section (num_vectors * dim * 4 bytes)
    let total_floats = num_vectors
        .checked_mul(dim)
        .ok_or_else(|| "total float count overflows usize".to_string())?;
    let mut data = Vec::with_capacity(total_floats);

    // Read vector floats in 32KB chunks
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

    // 9. Reconstruct index without re-validating uniqueness (data in valid file was already verified)
    let batch = VectorBatch::from_parts(data, dim, num_vectors)?;
    let index = FlatIndex::from_parts(batch, ids, metric)?;

    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
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

        // Write garbage magic bytes
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
        file.write_all(&999u32.to_le_bytes()).unwrap(); // Version 999
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
        file.write_all(&[INDEX_TYPE_IVF]).unwrap(); // Type 1 (IVFIndex) instead of FlatIndex
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

    /// Test 6: Loading a genuinely truncated/corrupted file returns Err gracefully, does not panic.
    #[test]
    fn test_load_truncated_corrupted_file_returns_err() {
        let file_path = temp_file_path("test_truncated");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(MAGIC_BYTES).unwrap();
        file.write_all(&FORMAT_VERSION.to_le_bytes()).unwrap();
        file.write_all(&[INDEX_TYPE_FLAT]).unwrap();
        file.write_all(&[METRIC_EUCLIDEAN]).unwrap();
        file.write_all(&4u32.to_le_bytes()).unwrap(); // dim = 4
        file.write_all(&10u64.to_le_bytes()).unwrap(); // num_vectors = 10 (expected 10 IDs + 40 floats)

        // Write only 2 IDs instead of 10
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

    /// Test 7: Realistic-scale round-trip test:
    /// Build FlatIndex with 10,000 random 128-dim vectors, save to disk, load back,
    /// confirm len() matches and spot-check vectors match.
    /// Print file size and confirm it matches theoretical calculation:
    /// `22 + 10,000*8 + 10,000*128*4 = 5,200,022 bytes`.
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

        // Check file size
        let metadata = std::fs::metadata(&file_path).unwrap();
        let actual_size = metadata.len();
        let expected_size = 22 + (n as u64 * 8) + (n as u64 * dim as u64 * 4);

        println!(
            "\nPhase 25 Test 7: Realistic scale file size check (N={}, dim={}):",
            n, dim
        );
        println!(
            "  Theoretical expected file size: {} bytes ({:.2} MB)",
            expected_size,
            expected_size as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  Actual written file size:       {} bytes ({:.2} MB)",
            actual_size,
            actual_size as f64 / (1024.0 * 1024.0)
        );

        assert_eq!(
            actual_size, expected_size,
            "Written file size does not match theoretical calculation"
        );

        // Load back and verify
        let loaded = load_flat_index(&file_path).unwrap();
        let _ = std::fs::remove_file(&file_path);

        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.dim(), dim);
        assert_eq!(loaded.metric, Metric::Euclidean);

        // Spot check specific vectors
        for &check_idx in &[0, 500, 2500, 7777, 9999] {
            assert_eq!(loaded.ids[check_idx], ids[check_idx]);
            assert_eq!(
                loaded.get_vector(ids[check_idx]).unwrap(),
                vectors.get(check_idx)
            );
        }
    }

    /// Test 8: Timing test:
    /// Time how long save and load each take for the 10,000-vector case (dim=128).
    /// Print both timings.
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
}
