//! Write-Ahead Log (WAL) and checkpointing for crash-safe live mutations.
//!
//! This module provides durability and crash-safety for [`FlatIndex`]:
//! - [`WalOp`]: Operations recorded in the write-ahead log.
//! - [`WalWriter`]: Append-only log writer that flushes and fsyncs every operation.
//! - [`replay_wal`]: Parse operations from an on-disk WAL, gracefully dropping incomplete trailing entries.
//! - [`checkpoint`]: Atomic snapshot and WAL truncation protocol.
//! - [`load_with_wal_recovery`]: Restore an index from a base snapshot and catch it up with uncheckpointed log entries.
//!
//! # Binary Layout Specification
//!
//! Each entry in the WAL file uses length-prefixed framing:
//!
//! ```text
//! [ENTRY]
//!   - entry_len:   u32 (LE)  (byte length of the payload that follows)
//!   - op_type:     u8        (0 = Insert)
//!   - id:          u64 (LE)  (external vector ID)
//!   - dim:         u32 (LE)  (vector dimensionality)
//!   - vector data: dim * 4 bytes (contiguous f32 in little-endian)
//! ```
//!
//! ### Crash-Safe Framing Design
//! The 4-byte `entry_len` prefix specifies the exact byte size of the payload (`1 + 8 + 4 + dim * 4`).
//! During log replay, if the file ends before `entry_len` bytes can be read (for instance, due to
//! a process crash or power cut mid-write), [`replay_wal`] detects this truncated trailing entry,
//! drops it gracefully, and returns all successfully written operations up to that point without
//! panicking or failing. This length-prefixed framing is the fundamental design pattern enabling
//! crash-safe WAL recovery.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::core::flat_index::FlatIndex;
use crate::core::serialize::{load_flat_index, save_flat_index};

/// A write-ahead log operation representing an uncommitted or recovering mutation.
///
/// NOTE: A `Delete` variant is omitted for now because `FlatIndex` does not yet support
/// vector deletion. It will be added in a future phase when index-level deletions are supported.
#[derive(Debug, Clone, PartialEq)]
pub enum WalOp {
    /// Insert a vector with its external identifier.
    Insert {
        /// External vector ID.
        id: u64,
        /// Vector coordinate components.
        vector: Vec<f32>,
    },
}

/// Append-only write-ahead log writer.
///
/// Wraps a [`BufWriter<File>`] opened in append mode.
pub struct WalWriter {
    writer: BufWriter<File>,
    path: PathBuf,
}

/// Open or create a write-ahead log file in append mode.
///
/// If the file does not exist, it will be created. If it already exists, new entries
/// will be appended to the end of the file.
///
/// # Errors
/// Returns an [`std::io::Error`] if the file cannot be opened or created.
pub fn open_wal_writer(path: &Path) -> std::io::Result<WalWriter> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    Ok(WalWriter {
        writer: BufWriter::new(file),
        path: path.to_path_buf(),
    })
}

impl WalWriter {
    /// Return the path to the write-ahead log file.
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Log an insert operation to the WAL file.
    ///
    /// Writes the length-prefixed binary frame `[entry_len: u32][op_type: u8][id: u64][dim: u32][data: dim*4 bytes]`.
    ///
    /// # Durability & Performance Tradeoff
    /// This method calls both `self.writer.flush()?` (flushing the user-space `BufWriter` buffer to the OS)
    /// and `self.writer.get_ref().sync_all()?` (issuing an `fsync` / `FlushFileBuffers` system call).
    ///
    /// **Durability**: Ensures that data is physically committed to durable storage before this method
    /// returns. If the system experiences a power loss or kernel panic immediately after `log_insert`
    /// returns, the operation is guaranteed to survive and be recovered during replay.
    ///
    /// **Performance Tradeoff**: An fsync syscall requires synchronous I/O barrier execution on the underlying
    /// storage medium (SSD/HDD), which incurs substantial latency (often 1-10 ms per write) compared to
    /// asynchronous buffered I/O. In high-throughput streaming workloads, this limits write throughput to
    /// the physical IOPS capacity of the disk unless writes are batched or group-committed.
    ///
    /// # Errors
    /// Returns an [`std::io::Error`] if formatting, writing, flushing, or syncing fails.
    pub fn log_insert(&mut self, id: u64, vector: &[f32]) -> std::io::Result<()> {
        let dim = vector.len();
        let payload_len = 1 + 8 + 4 + (dim * 4);
        let entry_len = u32::try_from(payload_len)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        // 1. Length prefix
        self.writer.write_all(&entry_len.to_le_bytes())?;

        // 2. Op type (0 = Insert)
        self.writer.write_all(&[0u8])?;

        // 3. ID
        self.writer.write_all(&id.to_le_bytes())?;

        // 4. Dimension
        self.writer.write_all(&(dim as u32).to_le_bytes())?;

        // 5. Vector floats
        for &coord in vector {
            self.writer.write_all(&coord.to_le_bytes())?;
        }

        // Flush user-space buffer and fsync to physical storage
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;

        Ok(())
    }
}

/// Replay operations from a write-ahead log file.
///
/// Reads all length-prefixed log entries in chronological order.
///
/// # Crash-Safe Truncation Handling
/// If the file does not exist, returns `Ok(Vec::new())`.
/// If the file ends mid-entry (e.g. fewer bytes remain than `entry_len` specifies, or an incomplete
/// 4-byte length prefix), this function does **NOT** error or panic. It gracefully drops the incomplete
/// trailing entry and returns all fully-written operations up to that point. This handles the exact
/// condition where a process crashed or power was lost mid-write.
///
/// # Errors
/// Returns an `Err(String)` only if the file cannot be read due to OS permissions, or if an entry
/// payload contains an unknown operation type or internally inconsistent dimension length.
pub fn replay_wal(path: &Path) -> Result<Vec<WalOp>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("failed to read WAL file {}: {}", path.display(), e)),
    };

    let mut ops = Vec::new();
    let mut offset = 0;

    while offset < bytes.len() {
        let remaining = bytes.len() - offset;
        if remaining < 4 {
            // Incomplete 4-byte entry_len header due to mid-write crash; drop trailing entry
            break;
        }

        let entry_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let remaining_payload = bytes.len() - offset;
        if remaining_payload < entry_len {
            // Trailing entry was cut off mid-write; drop incomplete trailing entry
            break;
        }

        let payload = &bytes[offset..offset + entry_len];
        offset += entry_len;

        if payload.is_empty() {
            return Err("corrupted WAL entry: zero-length payload".to_string());
        }

        let op_type = payload[0];
        match op_type {
            0 => {
                // Insert: op_type (1) + id (8) + dim (4) = 13 bytes minimum
                if payload.len() < 13 {
                    return Err(format!(
                        "corrupted WAL insert entry: payload length {} < 13 bytes",
                        payload.len()
                    ));
                }

                let id = u64::from_le_bytes(payload[1..9].try_into().unwrap());
                let dim = u32::from_le_bytes(payload[9..13].try_into().unwrap()) as usize;
                let expected_len = 13 + dim * 4;
                if payload.len() != expected_len {
                    return Err(format!(
                        "corrupted WAL insert entry: payload length {} does not match expected {}",
                        payload.len(),
                        expected_len
                    ));
                }

                let mut vector = Vec::with_capacity(dim);
                let mut coord_offset = 13;
                for _ in 0..dim {
                    let coord = f32::from_le_bytes(
                        payload[coord_offset..coord_offset + 4].try_into().unwrap(),
                    );
                    vector.push(coord);
                    coord_offset += 4;
                }

                ops.push(WalOp::Insert { id, vector });
            }
            other => {
                return Err(format!("unknown WAL op type: {}", other));
            }
        }
    }

    Ok(ops)
}

/// Create a snapshot checkpoint of the [`FlatIndex`] and clear the write-ahead log.
///
/// # Ordering Invariant & Safety
/// The snapshot must be successfully written and committed to disk before the WAL file is truncated.
/// If writing the base snapshot fails for any reason (e.g. out of disk space, I/O error),
/// the WAL is left completely untouched so uncommitted operations remain intact and recoverable.
///
/// Only after `save_flat_index` succeeds is the WAL file truncated to 0 bytes.
///
/// # Errors
/// Returns an [`std::io::Error`] if saving the snapshot or truncating the WAL fails.
pub fn checkpoint(index: &FlatIndex, index_path: &Path, wal_path: &Path) -> std::io::Result<()> {
    // 1. Save fresh base snapshot first
    save_flat_index(index, index_path)?;

    // 2. Truncate / clear WAL file to 0 bytes only AFTER snapshot succeeds
    let _ = File::create(wal_path)?;
    Ok(())
}

/// Load a [`FlatIndex`] from a base snapshot and replay any pending operations from the WAL.
///
/// This provides crash-resilient index recovery:
/// 1. The base index is restored from `index_path` via [`load_flat_index`].
/// 2. If `wal_path` exists, [`replay_wal`] decodes all complete log entries recorded since the last checkpoint.
/// 3. Each recovered operation is applied in strict chronological order to catch the index up to the latest state.
///
/// # Errors
/// Returns an `Err(String)` if the base index fails to load, if replay encounters corrupted records
/// that cannot be explained by mid-write truncation, or if an insert conflicts with the index.
pub fn load_with_wal_recovery(index_path: &Path, wal_path: &Path) -> Result<FlatIndex, String> {
    let mut index = load_flat_index(index_path)?;
    let ops = replay_wal(wal_path)?;

    for op in ops {
        match op {
            WalOp::Insert { id, vector } => {
                if vector.len() != index.dim() {
                    return Err(format!(
                        "WAL replay error: vector dim {} does not match index dim {}",
                        vector.len(),
                        index.dim()
                    ));
                }
                if index.ids.contains(&id) {
                    return Err(format!(
                        "WAL replay error: duplicate id {} already in index",
                        id
                    ));
                }
                index.add(id, &vector);
            }
        }
    }

    Ok(index)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flat_index::Metric;
    use rand::Rng;

    /// Helper to generate a unique temporary file path for isolated testing.
    fn temp_file_path(prefix: &str, ext: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let rand_val: u64 = rand::thread_rng().gen();
        path.push(format!("{}_{}_{}.{}", prefix, timestamp, rand_val, ext));
        path
    }

    /// Test 1: Append multiple entries and replay in exact order.
    #[test]
    fn test_wal_append_and_replay_order() {
        let wal_path = temp_file_path("test_wal_append", "wal");
        let mut writer = open_wal_writer(&wal_path).unwrap();

        let entries = vec![
            (101u64, vec![1.0f32, 2.0, 3.0]),
            (102u64, vec![4.0f32, 5.0, 6.0]),
            (103u64, vec![7.0f32, 8.0, 9.0]),
            (104u64, vec![10.0f32, 11.0, 12.0]),
        ];

        for (id, vec) in &entries {
            writer.log_insert(*id, vec).unwrap();
        }
        drop(writer);

        let ops = replay_wal(&wal_path).unwrap();
        let _ = std::fs::remove_file(&wal_path);

        assert_eq!(ops.len(), entries.len());
        for (i, op) in ops.iter().enumerate() {
            match op {
                WalOp::Insert { id, vector } => {
                    assert_eq!(*id, entries[i].0);
                    assert_eq!(vector, &entries[i].1);
                }
            }
        }
    }

    /// Test 2: Replay on nonexistent path returns Ok(empty).
    #[test]
    fn test_wal_replay_nonexistent_returns_empty() {
        let nonexistent_path = temp_file_path("nonexistent_wal", "wal");
        assert!(!nonexistent_path.exists());

        let ops = replay_wal(&nonexistent_path).unwrap();
        assert!(ops.is_empty());
    }

    /// Test 3: Truncated trailing entry dropped gracefully without panic/error.
    #[test]
    fn test_wal_truncated_trailing_entry_dropped_gracefully() {
        let wal_path = temp_file_path("test_wal_truncated", "wal");

        // Step 1: Write two valid entries
        let mut writer = open_wal_writer(&wal_path).unwrap();
        writer.log_insert(1, &[1.0, 2.0, 3.0]).unwrap();
        writer.log_insert(2, &[4.0, 5.0, 6.0]).unwrap();
        drop(writer);

        // Record file size with 2 valid entries
        let valid_size = std::fs::metadata(&wal_path).unwrap().len();

        // Sub-case A: Append 2 partial bytes of entry_len (less than 4 bytes for length header)
        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(&[0x19, 0x00]).unwrap(); // 2 bytes instead of 4
            file.flush().unwrap();
        }

        let ops_sub_a = replay_wal(&wal_path).unwrap();
        assert_eq!(
            ops_sub_a.len(),
            2,
            "Replay must gracefully discard incomplete entry_len prefix"
        );
        assert_eq!(
            ops_sub_a[0],
            WalOp::Insert {
                id: 1,
                vector: vec![1.0, 2.0, 3.0]
            }
        );
        assert_eq!(
            ops_sub_a[1],
            WalOp::Insert {
                id: 2,
                vector: vec![4.0, 5.0, 6.0]
            }
        );

        // Sub-case B: Restore file to valid size, then append a full entry_len (25 bytes)
        // but only 8 bytes of payload (simulating process death mid-payload write)
        let file = OpenOptions::new().write(true).open(&wal_path).unwrap();
        file.set_len(valid_size).unwrap();
        drop(file);

        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            let payload_len: u32 = 1 + 8 + 4 + (3 * 4); // 25 bytes
            file.write_all(&payload_len.to_le_bytes()).unwrap();
            file.write_all(&[0u8]).unwrap(); // op_type: Insert
            file.write_all(&3u64.to_le_bytes()).unwrap(); // id: 3
                                                          // Cut off here before dim and vector floats! (8 bytes of payload instead of 25)
            file.flush().unwrap();
        }

        let ops_sub_b = replay_wal(&wal_path).unwrap();
        let _ = std::fs::remove_file(&wal_path);

        assert_eq!(
            ops_sub_b.len(),
            2,
            "Replay must gracefully discard truncated trailing payload without panic/error"
        );
        assert_eq!(
            ops_sub_b[0],
            WalOp::Insert {
                id: 1,
                vector: vec![1.0, 2.0, 3.0]
            }
        );
        assert_eq!(
            ops_sub_b[1],
            WalOp::Insert {
                id: 2,
                vector: vec![4.0, 5.0, 6.0]
            }
        );
    }

    /// Test 4: Full crash recovery simulation:
    /// Save snapshot -> log inserts to WAL -> simulate crash (drop memory) ->
    /// load_with_wal_recovery -> exact data & search match.
    #[test]
    fn test_wal_crash_recovery_simulation() {
        let index_path = temp_file_path("test_crash_idx", "vcta");
        let wal_path = temp_file_path("test_crash_wal", "wal");
        let dim = 4;

        // 1. Create base index with 3 vectors and persist snapshot
        let mut base_index = FlatIndex::new(dim, Metric::Euclidean);
        base_index.add(1, &[1.0, 0.0, 0.0, 0.0]);
        base_index.add(2, &[0.0, 1.0, 0.0, 0.0]);
        base_index.add(3, &[0.0, 0.0, 1.0, 0.0]);
        save_flat_index(&base_index, &index_path).unwrap();

        // 2. Log 2 additional live updates to the WAL
        let mut wal_writer = open_wal_writer(&wal_path).unwrap();
        wal_writer.log_insert(4, &[0.0, 0.0, 0.0, 1.0]).unwrap();
        wal_writer.log_insert(5, &[0.5, 0.5, 0.5, 0.5]).unwrap();
        drop(wal_writer);

        // 3. Simulate process crash by discarding `base_index` without saving
        drop(base_index);

        // 4. Recover from disk snapshot + WAL
        let recovered = load_with_wal_recovery(&index_path, &wal_path).unwrap();

        // 5. Build an oracle index with all 5 vectors for ground-truth comparison
        let mut oracle = FlatIndex::new(dim, Metric::Euclidean);
        oracle.add(1, &[1.0, 0.0, 0.0, 0.0]);
        oracle.add(2, &[0.0, 1.0, 0.0, 0.0]);
        oracle.add(3, &[0.0, 0.0, 1.0, 0.0]);
        oracle.add(4, &[0.0, 0.0, 0.0, 1.0]);
        oracle.add(5, &[0.5, 0.5, 0.5, 0.5]);

        // Clean up test files
        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_file(&wal_path);

        assert_eq!(recovered.len(), 5);
        assert_eq!(recovered.ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(recovered.batch.data, oracle.batch.data);

        let query = [0.1, 0.2, 0.8, 0.0];
        let oracle_res = oracle.search(&query, 3);
        let recov_res = recovered.search(&query, 3);

        println!("\nPhase 28 Confirmation: Crash Recovery Search Consistency:");
        for i in 0..oracle_res.len() {
            println!(
                "  Rank #{}: Oracle=(id={}, score={:.6}) | Recovered=(id={}, score={:.6})",
                i + 1,
                oracle_res[i].id,
                oracle_res[i].score,
                recov_res[i].id,
                recov_res[i].score
            );
        }

        assert_eq!(oracle_res.len(), recov_res.len());
        for i in 0..oracle_res.len() {
            assert_eq!(oracle_res[i].id, recov_res[i].id);
            assert_eq!(oracle_res[i].score, recov_res[i].score);
        }
    }

    /// Test 5: Checkpoint test:
    /// Snapshot updated, WAL cleared to 0 bytes, load_with_wal_recovery matches snapshot.
    #[test]
    fn test_wal_checkpoint() {
        let index_path = temp_file_path("test_ckpt_idx", "vcta");
        let wal_path = temp_file_path("test_ckpt_wal", "wal");
        let dim = 3;

        // Base index with 2 vectors
        let mut index = FlatIndex::new(dim, Metric::Cosine);
        index.add(10, &[1.0, 0.0, 0.0]);
        index.add(20, &[0.0, 1.0, 0.0]);
        save_flat_index(&index, &index_path).unwrap();

        // WAL with 2 inserts
        let mut wal_writer = open_wal_writer(&wal_path).unwrap();
        wal_writer.log_insert(30, &[0.0, 0.0, 1.0]).unwrap();
        wal_writer.log_insert(40, &[0.707, 0.707, 0.0]).unwrap();
        drop(wal_writer);

        // Apply inserts to in-memory index
        index.add(30, &[0.0, 0.0, 1.0]);
        index.add(40, &[0.707, 0.707, 0.0]);

        // Checkpoint: save snapshot, clear WAL
        checkpoint(&index, &index_path, &wal_path).unwrap();

        // Verify WAL file exists and is 0 bytes
        let wal_len = std::fs::metadata(&wal_path).unwrap().len();
        assert_eq!(wal_len, 0, "Checkpoint must truncate WAL file to 0 bytes");

        // Load with recovery: snapshot has all 4 vectors, empty WAL contributes 0
        let loaded = load_with_wal_recovery(&index_path, &wal_path).unwrap();

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_file(&wal_path);

        assert_eq!(loaded.len(), 4);
        assert_eq!(loaded.ids, vec![10, 20, 30, 40]);
        assert_eq!(loaded.batch.data, index.batch.data);
    }

    /// Test 6: Full lifecycle test:
    /// checkpoint -> inserts -> crash -> recover -> checkpoint again.
    #[test]
    fn test_wal_full_lifecycle() {
        let index_path = temp_file_path("test_lifecycle_idx", "vcta");
        let wal_path = temp_file_path("test_lifecycle_wal", "wal");
        let dim = 2;

        // Step 1: Initial state (id 1)
        let mut index = FlatIndex::new(dim, Metric::Euclidean);
        index.add(1, &[1.0, 1.0]);
        checkpoint(&index, &index_path, &wal_path).unwrap();

        // Step 2: Live inserts logged to WAL (ids 2, 3)
        let mut writer = open_wal_writer(&wal_path).unwrap();
        writer.log_insert(2, &[2.0, 2.0]).unwrap();
        writer.log_insert(3, &[3.0, 3.0]).unwrap();
        drop(writer);

        // In-memory index also receives inserts
        index.add(2, &[2.0, 2.0]);
        index.add(3, &[3.0, 3.0]);

        // Step 3: Crash simulation: drop in-memory index
        drop(index);

        // Step 4: Recover from snapshot + WAL
        let mut recovered_index = load_with_wal_recovery(&index_path, &wal_path).unwrap();
        assert_eq!(recovered_index.len(), 3);
        assert_eq!(recovered_index.ids, vec![1, 2, 3]);

        // Step 5: Checkpoint the recovered state: snapshot now has ids 1, 2, 3; WAL reset to 0
        checkpoint(&recovered_index, &index_path, &wal_path).unwrap();
        assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), 0);

        // Step 6: Next cycle of live inserts (ids 4, 5)
        let mut writer = open_wal_writer(&wal_path).unwrap();
        writer.log_insert(4, &[4.0, 4.0]).unwrap();
        writer.log_insert(5, &[5.0, 5.0]).unwrap();
        drop(writer);

        recovered_index.add(4, &[4.0, 4.0]);
        recovered_index.add(5, &[5.0, 5.0]);

        // Step 7: Another crash simulation: drop in-memory index
        drop(recovered_index);

        // Step 8: Recover again
        let final_index = load_with_wal_recovery(&index_path, &wal_path).unwrap();
        assert_eq!(final_index.len(), 5);
        assert_eq!(final_index.ids, vec![1, 2, 3, 4, 5]);

        // Step 9: Final checkpoint
        checkpoint(&final_index, &index_path, &wal_path).unwrap();
        assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), 0);

        let final_loaded = load_with_wal_recovery(&index_path, &wal_path).unwrap();
        assert_eq!(final_loaded.len(), 5);
        assert_eq!(final_loaded.ids, vec![1, 2, 3, 4, 5]);

        let _ = std::fs::remove_file(&index_path);
        let _ = std::fs::remove_file(&wal_path);
    }
}
