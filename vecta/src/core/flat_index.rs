// Brute-force flat index — the ground-truth oracle for vecta.
//
// Every future ANN algorithm (IVF, HNSW, PQ) gets its recall@k validated
// against this index. It stays in the codebase permanently, mirroring
// FAISS's IndexFlatL2/IndexFlatIP role as a correctness baseline.

use super::batch::VectorBatch;

/// Distance / similarity metric used for search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// A flat (brute-force) vector index backed by contiguous storage.
///
/// Vectors are stored in a [`VectorBatch`] (single flat `Vec<f32>`),
/// with a parallel `ids` vec mapping each row index to an external ID.
pub struct FlatIndex {
    /// Contiguous vector storage.
    pub batch: VectorBatch,
    /// External ID for each row: `ids[i]` corresponds to `batch.get(i)`.
    pub ids: Vec<u64>,
    /// Distance metric used for search queries.
    pub metric: Metric,
}

impl FlatIndex {
    /// Create an empty index for vectors of the given dimensionality.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            batch: VectorBatch::new(dim),
            ids: Vec::new(),
            metric,
        }
    }

    /// Insert a single vector with its external ID.
    ///
    /// # Panics
    /// - If `vector.len() != self.dim()`.
    /// - If `id` already exists in the index.
    pub fn add(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.batch.dim,
            "FlatIndex::add: expected dim {}, got {}",
            self.batch.dim,
            vector.len()
        );
        assert!(
            !self.ids.contains(&id),
            "FlatIndex::add: duplicate id {}",
            id
        );
        self.batch.push(vector);
        self.ids.push(id);
    }

    /// Bulk-insert vectors with their external IDs.
    ///
    /// Performs ONE upfront duplicate-ID check across both the incoming batch
    /// and the existing index, rather than N individual checks per vector.
    ///
    /// # Panics
    /// - If `ids.len() != vectors.num_vectors`.
    /// - If `vectors.dim != self.batch.dim`.
    /// - If any ID in `ids` already exists in the index.
    /// - If `ids` contains duplicates within itself.
    pub fn add_batch(&mut self, ids: &[u64], vectors: &VectorBatch) {
        assert_eq!(
            ids.len(),
            vectors.num_vectors,
            "FlatIndex::add_batch: ids count ({}) != vectors count ({})",
            ids.len(),
            vectors.num_vectors
        );
        assert_eq!(
            vectors.dim, self.batch.dim,
            "FlatIndex::add_batch: incoming dim {} != index dim {}",
            vectors.dim, self.batch.dim
        );

        // ONE bulk duplicate check: incoming IDs vs existing index + within themselves.
        for (i, &new_id) in ids.iter().enumerate() {
            assert!(
                !self.ids.contains(&new_id),
                "FlatIndex::add_batch: duplicate id {} (already in index)",
                new_id
            );
            // Also check for duplicates within the incoming batch itself.
            assert!(
                !ids[..i].contains(&new_id),
                "FlatIndex::add_batch: duplicate id {} (within incoming batch)",
                new_id
            );
        }

        // All checks passed — append data in bulk.
        self.batch.data.extend_from_slice(&vectors.data);
        self.batch.num_vectors += vectors.num_vectors;
        self.ids.extend_from_slice(ids);
    }

    /// Number of vectors currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.batch.num_vectors
    }

    /// Returns `true` if the index contains no vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.batch.num_vectors == 0
    }

    /// Dimensionality of vectors in this index.
    #[inline]
    pub fn dim(&self) -> usize {
        self.batch.dim
    }

    /// Look up a vector by its external ID.
    ///
    /// Linear scan through `self.ids` — acceptable for correctness testing,
    /// not intended as a high-performance lookup path.
    ///
    /// Returns `None` if the ID is not found.
    pub fn get_vector(&self, id: u64) -> Option<&[f32]> {
        self.ids
            .iter()
            .position(|&stored_id| stored_id == id)
            .map(|idx| self.batch.get(idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let index = FlatIndex::new(3, Metric::Euclidean);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_three_vectors() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(1, &[1.0, 2.0, 3.0]);
        index.add(2, &[4.0, 5.0, 6.0]);
        index.add(3, &[7.0, 8.0, 9.0]);

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    #[should_panic(expected = "expected dim 3, got 2")]
    fn test_add_wrong_dimension_panics() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(1, &[1.0, 2.0]); // dim 2 into dim-3 index
    }

    #[test]
    #[should_panic(expected = "duplicate id 42")]
    fn test_add_duplicate_id_panics() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(42, &[1.0, 2.0, 3.0]);
        index.add(42, &[4.0, 5.0, 6.0]); // duplicate
    }

    #[test]
    fn test_add_batch_inserts_correctly() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(0, &[0.0, 0.0, 0.0]); // pre-existing

        let mut incoming = VectorBatch::new(3);
        incoming.push(&[1.0, 2.0, 3.0]);
        incoming.push(&[4.0, 5.0, 6.0]);
        incoming.push(&[7.0, 8.0, 9.0]);

        index.add_batch(&[10, 20, 30], &incoming);
        assert_eq!(index.len(), 4); // 1 + 3
    }

    #[test]
    fn test_get_vector_found() {
        let mut index = FlatIndex::new(3, Metric::Euclidean);
        index.add(10, &[1.0, 2.0, 3.0]);
        index.add(20, &[4.0, 5.0, 6.0]);
        index.add(30, &[7.0, 8.0, 9.0]);

        let v = index.get_vector(20).expect("id 20 should exist");
        assert_eq!(v, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_get_vector_not_found() {
        let index = FlatIndex::new(3, Metric::Euclidean);
        assert!(index.get_vector(999).is_none());
    }

    #[test]
    fn test_add_batch_1000_vectors() {
        let dim = 16;
        let n = 1000;

        let mut batch = VectorBatch::new(dim);
        let ids: Vec<u64> = (0..n as u64).collect();

        // Deterministic fill.
        let mut rng: u64 = 7;
        for _ in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng >> 33) as f32) / (u32::MAX as f32)
                })
                .collect();
            batch.push(&v);
        }

        let mut index = FlatIndex::new(dim, Metric::Euclidean);
        index.add_batch(&ids, &batch);
        assert_eq!(index.len(), n);
    }
}
