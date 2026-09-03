//! PyO3 bindings layer.
//!
//! This is the ONLY file in the crate allowed to import or use pyo3 types.
//! It acts as a thin bridge between the Python world and the pure-Rust core engine.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::batch::VectorBatch;
use crate::core::flat_index::{FlatIndex as CoreFlatIndex, Metric};

/// Placeholder function — confirms the extension module loads correctly.
#[pyfunction]
fn hello_vecta() -> String {
    "vecta engine initialized".to_string()
}

/// Python wrapper around the pure-Rust [`CoreFlatIndex`].
///
/// Keeps core structs untouched and provides safe, panic-free Python bindings.
#[pyclass]
pub struct FlatIndex {
    inner: CoreFlatIndex,
}

#[pymethods]
impl FlatIndex {
    /// Create a new FlatIndex with the given dimension and metric.
    ///
    /// Supported metrics:
    /// - `"euclidean"` or `"l2"`
    /// - `"cosine"` or `"cos"`
    /// - `"dot_product"`, `"dot"`, or `"ip"`
    #[new]
    pub fn new(dim: usize, metric: &str) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dimension must be greater than 0"));
        }

        let rust_metric = match metric.to_ascii_lowercase().as_str() {
            "euclidean" | "l2" => Metric::Euclidean,
            "cosine" | "cos" => Metric::Cosine,
            "dot_product" | "dot" | "ip" => Metric::DotProduct,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown metric '{}': expected 'euclidean', 'cosine', or 'dot_product'",
                    metric
                )))
            }
        };

        Ok(Self {
            inner: CoreFlatIndex::new(dim, rust_metric),
        })
    }

    /// Add a single vector with an external ID to the index.
    ///
    /// Raises [`PyValueError`] on dimension mismatch or duplicate ID.
    pub fn add(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        if vector.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "vector dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                vector.len()
            )));
        }

        if self.inner.ids.contains(&id) {
            return Err(PyValueError::new_err(format!(
                "duplicate id {}: already exists in index",
                id
            )));
        }

        self.inner.add(id, &vector);
        Ok(())
    }

    /// Bulk-add vectors with external IDs to the index.
    ///
    /// Raises [`PyValueError`] on length mismatch, dimension mismatch, or duplicate ID.
    pub fn add_batch(&mut self, ids: Vec<u64>, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        if ids.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "ids count ({}) != vectors count ({})",
                ids.len(),
                vectors.len()
            )));
        }

        let dim = self.inner.dim();

        // 1. Check all vector dimensions before making any mutations
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(PyValueError::new_err(format!(
                    "vector at index {} has dimension {}, expected {}",
                    i,
                    v.len(),
                    dim
                )));
            }
        }

        // 2. Check for duplicate IDs within the incoming batch and against existing IDs
        let mut seen = std::collections::HashSet::with_capacity(ids.len());
        for &id in &ids {
            if !seen.insert(id) {
                return Err(PyValueError::new_err(format!(
                    "duplicate id {} within incoming batch",
                    id
                )));
            }
            if self.inner.ids.contains(&id) {
                return Err(PyValueError::new_err(format!(
                    "duplicate id {} (already exists in index)",
                    id
                )));
            }
        }

        // 3. Construct VectorBatch and append
        let mut batch = VectorBatch::new(dim);
        for v in &vectors {
            batch.push(v);
        }

        self.inner.add_batch(&ids, &batch);
        Ok(())
    }

    /// Search for the top-`k` nearest neighbors to `query`.
    ///
    /// Returns a list of `(id, score)` tuples.
    /// Raises [`PyValueError`] on query dimension mismatch.
    pub fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(u64, f32)>> {
        if query.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                query.len()
            )));
        }

        let scored = self.inner.search(&query, k);
        Ok(scored.into_iter().map(|s| (s.id, s.score)).collect())
    }

    /// Return the number of vectors stored in the index (`len(index)`).
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Alias for `__len__`.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return vector dimensionality.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }
}

/// Register all Python-exposed functions and classes onto the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_vecta, m)?)?;
    Ok(())
}
