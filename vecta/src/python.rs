//! PyO3 bindings layer.
//!
//! This is the ONLY file in the crate allowed to import or use pyo3 types.
//! It acts as a thin bridge between the Python world and the pure-Rust core engine.

use std::path::Path;
use std::sync::Arc;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyTuple};

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::core::batch::VectorBatch;
use crate::core::concurrent_index::ConcurrentFlatIndex as CoreConcurrentFlatIndex;
use crate::core::flat_index::{FlatIndex as CoreFlatIndex, Metric};
use crate::core::hnsw::insert::insert as hnsw_insert;
use crate::core::hnsw::{HnswConfig, HnswGraph};
use crate::core::ivf_index::IVFIndex as CoreIVFIndex;
use crate::core::ivf_pq_index::IVFPQIndex as CoreIVFPQIndex;
use crate::core::kmeans::KMeansConfig;
use crate::core::metadata::{
    filtered_top_k as core_filtered_top_k, Filter, MetaValue, MetadataStore as CoreMetadataStore,
};
use crate::core::pq::PQConfig;
use crate::core::topk::ScoredId;

/// Placeholder function — confirms the extension module loads correctly.
#[pyfunction]
fn hello_vecta() -> String {
    "vecta engine initialized".to_string()
}

/// Helper to parse user-facing string metrics into internal [`Metric`] enum.
fn parse_metric(metric: &str) -> PyResult<Metric> {
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(Metric::Euclidean),
        "cosine" | "cos" => Ok(Metric::Cosine),
        "dot_product" | "dot" | "ip" => Ok(Metric::DotProduct),
        _ => Err(PyValueError::new_err(format!(
            "unknown metric '{}': expected 'euclidean', 'cosine', or 'dot_product'",
            metric
        ))),
    }
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

        let rust_metric = parse_metric(metric)?;

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

    /// Save the index to disk in the binary vecta format.
    ///
    /// # Errors
    /// Raises [`PyIOError`] if writing or file creation fails.
    pub fn save(&self, path: String) -> PyResult<()> {
        crate::core::serialize::save_flat_index(&self.inner, Path::new(&path)).map_err(|e| {
            PyIOError::new_err(format!("failed to save FlatIndex to '{}': {}", path, e))
        })
    }
}

// ============================================================================
// Phase 32: ConcurrentFlatIndex with GIL Release
// ============================================================================

/// Thread-safe concurrent flat index for multi-threaded Python applications.
///
/// Wraps an internal [`Arc<CoreConcurrentFlatIndex>`] and releases Python's Global Interpreter Lock (GIL)
/// during search, insert, and batch operations via [`Python::allow_threads`]. This allows true multi-core
/// parallelism when queried from multiple Python `threading.Thread` instances.
#[pyclass]
pub struct ConcurrentFlatIndex {
    inner: Arc<CoreConcurrentFlatIndex>,
}

#[pymethods]
impl ConcurrentFlatIndex {
    /// Create a new thread-safe concurrent flat index.
    #[new]
    pub fn new(dim: usize, metric: &str) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dimension must be greater than 0"));
        }
        let rust_metric = parse_metric(metric)?;
        Ok(Self {
            inner: Arc::new(CoreConcurrentFlatIndex::new(dim, rust_metric)),
        })
    }

    /// Search for the top-`k` nearest neighbors to `query` concurrently.
    ///
    /// Releases the GIL via `py.detach` during the search execution, allowing
    /// other Python threads to execute concurrently.
    pub fn search(&self, py: Python<'_>, query: Vec<f32>, k: usize) -> PyResult<Vec<(u64, f32)>> {
        if query.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                query.len()
            )));
        }

        let scored = py.detach(|| self.inner.search(&query, k));
        Ok(scored.into_iter().map(|s| (s.id, s.score)).collect())
    }

    /// Add a single vector with an external ID under an exclusive write lock.
    ///
    /// Releases the GIL via `py.detach` while waiting for and holding the write lock.
    pub fn add(&self, py: Python<'_>, id: u64, vector: Vec<f32>) -> PyResult<()> {
        if vector.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "vector dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                vector.len()
            )));
        }

        py.detach(|| self.inner.add(id, &vector))
            .map_err(PyValueError::new_err)
    }

    /// Bulk-add vectors with external IDs under an exclusive write lock.
    ///
    /// Validates count and vector dimensions before entering `detach`.
    pub fn add_batch(&self, py: Python<'_>, ids: Vec<u64>, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        if ids.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "ids count ({}) != vectors count ({})",
                ids.len(),
                vectors.len()
            )));
        }

        let dim = self.inner.dim();
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

        let mut batch = VectorBatch::new(dim);
        for v in &vectors {
            batch.push(v);
        }

        py.detach(|| self.inner.add_batch(&ids, &batch))
            .map_err(PyValueError::new_err)
    }

    /// Return the number of vectors stored in the index (`len(index)`).
    ///
    /// Releases the GIL across the read lock. For fast operations the GIL-release overhead
    /// can sometimes exceed the wait time, but it is applied consistently here to ensure
    /// readers never block Python-level threads during write contention.
    pub fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.len())
    }

    /// Alias for `__len__`.
    pub fn len(&self, py: Python<'_>) -> usize {
        self.__len__(py)
    }

    /// Return `true` if the index contains no vectors.
    pub fn is_empty(&self, py: Python<'_>) -> bool {
        self.__len__(py) == 0
    }

    /// Return vector dimensionality.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }
}

/// Python wrapper around the pure-Rust [`CoreIVFIndex`].
#[pyclass]
pub struct IVFIndex {
    inner: CoreIVFIndex,
}

#[pymethods]
impl IVFIndex {
    /// Create a new IVFIndex with the given dimension, cluster count, and metric.
    #[new]
    pub fn new(dim: usize, num_clusters: usize, metric: &str) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dimension must be greater than 0"));
        }
        if num_clusters == 0 {
            return Err(PyValueError::new_err("num_clusters must be greater than 0"));
        }

        let rust_metric = parse_metric(metric)?;

        Ok(Self {
            inner: CoreIVFIndex::new(dim, num_clusters, rust_metric),
        })
    }

    /// Train coarse quantizer centroids on training vectors via k-means.
    #[pyo3(signature = (training_data, k, max_iterations=100, seed=42, tolerance=1e-4))]
    pub fn train(
        &mut self,
        training_data: Vec<Vec<f32>>,
        k: usize,
        max_iterations: usize,
        seed: u64,
        tolerance: f32,
    ) -> PyResult<()> {
        if training_data.is_empty() {
            return Err(PyValueError::new_err("training_data cannot be empty"));
        }
        if k != self.inner.num_clusters() {
            return Err(PyValueError::new_err(format!(
                "k ({}) must equal index num_clusters ({})",
                k,
                self.inner.num_clusters()
            )));
        }

        let dim = self.inner.dim();
        for (i, v) in training_data.iter().enumerate() {
            if v.len() != dim {
                return Err(PyValueError::new_err(format!(
                    "training vector at index {} has dimension {}, expected {}",
                    i,
                    v.len(),
                    dim
                )));
            }
        }

        let mut batch = VectorBatch::new(dim);
        for v in &training_data {
            batch.push(v);
        }

        let config = KMeansConfig {
            k,
            max_iterations,
            tolerance,
        };

        self.inner.train(&batch, &config, seed);
        Ok(())
    }

    /// Add a single vector with an external ID into the index.
    pub fn add(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        if !self.inner.is_trained() {
            return Err(PyValueError::new_err(
                "IVFIndex must be trained before adding vectors",
            ));
        }
        if vector.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "vector dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                vector.len()
            )));
        }

        self.inner.add(id, &vector).map_err(PyValueError::new_err)
    }

    /// Bulk-add vectors with external IDs to the index.
    pub fn add_batch(&mut self, ids: Vec<u64>, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        if !self.inner.is_trained() {
            return Err(PyValueError::new_err(
                "IVFIndex must be trained before adding vectors",
            ));
        }
        if ids.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "ids count ({}) != vectors count ({})",
                ids.len(),
                vectors.len()
            )));
        }

        let dim = self.inner.dim();
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

        let mut batch = VectorBatch::new(dim);
        for v in &vectors {
            batch.push(v);
        }

        self.inner
            .add_batch(&ids, &batch)
            .map_err(PyValueError::new_err)
    }

    /// Search for top-`k` nearest neighbors probing `nprobe` centroids.
    pub fn search(&self, query: Vec<f32>, k: usize, nprobe: usize) -> PyResult<Vec<(u64, f32)>> {
        if !self.inner.is_trained() {
            return Err(PyValueError::new_err(
                "IVFIndex must be trained before searching",
            ));
        }
        if query.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                query.len()
            )));
        }

        let scored = self.inner.search(&query, k, nprobe);
        Ok(scored.into_iter().map(|s| (s.id, s.score)).collect())
    }

    /// Number of vectors stored in the index (`len(index)`).
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Alias for `__len__`.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` if index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return `true` if index centroids have been trained.
    pub fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }

    /// Return vector dimensionality.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return number of clusters.
    pub fn num_clusters(&self) -> usize {
        self.inner.num_clusters()
    }

    /// Return vector counts per cluster in centroid order.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.inner.cluster_sizes()
    }

    /// Return total vector count in the `nprobe` nearest clusters to `query`.
    pub fn nprobe_coverage(&self, query: Vec<f32>, nprobe: usize) -> PyResult<usize> {
        if !self.inner.is_trained() {
            return Err(PyValueError::new_err(
                "IVFIndex must be trained before querying coverage",
            ));
        }
        if query.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                query.len()
            )));
        }

        Ok(self.inner.nprobe_coverage(&query, nprobe))
    }

    /// Save the index to disk in the binary vecta format.
    ///
    /// # Errors
    /// Raises [`PyIOError`] if writing or file creation fails.
    pub fn save(&self, path: String) -> PyResult<()> {
        crate::core::serialize::save_ivf_index(&self.inner, Path::new(&path)).map_err(|e| {
            PyIOError::new_err(format!("failed to save IVFIndex to '{}': {}", path, e))
        })
    }
}

/// Python wrapper around the pure-Rust [`HnswGraph`].
#[pyclass]
pub struct HnswIndex {
    inner: HnswGraph,
    config: HnswConfig,
    rng: StdRng,
}

#[pymethods]
impl HnswIndex {
    /// Create a new HNSW index with default or custom parameters.
    #[new]
    #[pyo3(signature = (dim, metric="euclidean", m=16, ef_construction=200, ef_search=50, seed=42))]
    pub fn new(
        dim: usize,
        metric: &str,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dimension must be greater than 0"));
        }
        if m <= 1 {
            return Err(PyValueError::new_err("m must be greater than 1"));
        }
        if ef_construction == 0 {
            return Err(PyValueError::new_err(
                "ef_construction must be greater than 0",
            ));
        }

        let rust_metric = parse_metric(metric)?;
        let config = HnswConfig {
            m,
            ef_construction,
            ef_search,
        };

        Ok(Self {
            inner: HnswGraph::new(dim, rust_metric),
            config,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Add a single vector with an external ID to the HNSW graph.
    pub fn add(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        if vector.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "vector dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                vector.len()
            )));
        }

        hnsw_insert(&mut self.inner, id, &vector, &self.config, &mut self.rng)
            .map_err(PyValueError::new_err)
    }

    /// Bulk-add vectors with external IDs to the HNSW graph sequentially.
    ///
    /// Note: HNSW insertion is inherently sequential because each new node connects
    /// to nearest neighbors in the graph constructed up to that point.
    pub fn add_batch(&mut self, ids: Vec<u64>, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        if ids.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "ids count ({}) != vectors count ({})",
                ids.len(),
                vectors.len()
            )));
        }

        let dim = self.inner.dim();
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

        for (i, &id) in ids.iter().enumerate() {
            hnsw_insert(
                &mut self.inner,
                id,
                &vectors[i],
                &self.config,
                &mut self.rng,
            )
            .map_err(PyValueError::new_err)?;
        }

        Ok(())
    }

    /// Search for top-`k` nearest neighbors with optional `ef_search` override.
    #[pyo3(signature = (query, k, ef_search=None))]
    pub fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        ef_search: Option<usize>,
    ) -> PyResult<Vec<(u64, f32)>> {
        if query.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                query.len()
            )));
        }

        let ef = ef_search.unwrap_or(self.config.ef_search);
        let scored = self.inner.search(&query, k, ef);
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

    /// Return `true` if index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return vector dimensionality.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return a dictionary of node counts by maximum layer (`{layer: count}`).
    pub fn max_layer_distribution(&self) -> std::collections::HashMap<usize, usize> {
        let mut distribution = std::collections::HashMap::new();
        for node in &self.inner.nodes {
            *distribution.entry(node.max_layer).or_insert(0) += 1;
        }
        distribution
    }

    /// Save the index to disk in the binary vecta format.
    ///
    /// # Errors
    /// Raises [`PyIOError`] if writing or file creation fails.
    pub fn save(&self, path: String) -> PyResult<()> {
        crate::core::serialize::save_hnsw_index(&self.inner, &self.config, Path::new(&path))
            .map_err(|e| {
                PyIOError::new_err(format!("failed to save HnswIndex to '{}': {}", path, e))
            })
    }
}

/// Python wrapper around [`CoreIVFPQIndex`].
///
/// Inverted File with Product Quantization (IndexIVFPQ).
///
/// Combines coarse Voronoi cell partitioning with Product Quantization compression:
/// - Coarse quantizer: Centroids stored at full precision.
/// - Inverted lists: Vectors stored as compact PQ codes.
/// - Fine search: Evaluated using fast Asymmetric Distance Computation (ADC) lookup tables.
///
/// Note:
///     IVFPQIndex operates exclusively with Euclidean distance for ADC fine search.
///     There is no `metric` parameter for this index type in v1.
#[pyclass]
pub struct IVFPQIndex {
    inner: CoreIVFPQIndex,
}

#[pymethods]
impl IVFPQIndex {
    /// Create a new IVFPQIndex.
    ///
    /// Args:
    ///     dim: Dimensionality of vectors. Must be evenly divisible by `m`.
    ///     num_clusters: Number of coarse Voronoi clusters (inverted lists).
    ///     m: Number of subquantizers (subvectors).
    ///     k_per_subvector: Centroids per subquantizer codebook (default 256 for 1 byte/subvector).
    ///     max_iterations: Maximum Lloyd k-means iterations for PQ codebook training (default 100).
    ///
    /// Note:
    ///     IVFPQIndex operates exclusively with Euclidean distance for ADC fine search.
    ///     There is no `metric` parameter for this index type in v1.
    #[new]
    #[pyo3(signature = (dim, num_clusters, m, k_per_subvector=256, max_iterations=100))]
    pub fn new(
        dim: usize,
        num_clusters: usize,
        m: usize,
        k_per_subvector: usize,
        max_iterations: usize,
    ) -> PyResult<Self> {
        let pq_config = PQConfig {
            m,
            k_per_subvector,
            max_iterations,
        };

        let inner =
            CoreIVFPQIndex::new(dim, num_clusters, pq_config).map_err(PyValueError::new_err)?;

        Ok(Self { inner })
    }

    /// Train both coarse cluster centroids and PQ codebooks.
    ///
    /// Args:
    ///     training_data: List of vectors to train coarse and PQ quantizers.
    ///     ivf_seed: Random seed for coarse k-means clustering (default: 42).
    ///     pq_seed: Random seed for subquantizer k-means clustering (default: 42).
    #[pyo3(signature = (training_data, ivf_seed=42, pq_seed=42))]
    pub fn train(
        &mut self,
        training_data: Vec<Vec<f32>>,
        ivf_seed: u64,
        pq_seed: u64,
    ) -> PyResult<()> {
        if training_data.is_empty() {
            return Err(PyValueError::new_err("training_data cannot be empty"));
        }

        let dim = self.inner.dim();
        for (i, v) in training_data.iter().enumerate() {
            if v.len() != dim {
                return Err(PyValueError::new_err(format!(
                    "training vector at index {} has dimension {}, expected {}",
                    i,
                    v.len(),
                    dim
                )));
            }
        }

        if training_data.len() < self.inner.num_clusters() {
            return Err(PyValueError::new_err(format!(
                "insufficient training vectors ({}) for num_clusters={}",
                training_data.len(),
                self.inner.num_clusters()
            )));
        }

        let mut batch = VectorBatch::new(dim);
        for v in &training_data {
            batch.push(v);
        }

        let km_config = KMeansConfig {
            k: self.inner.num_clusters(),
            max_iterations: 100,
            tolerance: 1e-4,
        };

        self.inner
            .train(&batch, &km_config, ivf_seed, pq_seed)
            .map_err(PyValueError::new_err)
    }

    /// Add a single vector with an external ID into the index.
    pub fn add(&mut self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        if !self.inner.is_trained() {
            return Err(PyValueError::new_err(
                "IVFPQIndex must be trained before adding vectors",
            ));
        }
        if vector.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "vector dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                vector.len()
            )));
        }

        self.inner.add(id, &vector).map_err(PyValueError::new_err)
    }

    /// Bulk-add vectors with external IDs into the index.
    pub fn add_batch(&mut self, ids: Vec<u64>, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        if !self.inner.is_trained() {
            return Err(PyValueError::new_err(
                "IVFPQIndex must be trained before adding vectors",
            ));
        }
        if ids.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "ids count ({}) != vectors count ({})",
                ids.len(),
                vectors.len()
            )));
        }

        let dim = self.inner.dim();
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

        let mut batch = VectorBatch::new(dim);
        for v in &vectors {
            batch.push(v);
        }

        self.inner
            .add_batch(&ids, &batch)
            .map_err(PyValueError::new_err)
    }

    /// Search for top-`k` nearest neighbors probing `nprobe` centroids using ADC.
    ///
    /// Returns:
    ///     List of (id, score) tuples, where score is the approximate squared Euclidean distance.
    #[pyo3(signature = (query, k, nprobe=1))]
    pub fn search(&self, query: Vec<f32>, k: usize, nprobe: usize) -> PyResult<Vec<(u64, f32)>> {
        if !self.inner.is_trained() {
            return Err(PyValueError::new_err(
                "IVFPQIndex must be trained before searching",
            ));
        }
        if query.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                query.len()
            )));
        }

        if k == 0 || self.inner.is_empty() {
            return Ok(Vec::new());
        }

        let scored = self
            .inner
            .search(&query, k, nprobe)
            .map_err(PyValueError::new_err)?;

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

    /// Return `true` if index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return `true` if index centroids and PQ codebooks have been trained.
    pub fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }

    /// Return vector dimensionality.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return number of coarse clusters.
    pub fn num_clusters(&self) -> usize {
        self.inner.num_clusters()
    }

    /// Return total memory footprint in bytes.
    pub fn memory_footprint_bytes(&self) -> usize {
        self.inner.memory_footprint_bytes()
    }

    /// Save the index to disk in the binary vecta format.
    ///
    /// # Errors
    /// Raises [`PyIOError`] if writing or file creation fails.
    pub fn save(&self, path: String) -> PyResult<()> {
        crate::core::serialize::save_ivf_pq_index(&self.inner, Path::new(&path)).map_err(|e| {
            PyIOError::new_err(format!("failed to save IVFPQIndex to '{}': {}", path, e))
        })
    }
}

/// Load any saved vecta index from disk, automatically detecting its index type.
///
/// Returns a [`FlatIndex`], [`IVFIndex`], [`HnswIndex`], or [`IVFPQIndex`] instance.
///
/// # Errors
/// Raises [`PyIOError`] if the file does not exist or I/O fails.
/// Raises [`PyValueError`] if the file contains an invalid or corrupted index.
#[pyfunction]
pub fn load<'py>(py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
    let p = Path::new(&path);
    if !p.exists() {
        return Err(PyIOError::new_err(format!("file not found: '{}'", path)));
    }

    let type_code = match crate::core::serialize::peek_index_type(p) {
        Ok(code) => code,
        Err(e) => {
            if e.contains("failed to open file") || e.contains("failed to read") {
                return Err(PyIOError::new_err(e));
            } else {
                return Err(PyValueError::new_err(e));
            }
        }
    };

    match type_code {
        crate::core::serialize::INDEX_TYPE_FLAT => {
            let inner =
                crate::core::serialize::load_flat_index(p).map_err(PyValueError::new_err)?;
            let obj = Bound::new(py, FlatIndex { inner })?;
            Ok(obj.into_any())
        }
        crate::core::serialize::INDEX_TYPE_IVF => {
            let inner = crate::core::serialize::load_ivf_index(p).map_err(PyValueError::new_err)?;
            let obj = Bound::new(py, IVFIndex { inner })?;
            Ok(obj.into_any())
        }
        crate::core::serialize::INDEX_TYPE_HNSW => {
            let (inner, config) =
                crate::core::serialize::load_hnsw_index(p).map_err(PyValueError::new_err)?;
            let obj = Bound::new(
                py,
                HnswIndex {
                    inner,
                    config,
                    rng: StdRng::seed_from_u64(42),
                },
            )?;
            Ok(obj.into_any())
        }
        crate::core::serialize::INDEX_TYPE_IVF_PQ => {
            let inner =
                crate::core::serialize::load_ivf_pq_index(p).map_err(PyValueError::new_err)?;
            let obj = Bound::new(py, IVFPQIndex { inner })?;
            Ok(obj.into_any())
        }
        other => Err(PyValueError::new_err(format!(
            "unknown index type code {} in file '{}'",
            other, path
        ))),
    }
}

// ============================================================================
// Phase 30: MetadataStore and Filtered Search
// ============================================================================

/// Convert a Python value (int, float, str, bool) into an internal [`MetaValue`].
///
/// NOTE: In Python, `bool` is a subclass of `int` (`isinstance(True, int) == True`).
/// Therefore, [`PyBool`] MUST be checked before [`PyInt`] to avoid converting booleans into integers.
fn py_any_to_meta_value(val: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
    if let Ok(b) = val.cast::<PyBool>() {
        Ok(MetaValue::Bool(b.is_true()))
    } else if let Ok(i) = val.cast::<PyInt>() {
        let int_val: i64 = i.extract()?;
        Ok(MetaValue::Int(int_val))
    } else if let Ok(f) = val.cast::<PyFloat>() {
        let float_val: f64 = f.extract()?;
        Ok(MetaValue::Float(float_val))
    } else if let Ok(s) = val.cast::<PyString>() {
        let str_val: String = s.extract()?;
        Ok(MetaValue::Str(str_val))
    } else {
        Err(PyValueError::new_err(format!(
            "unsupported metadata value type '{}': expected int, float, str, or bool",
            val.get_type().name()?
        )))
    }
}

/// Recursively parse a Python filter expression into an internal [`Filter`] enum.
///
/// Mini-syntax using nested Python tuples:
/// - `("eq", field_name, value)`
/// - `("gt", field_name, value)`
/// - `("lt", field_name, value)`
/// - `("and", filter_a, filter_b)`
/// - `("or", filter_a, filter_b)`
/// - `("not", filter_a)`
pub fn parse_filter(py_filter: &Bound<'_, PyAny>) -> PyResult<Filter> {
    let tuple = py_filter.cast::<PyTuple>().map_err(|_| {
        let type_name = py_filter
            .get_type()
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        PyValueError::new_err(format!("filter must be a tuple, got '{}'", type_name))
    })?;

    if tuple.is_empty() {
        return Err(PyValueError::new_err("filter tuple cannot be empty"));
    }

    let op_any = tuple.get_item(0)?;
    let op_str: String = op_any.extract().map_err(|_| {
        PyValueError::new_err("first element of filter tuple must be an operator string (e.g. 'eq', 'gt', 'lt', 'and', 'or', 'not')")
    })?;

    match op_str.to_ascii_lowercase().as_str() {
        "eq" => {
            if tuple.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "'eq' filter expects 3 elements ('eq', field_name, value), got {}",
                    tuple.len()
                )));
            }
            let field: String = tuple
                .get_item(1)?
                .extract()
                .map_err(|_| PyValueError::new_err("field name in 'eq' filter must be a string"))?;
            let val = py_any_to_meta_value(&tuple.get_item(2)?)?;
            Ok(Filter::Eq(field, val))
        }
        "gt" => {
            if tuple.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "'gt' filter expects 3 elements ('gt', field_name, value), got {}",
                    tuple.len()
                )));
            }
            let field: String = tuple
                .get_item(1)?
                .extract()
                .map_err(|_| PyValueError::new_err("field name in 'gt' filter must be a string"))?;
            let val = py_any_to_meta_value(&tuple.get_item(2)?)?;
            Ok(Filter::Gt(field, val))
        }
        "lt" => {
            if tuple.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "'lt' filter expects 3 elements ('lt', field_name, value), got {}",
                    tuple.len()
                )));
            }
            let field: String = tuple
                .get_item(1)?
                .extract()
                .map_err(|_| PyValueError::new_err("field name in 'lt' filter must be a string"))?;
            let val = py_any_to_meta_value(&tuple.get_item(2)?)?;
            Ok(Filter::Lt(field, val))
        }
        "and" => {
            if tuple.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "'and' filter expects 3 elements ('and', filter_a, filter_b), got {}",
                    tuple.len()
                )));
            }
            let left = parse_filter(&tuple.get_item(1)?)?;
            let right = parse_filter(&tuple.get_item(2)?)?;
            Ok(Filter::And(Box::new(left), Box::new(right)))
        }
        "or" => {
            if tuple.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "'or' filter expects 3 elements ('or', filter_a, filter_b), got {}",
                    tuple.len()
                )));
            }
            let left = parse_filter(&tuple.get_item(1)?)?;
            let right = parse_filter(&tuple.get_item(2)?)?;
            Ok(Filter::Or(Box::new(left), Box::new(right)))
        }
        "not" => {
            if tuple.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "'not' filter expects 2 elements ('not', filter_expr), got {}",
                    tuple.len()
                )));
            }
            let sub = parse_filter(&tuple.get_item(1)?)?;
            Ok(Filter::Not(Box::new(sub)))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown filter operator '{}': expected 'eq', 'gt', 'lt', 'and', 'or', or 'not'",
            other
        ))),
    }
}

/// Python wrapper around the pure-Rust [`CoreMetadataStore`].
///
/// Stores arbitrary key-value metadata attributes keyed by vector external ID.
#[pyclass]
pub struct MetadataStore {
    inner: CoreMetadataStore,
}

#[pymethods]
impl MetadataStore {
    /// Create a new empty metadata store.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CoreMetadataStore::new(),
        }
    }

    /// Set a metadata attribute on a vector ID.
    ///
    /// Accepts int, float, str, or bool values.
    /// Raises [`PyValueError`] for unsupported types.
    pub fn set(&mut self, id: u64, field: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let meta_val = py_any_to_meta_value(value)?;
        self.inner.set(id, &field, meta_val);
        Ok(())
    }

    /// Retrieve a metadata attribute for a vector ID.
    ///
    /// Returns the native Python value (int, float, str, bool), or `None` if not found.
    pub fn get(&self, py: Python<'_>, id: u64, field: String) -> PyResult<Option<Py<PyAny>>> {
        match self.inner.get(id, &field) {
            Some(MetaValue::Int(i)) => Ok(Some(i.into_pyobject(py)?.into_any().unbind())),
            Some(MetaValue::Float(f)) => Ok(Some(f.into_pyobject(py)?.into_any().unbind())),
            Some(MetaValue::Str(s)) => Ok(Some(s.into_pyobject(py)?.into_any().unbind())),
            Some(MetaValue::Bool(b)) => {
                Ok(Some(PyBool::new(py, *b).to_owned().into_any().unbind()))
            }
            None => Ok(None),
        }
    }

    /// Remove all metadata attributes associated with a vector ID.
    pub fn remove(&mut self, id: u64) {
        self.inner.remove(id);
    }

    /// Return the number of unique vector IDs stored in the metadata store.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Alias for `__len__`.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return true if the metadata store contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Filter candidate search results against metadata constraints.
///
/// Implements the "over-fetch then filter" strategy. The caller executes an index
/// search with `overfetch_k` candidates and passes the `(id, score)` results here,
/// along with a [`MetadataStore`], a filter tuple expression, and target `k`.
///
/// Returns the first `k` matching candidates, preserving the index's ranking order.
#[pyfunction]
pub fn filtered_search(
    results: Vec<(u64, f32)>,
    store: &MetadataStore,
    filter: &Bound<'_, PyAny>,
    k: usize,
) -> PyResult<Vec<(u64, f32)>> {
    let parsed_filter = parse_filter(filter)?;
    let candidates: Vec<ScoredId> = results
        .into_iter()
        .map(|(id, score)| ScoredId { id, score })
        .collect();
    let survivors = core_filtered_top_k(candidates, &store.inner, &parsed_filter, k);
    Ok(survivors.into_iter().map(|s| (s.id, s.score)).collect())
}

/// Register all Python-exposed functions and classes onto the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_vecta, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(filtered_search, m)?)?;
    Ok(())
}
