//! PyO3 bindings layer.
//!
//! This is the ONLY file in the crate allowed to import or use pyo3 types.
//! It acts as a thin bridge between the Python world and the pure-Rust core engine.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::core::batch::VectorBatch;
use crate::core::flat_index::{FlatIndex as CoreFlatIndex, Metric};
use crate::core::hnsw::insert::insert as hnsw_insert;
use crate::core::hnsw::{HnswConfig, HnswGraph};
use crate::core::ivf_index::IVFIndex as CoreIVFIndex;
use crate::core::ivf_pq_index::IVFPQIndex as CoreIVFPQIndex;
use crate::core::kmeans::KMeansConfig;
use crate::core::pq::PQConfig;

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
}

/// Register all Python-exposed functions and classes onto the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_vecta, m)?)?;
    Ok(())
}
