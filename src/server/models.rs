//! Request and response JSON data transfer models for the Vecta REST API.

use serde::{Deserialize, Serialize};

/// Request payload for creating a new vector collection.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateCollectionRequest {
    /// Unique name identifying the collection.
    pub name: String,
    /// Dimensionality of vectors stored in this collection.
    pub dim: usize,
    /// Vector indexing architecture: `"flat"`, `"ivf"`, `"hnsw"`, or `"ivfpq"`.
    pub index_type: String,
    /// Distance metric: `"euclidean"` (or `"l2"`), `"cosine"` (or `"cos"`), `"dot_product"` (or `"dot"`).
    pub metric: String,
}

/// Request payload for inserting a single vector with an external ID.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InsertPointRequest {
    /// Unique external identifier (e.g. primary key, document ID).
    pub id: u64,
    /// Floating-point coordinate array matching the collection's dimensionality.
    pub vector: Vec<f32>,
}

/// Request payload for executing a k-nearest-neighbor search query.
///
/// # Parameter Scope
/// - `nprobe`: Only relevant for `"ivf"` and `"ivfpq"` collections; ignored otherwise.
/// - `ef_search`: Only relevant for `"hnsw"` collections; ignored otherwise.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchRequest {
    /// Query vector coordinates matching the collection's dimensionality.
    pub vector: Vec<f32>,
    /// Number of nearest neighbors to retrieve.
    pub k: usize,
    /// Number of Voronoi clusters to probe (IVF/IVF-PQ only).
    pub nprobe: Option<usize>,
    /// Size of dynamic candidate list during graph traversal (HNSW only).
    pub ef_search: Option<usize>,
}

/// A single matched search candidate result.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SearchResultItem {
    /// External identifier of the matched vector.
    pub id: u64,
    /// Distance or similarity score according to the collection metric.
    pub score: f32,
}

/// Response payload containing top-k search results.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchResponse {
    /// List of search candidate items sorted best-first.
    pub results: Vec<SearchResultItem>,
}

/// Basic summary metadata describing a collection.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CollectionInfo {
    /// Unique collection name.
    pub name: String,
    /// Index architecture type (`"flat"`, `"ivf"`, `"hnsw"`, `"ivfpq"`).
    pub index_type: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric name.
    pub metric: String,
    /// Current number of indexed vectors.
    pub vector_count: usize,
}

/// Generic error response returned on 4xx/5xx HTTP failures.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    /// Descriptive error message explaining the failure reason.
    pub error: String,
}

/// Health check response confirming server process liveness.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HealthResponse {
    /// Health status string (e.g. `"ok"`).
    pub status: String,
}
