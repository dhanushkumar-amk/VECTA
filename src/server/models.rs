//! Request and response JSON data transfer models for the Vecta REST API.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Request payload for creating a new vector collection.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct CreateCollectionRequest {
    /// Unique name identifying the collection.
    #[schema(example = "documents")]
    pub name: String,
    /// Dimensionality of vectors stored in this collection.
    #[schema(example = 128)]
    pub dim: usize,
    /// Vector indexing architecture: `"flat"`, `"ivf"`, `"hnsw"`, or `"ivfpq"`.
    #[schema(example = "flat")]
    pub index_type: String,
    /// Distance metric: `"euclidean"` (or `"l2"`), `"cosine"` (or `"cos"`), `"dot_product"` (or `"dot"`).
    #[schema(example = "euclidean")]
    pub metric: String,
}

/// Request payload for inserting a single vector with an external ID.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct InsertPointRequest {
    /// Unique external identifier (e.g. primary key, document ID).
    #[schema(example = 1)]
    pub id: u64,
    /// Floating-point coordinate array matching the collection's dimensionality.
    #[schema(example = json!([0.1, 0.2, 0.3]))]
    pub vector: Vec<f32>,
}

/// Request payload for executing a k-nearest-neighbor search query.
///
/// # Parameter Scope
/// - `nprobe`: Only relevant for `"ivf"` and `"ivfpq"` collections; ignored otherwise.
/// - `ef_search`: Only relevant for `"hnsw"` collections; ignored otherwise.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct SearchRequest {
    /// Query vector coordinates matching the collection's dimensionality.
    #[schema(example = json!([0.1, 0.2, 0.3]))]
    pub vector: Vec<f32>,
    /// Number of nearest neighbors to retrieve.
    #[schema(example = 5)]
    pub k: usize,
    /// Number of Voronoi clusters to probe (IVF/IVF-PQ only).
    pub nprobe: Option<usize>,
    /// Size of dynamic candidate list during graph traversal (HNSW only).
    pub ef_search: Option<usize>,
}

/// A single matched search candidate result.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, ToSchema)]
pub struct SearchResultItem {
    /// External identifier of the matched vector.
    #[schema(example = 1)]
    pub id: u64,
    /// Distance or similarity score according to the collection metric.
    #[schema(example = 0.042)]
    pub score: f32,
}

/// Response payload containing top-k search results.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct SearchResponse {
    /// List of search candidate items sorted best-first.
    pub results: Vec<SearchResultItem>,
}

/// Basic summary metadata describing a collection.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct CollectionInfo {
    /// Unique collection name.
    #[schema(example = "documents")]
    pub name: String,
    /// Index architecture type (`"flat"`, `"ivf"`, `"hnsw"`, `"ivfpq"`).
    #[schema(example = "flat")]
    pub index_type: String,
    /// Vector dimensionality.
    #[schema(example = 128)]
    pub dim: usize,
    /// Distance metric name.
    #[schema(example = "euclidean")]
    pub metric: String,
    /// Current number of indexed vectors.
    #[schema(example = 1000)]
    pub vector_count: usize,
}

/// Generic error response returned on 4xx/5xx HTTP failures.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Descriptive error message explaining the failure reason.
    #[schema(example = "collection 'documents' not found")]
    pub error: String,
}

/// Health check response confirming server process liveness.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct HealthResponse {
    /// Health status string (e.g. `"ok"`).
    #[schema(example = "ok")]
    pub status: String,
}
