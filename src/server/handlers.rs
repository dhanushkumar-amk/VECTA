//! Request handlers, error mapping, and OpenAPI annotations for the Vecta REST API.

use std::sync::Arc;

use axum::extract::{Path, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::core::flat_index::{FlatIndex, Metric};
use crate::core::hnsw::layer::HnswConfig;
use crate::core::hnsw::HnswGraph;
use crate::core::ivf_index::IVFIndex;
use crate::core::ivf_pq_index::IVFPQIndex;
use crate::core::pq::PQConfig;
use crate::server::models::{
    CollectionInfo, CreateCollectionRequest, ErrorResponse, HealthResponse, InsertPointRequest,
    SearchRequest, SearchResponse, SearchResultItem,
};
use crate::server::state::{AppState, CollectionIndex};

/// Custom application error type mapping internal domain errors into HTTP status codes
/// and uniform JSON error payloads.
#[derive(Debug)]
pub enum AppError {
    /// 401 Unauthorized: missing or invalid authentication token.
    Unauthorized(String),
    /// 404 Not Found: requested entity does not exist.
    NotFound(String),
    /// 400 Bad Request: client provided invalid inputs (dimension mismatch, bad enum, etc.).
    BadRequest(String),
    /// 409 Conflict: resource already exists (e.g. duplicate collection name).
    Conflict(String),
    /// 500 Internal Server Error: unexpected internal failure or poisoned lock.
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::Conflict(msg) => (StatusCode::CONFLICT, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

/// Axum middleware enforcing Bearer API key authentication.
///
/// If `state.api_key` is None, authentication is bypassed (local development mode).
/// If `state.api_key` is Some(key), incoming requests must provide an `Authorization: Bearer <key>`
/// header matching the expected secret, or a 401 Unauthorized response is returned.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Result<Response, AppError> {
    if let Some(ref expected_key) = state.api_key {
        let auth_header = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok());

        let authorized = match auth_header {
            Some(val) if val.starts_with("Bearer ") => {
                let token = val[7..].trim();
                token == expected_key
            }
            _ => false,
        };

        if !authorized {
            return Err(AppError::Unauthorized(
                "Unauthorized: missing or invalid Bearer API key".to_string(),
            ));
        }
    }

    Ok(next.run(req).await)
}

/// Helper to parse user-facing string metrics into internal [`Metric`] enum.
fn parse_metric(metric: &str) -> Result<Metric, AppError> {
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(Metric::Euclidean),
        "cosine" | "cos" => Ok(Metric::Cosine),
        "dot_product" | "dot" | "ip" => Ok(Metric::DotProduct),
        _ => Err(AppError::BadRequest(format!(
            "invalid metric '{}': expected 'euclidean', 'cosine', or 'dot_product'",
            metric
        ))),
    }
}

/// Helper to format internal [`Metric`] into standard string representation.
fn metric_to_str(metric: Metric) -> &'static str {
    match metric {
        Metric::Euclidean => "euclidean",
        Metric::Cosine => "cosine",
        Metric::DotProduct => "dot_product",
    }
}

/// GET /health
///
/// Liveness probe returning 200 OK. Requires no authentication or state access.
#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Service liveness status", body = HealthResponse)
    ),
    tag = "System"
)]
pub async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

/// POST /collections
///
/// Creates a new named collection.
#[utoipa::path(
    post,
    path = "/collections",
    request_body = CreateCollectionRequest,
    responses(
        (status = 201, description = "Collection successfully created", body = CollectionInfo),
        (status = 400, description = "Invalid request payload", body = ErrorResponse),
        (status = 401, description = "Unauthorized: missing or invalid API key", body = ErrorResponse),
        (status = 409, description = "Collection already exists", body = ErrorResponse)
    ),
    security(
        ("BearerAuth" = [])
    ),
    tag = "Collections"
)]
pub async fn create_collection_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<(StatusCode, Json<CollectionInfo>), AppError> {
    if req.name.trim().is_empty() {
        return Err(AppError::BadRequest(
            "collection name cannot be empty".to_string(),
        ));
    }

    if req.dim == 0 {
        return Err(AppError::BadRequest(
            "vector dimension must be greater than 0".to_string(),
        ));
    }

    let metric = parse_metric(&req.metric)?;
    let mut registry = state
        .collections
        .write()
        .map_err(|_| AppError::Internal("collection registry lock poisoned".to_string()))?;

    if registry.contains_key(&req.name) {
        return Err(AppError::Conflict(format!(
            "collection '{}' already exists",
            req.name
        )));
    }

    let index = match req.index_type.to_ascii_lowercase().as_str() {
        "flat" => CollectionIndex::Flat(FlatIndex::new(req.dim, metric)),
        "hnsw" => CollectionIndex::Hnsw(HnswGraph::new(req.dim, metric)),
        "ivf" => {
            let num_clusters = 4;
            CollectionIndex::Ivf(IVFIndex::new(req.dim, num_clusters, metric))
        }
        "ivfpq" => {
            let m = if req.dim % 2 == 0 { 2 } else { 1 };
            let pq_config = PQConfig {
                m,
                k_per_subvector: 256.min(1 << req.dim.min(8)),
                max_iterations: 100,
            };
            let num_clusters = 4;
            let ivf_pq = IVFPQIndex::new(req.dim, num_clusters, pq_config)
                .map_err(|e| AppError::BadRequest(e))?;
            CollectionIndex::IvfPq(ivf_pq)
        }
        _ => {
            return Err(AppError::BadRequest(format!(
                "invalid index_type '{}': expected 'flat', 'ivf', 'hnsw', or 'ivfpq'",
                req.index_type
            )));
        }
    };

    let info = CollectionInfo {
        name: req.name.clone(),
        index_type: index.index_type_str().to_string(),
        dim: index.dim(),
        metric: metric_to_str(index.metric()).to_string(),
        vector_count: index.len(),
    };

    // Save initial snapshot to disk
    let _ = state.save_collection_snapshot(&req.name, &index);

    registry.insert(req.name, index);

    Ok((StatusCode::CREATED, Json(info)))
}

/// GET /collections
///
/// Lists all registered collections with basic summary information.
#[utoipa::path(
    get,
    path = "/collections",
    responses(
        (status = 200, description = "List of all registered collections", body = Vec<CollectionInfo>),
        (status = 401, description = "Unauthorized: missing or invalid API key", body = ErrorResponse)
    ),
    security(
        ("BearerAuth" = [])
    ),
    tag = "Collections"
)]
pub async fn list_collections_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<CollectionInfo>>, AppError> {
    let registry = state
        .collections
        .read()
        .map_err(|_| AppError::Internal("collection registry lock poisoned".to_string()))?;

    let mut list = Vec::with_capacity(registry.len());
    for (name, index) in registry.iter() {
        list.push(CollectionInfo {
            name: name.clone(),
            index_type: index.index_type_str().to_string(),
            dim: index.dim(),
            metric: metric_to_str(index.metric()).to_string(),
            vector_count: index.len(),
        });
    }

    list.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(Json(list))
}

/// GET /collections/:name
///
/// Returns detailed metadata for a specific collection.
#[utoipa::path(
    get,
    path = "/collections/{name}",
    params(
        ("name" = String, Path, description = "Name of collection to inspect")
    ),
    responses(
        (status = 200, description = "Collection details", body = CollectionInfo),
        (status = 401, description = "Unauthorized", body = ErrorResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    ),
    security(
        ("BearerAuth" = [])
    ),
    tag = "Collections"
)]
pub async fn get_collection_handler(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfo>, AppError> {
    let registry = state
        .collections
        .read()
        .map_err(|_| AppError::Internal("collection registry lock poisoned".to_string()))?;

    let index = registry
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("collection '{}' not found", name)))?;

    Ok(Json(CollectionInfo {
        name,
        index_type: index.index_type_str().to_string(),
        dim: index.dim(),
        metric: metric_to_str(index.metric()).to_string(),
        vector_count: index.len(),
    }))
}

/// DELETE /collections/:name
///
/// Removes a collection from the registry and removes its on-disk snapshot and WAL.
#[utoipa::path(
    delete,
    path = "/collections/{name}",
    params(
        ("name" = String, Path, description = "Name of collection to delete")
    ),
    responses(
        (status = 200, description = "Collection successfully deleted"),
        (status = 401, description = "Unauthorized", body = ErrorResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    ),
    security(
        ("BearerAuth" = [])
    ),
    tag = "Collections"
)]
pub async fn delete_collection_handler(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    let mut registry = state
        .collections
        .write()
        .map_err(|_| AppError::Internal("collection registry lock poisoned".to_string()))?;

    if registry.remove(&name).is_some() {
        let _ = std::fs::remove_file(state.collection_file_path(&name));
        let _ = std::fs::remove_file(state.wal_file_path(&name));
        Ok(StatusCode::OK)
    } else {
        Err(AppError::NotFound(format!(
            "collection '{}' not found",
            name
        )))
    }
}

/// POST /collections/:name/checkpoint
///
/// Manually triggers a snapshot checkpoint to disk and truncates the WAL.
#[utoipa::path(
    post,
    path = "/collections/{name}/checkpoint",
    params(
        ("name" = String, Path, description = "Name of collection to checkpoint")
    ),
    responses(
        (status = 200, description = "Snapshot successfully saved and WAL truncated"),
        (status = 401, description = "Unauthorized", body = ErrorResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    ),
    security(
        ("BearerAuth" = [])
    ),
    tag = "Persistence"
)]
pub async fn checkpoint_handler(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    state.checkpoint_collection(&name).map_err(|e| {
        if e.contains("not found") {
            AppError::NotFound(e)
        } else {
            AppError::Internal(e)
        }
    })?;
    Ok(StatusCode::OK)
}

/// POST /collections/:name/points
///
/// Inserts a vector with its external ID into the specified collection.
#[utoipa::path(
    post,
    path = "/collections/{name}/points",
    params(
        ("name" = String, Path, description = "Target collection name")
    ),
    request_body = InsertPointRequest,
    responses(
        (status = 201, description = "Vector point successfully indexed"),
        (status = 400, description = "Dimension mismatch or duplicate ID", body = ErrorResponse),
        (status = 401, description = "Unauthorized", body = ErrorResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    ),
    security(
        ("BearerAuth" = [])
    ),
    tag = "Points"
)]
pub async fn insert_point_handler(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<InsertPointRequest>,
) -> Result<StatusCode, AppError> {
    let mut registry = state
        .collections
        .write()
        .map_err(|_| AppError::Internal("collection registry lock poisoned".to_string()))?;

    let index = registry
        .get_mut(&name)
        .ok_or_else(|| AppError::NotFound(format!("collection '{}' not found", name)))?;

    // Dimension check prior to insertion
    if req.vector.len() != index.dim() {
        return Err(AppError::BadRequest(format!(
            "vector dimension mismatch: expected {}, got {}",
            index.dim(),
            req.vector.len()
        )));
    }

    match index {
        CollectionIndex::Flat(flat) => {
            if flat.ids.contains(&req.id) {
                return Err(AppError::BadRequest(format!(
                    "duplicate id {} (already in index)",
                    req.id
                )));
            }
            flat.add(req.id, &req.vector);
            // Append to Write-Ahead Log for crash safety
            let wal_path = state.wal_file_path(&name);
            if let Ok(mut writer) = crate::core::wal::open_wal_writer(&wal_path) {
                let _ = writer.log_insert(req.id, &req.vector);
            }
            Ok(StatusCode::CREATED)
        }
        CollectionIndex::Hnsw(graph) => {
            if graph.id_to_index.contains_key(&req.id) {
                return Err(AppError::BadRequest(format!(
                    "duplicate id {} (already in index)",
                    req.id
                )));
            }
            let config = HnswConfig {
                m: 16,
                ef_construction: 200,
                ef_search: 50,
            };
            let mut rng = rand::thread_rng();
            crate::core::hnsw::insert::insert(graph, req.id, &req.vector, &config, &mut rng)
                .map_err(|e| AppError::BadRequest(e))?;
            Ok(StatusCode::CREATED)
        }
        CollectionIndex::Ivf(_) => Err(AppError::BadRequest(
            "IVF collections require training before adding points; training is not yet supported via the REST API in this phase".to_string(),
        )),
        CollectionIndex::IvfPq(_) => Err(AppError::BadRequest(
            "IVF-PQ collections require training before adding points; training is not yet supported via the REST API in this phase".to_string(),
        )),
    }
}

/// POST /collections/:name/search
///
/// Executes a k-nearest-neighbor search query on the collection.
#[utoipa::path(
    post,
    path = "/collections/{name}/search",
    params(
        ("name" = String, Path, description = "Target collection name")
    ),
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Top-k nearest neighbor results", body = SearchResponse),
        (status = 400, description = "Dimension mismatch or query error", body = ErrorResponse),
        (status = 401, description = "Unauthorized", body = ErrorResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    ),
    security(
        ("BearerAuth" = [])
    ),
    tag = "Search"
)]
pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    let registry = state
        .collections
        .read()
        .map_err(|_| AppError::Internal("collection registry lock poisoned".to_string()))?;

    let index = registry
        .get(&name)
        .ok_or_else(|| AppError::NotFound(format!("collection '{}' not found", name)))?;

    if req.vector.len() != index.dim() {
        return Err(AppError::BadRequest(format!(
            "query vector dimension mismatch: expected {}, got {}",
            index.dim(),
            req.vector.len()
        )));
    }

    let results = match index {
        CollectionIndex::Flat(flat) => {
            let scored = flat.search(&req.vector, req.k);
            scored
                .into_iter()
                .map(|s| SearchResultItem {
                    id: s.id,
                    score: s.score,
                })
                .collect()
        }
        CollectionIndex::Hnsw(graph) => {
            let ef_search = req.ef_search.unwrap_or(50);
            let scored = graph.search(&req.vector, req.k, ef_search);
            scored
                .into_iter()
                .map(|s| SearchResultItem {
                    id: s.id,
                    score: s.score,
                })
                .collect()
        }
        CollectionIndex::Ivf(ivf) => {
            if !ivf.is_trained {
                return Err(AppError::BadRequest(
                    "IVF collection is untrained and cannot be searched".to_string(),
                ));
            }
            let nprobe = req.nprobe.unwrap_or(1);
            let scored = ivf.search(&req.vector, req.k, nprobe);
            scored
                .into_iter()
                .map(|s| SearchResultItem {
                    id: s.id,
                    score: s.score,
                })
                .collect()
        }
        CollectionIndex::IvfPq(ivfpq) => {
            if !ivfpq.is_trained {
                return Err(AppError::BadRequest(
                    "IVF-PQ collection is untrained and cannot be searched".to_string(),
                ));
            }
            let nprobe = req.nprobe.unwrap_or(1);
            let scored = ivfpq
                .search(&req.vector, req.k, nprobe)
                .map_err(|e| AppError::BadRequest(e))?;
            scored
                .into_iter()
                .map(|s| SearchResultItem {
                    id: s.id,
                    score: s.score,
                })
                .collect()
        }
    };

    Ok(Json(SearchResponse { results }))
}
