//! Route registrations, OpenAPI generation, and Swagger UI integration for Vecta.

use std::sync::Arc;

use axum::middleware::from_fn_with_state;
use axum::routing::{get, post};
use axum::Router;
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::{Modify, OpenApi};
use utoipa_swagger_ui::SwaggerUi;

use crate::server::handlers::{
    auth_middleware, checkpoint_handler, create_collection_handler, delete_collection_handler,
    get_collection_handler, health_handler, insert_point_handler, list_collections_handler,
    search_handler,
};
use crate::server::models::{
    CollectionInfo, CreateCollectionRequest, ErrorResponse, HealthResponse, InsertPointRequest,
    SearchRequest, SearchResponse, SearchResultItem,
};
use crate::server::state::AppState;

/// Security scheme modifier registering HTTP Bearer authentication in the OpenAPI spec.
struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "BearerAuth",
                SecurityScheme::Http(
                    HttpBuilder::new()
                        .scheme(HttpAuthScheme::Bearer)
                        .bearer_format("Token")
                        .description(Some(
                            "Enter your VECTA_API_KEY secret token (e.g. Bearer <token>)",
                        ))
                        .build(),
                ),
            );
        }
    }
}

/// The root OpenAPI specification for Vecta REST API.
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::server::handlers::health_handler,
        crate::server::handlers::create_collection_handler,
        crate::server::handlers::list_collections_handler,
        crate::server::handlers::get_collection_handler,
        crate::server::handlers::delete_collection_handler,
        crate::server::handlers::checkpoint_handler,
        crate::server::handlers::insert_point_handler,
        crate::server::handlers::search_handler,
    ),
    components(
        schemas(
            CreateCollectionRequest,
            InsertPointRequest,
            SearchRequest,
            SearchResultItem,
            SearchResponse,
            CollectionInfo,
            ErrorResponse,
            HealthResponse,
        )
    ),
    modifiers(&SecurityAddon),
    tags(
        (name = "System", description = "System health and liveness checks"),
        (name = "Collections", description = "Vector collection lifecycle management"),
        (name = "Points", description = "Vector ingestion and insertion"),
        (name = "Search", description = "k-Nearest Neighbor approximate and exact search"),
        (name = "Persistence", description = "Durability, snapshot saving, and WAL checkpointing")
    ),
    info(
        title = "Vecta REST API",
        version = "0.1.0",
        description = "High-performance vector database engine built in pure Rust. Benchmarked against FAISS."
    )
)]
pub struct ApiDoc;

/// Construct the Axum application router with Swagger UI, OpenAPI spec, and API key auth.
pub fn create_router(state: Arc<AppState>) -> Router {
    // Protected routes requiring authentication if VECTA_API_KEY is configured
    let protected_routes = Router::new()
        .route(
            "/collections",
            post(create_collection_handler).get(list_collections_handler),
        )
        .route(
            "/collections/:name",
            get(get_collection_handler).delete(delete_collection_handler),
        )
        .route("/collections/:name/points", post(insert_point_handler))
        .route("/collections/:name/search", post(search_handler))
        .route("/collections/:name/checkpoint", post(checkpoint_handler))
        .layer(from_fn_with_state(state.clone(), auth_middleware));

    // Public endpoints: /health probe and interactive Swagger documentation
    Router::new()
        .route("/health", get(health_handler))
        .merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .merge(protected_routes)
        .with_state(state)
}
