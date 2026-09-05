//! Route registrations for the Vecta REST API.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use crate::server::handlers::{
    create_collection_handler, delete_collection_handler, get_collection_handler, health_handler,
    insert_point_handler, list_collections_handler, search_handler,
};
use crate::server::state::AppState;

/// Construct the Axum application router with all registered API endpoints and shared state.
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
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
        .with_state(state)
}
