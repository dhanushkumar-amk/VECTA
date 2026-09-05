//! Standalone Axum REST API server module for Vecta.
//!
//! Provides an HTTP interface for interacting with Vecta collections,
//! inserting vectors, and executing k-NN searches.

pub mod handlers;
pub mod models;
pub mod routes;
pub mod state;

pub use handlers::AppError;
pub use routes::create_router;
pub use state::AppState;
