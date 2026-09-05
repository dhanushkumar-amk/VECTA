//! Standalone binary entry point for the Vecta REST API server.

use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::net::TcpListener;
use vecta::server::{create_router, AppState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port: u16 = env::var("VECTA_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(6333);

    let data_dir = env::var("VECTA_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./data"));

    let state = Arc::new(AppState::new(data_dir));
    let app = create_router(state);

    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
    let listener = TcpListener::bind(addr).await?;

    println!("vecta-server listening on 0.0.0.0:{}", port);

    axum::serve(listener, app).await?;

    Ok(())
}
