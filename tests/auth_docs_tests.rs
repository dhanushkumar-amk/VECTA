//! Integration tests for Phase 46: API Key Authentication & OpenAPI/Swagger Documentation.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

use vecta::server::{create_router, AppState};

/// Helper: start test server with optional API key on an ephemeral port.
async fn start_auth_test_server(api_key: Option<String>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind ephemeral port");
    let addr = listener.local_addr().expect("failed to get local addr");
    let test_dir = PathBuf::from(format!("./target/test_auth_{}", addr.port()));
    let state = Arc::new(AppState::new(test_dir, api_key));
    let app = create_router(state);

    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("server failed to serve");
    });

    format!("http://{}", addr)
}

#[tokio::test]
async fn test_01_unset_api_key_allows_unauthenticated_requests() {
    let base_url = start_auth_test_server(None).await;
    let client = reqwest::Client::new();

    // Health works
    let h_resp = client.get(format!("{}/health", base_url)).send().await.unwrap();
    assert_eq!(h_resp.status(), reqwest::StatusCode::OK);

    // Create collection works without auth
    let create_resp = client
        .post(format!("{}/collections", base_url))
        .json(&serde_json::json!({
            "name": "open_col",
            "dim": 2,
            "index_type": "flat",
            "metric": "euclidean"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create_resp.status(), reqwest::StatusCode::CREATED);

    // List collections works without auth
    let list_resp = client.get(format!("{}/collections", base_url)).send().await.unwrap();
    assert_eq!(list_resp.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn test_02_missing_auth_header_returns_401_when_key_is_set() {
    let base_url = start_auth_test_server(Some("super-secret-token".to_string())).await;
    let client = reqwest::Client::new();

    let create_resp = client
        .post(format!("{}/collections", base_url))
        .json(&serde_json::json!({
            "name": "blocked_col",
            "dim": 2,
            "index_type": "flat",
            "metric": "euclidean"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(create_resp.status(), reqwest::StatusCode::UNAUTHORIZED);
    let err_body: serde_json::Value = create_resp.json().await.unwrap();
    assert!(err_body["error"].as_str().unwrap().contains("Unauthorized"));
}

#[tokio::test]
async fn test_03_valid_auth_header_succeeds() {
    let base_url = start_auth_test_server(Some("my-valid-api-key".to_string())).await;
    let client = reqwest::Client::new();

    let create_resp = client
        .post(format!("{}/collections", base_url))
        .header("Authorization", "Bearer my-valid-api-key")
        .json(&serde_json::json!({
            "name": "authed_col",
            "dim": 2,
            "index_type": "flat",
            "metric": "euclidean"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(create_resp.status(), reqwest::StatusCode::CREATED);
}

#[tokio::test]
async fn test_04_wrong_auth_header_returns_401() {
    let base_url = start_auth_test_server(Some("correct-key-123".to_string())).await;
    let client = reqwest::Client::new();

    let create_resp = client
        .post(format!("{}/collections", base_url))
        .header("Authorization", "Bearer WRONG-KEY")
        .json(&serde_json::json!({
            "name": "fail_col",
            "dim": 2,
            "index_type": "flat",
            "metric": "euclidean"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(create_resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_05_health_is_always_open_even_with_api_key_set() {
    let base_url = start_auth_test_server(Some("strict-vault-key".to_string())).await;
    let client = reqwest::Client::new();

    // No Authorization header
    let resp = client.get(format!("{}/health", base_url)).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn test_06_swagger_ui_html_served_at_docs() {
    let base_url = start_auth_test_server(None).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/docs/", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let html = resp.text().await.unwrap();
    assert!(html.contains("swagger-ui") || html.contains("Swagger UI") || html.contains("html"));
}

#[tokio::test]
async fn test_07_openapi_json_structure() {
    let base_url = start_auth_test_server(None).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/api-docs/openapi.json", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let spec: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(spec["openapi"], "3.0.3");
    assert_eq!(spec["info"]["title"], "Vecta REST API");
    assert!(spec["paths"]["/collections"].is_object());
    assert!(spec["paths"]["/health"].is_object());
    assert!(spec["paths"]["/collections/{name}/search"].is_object());
    assert!(spec["components"]["securitySchemes"]["BearerAuth"].is_object());
}
