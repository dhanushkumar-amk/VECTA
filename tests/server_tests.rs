//! Integration tests for the Vecta REST API server (Phase 41).

use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

use vecta::server::{create_router, AppState};

/// Helper: start a test server instance listening on an ephemeral port.
/// Returns the base URL (e.g. "http://127.0.0.1:54321").
async fn start_test_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind ephemeral port");
    let addr = listener.local_addr().expect("failed to get local addr");
    let test_dir = PathBuf::from(format!("./target/test_server_{}", addr.port()));
    let _ = std::fs::remove_dir_all(&test_dir);
    let state = Arc::new(AppState::new(test_dir, None));
    let app = create_router(state);

    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("server failed to serve");
    });

    format!("http://{}", addr)
}

#[tokio::test]
async fn test_01_get_health() {
    let base_url = start_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.expect("json parse failed");
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn test_02_create_and_list_collection() {
    let base_url = start_test_server().await;
    let client = reqwest::Client::new();

    let create_payload = serde_json::json!({
        "name": "test_flat",
        "dim": 2,
        "index_type": "flat",
        "metric": "euclidean"
    });

    let resp = client
        .post(format!("{}/collections", base_url))
        .json(&create_payload)
        .send()
        .await
        .expect("create request failed");

    assert_eq!(resp.status(), reqwest::StatusCode::CREATED);
    let body: serde_json::Value = resp.json().await.expect("json parse failed");
    assert_eq!(body["name"], "test_flat");
    assert_eq!(body["index_type"], "flat");
    assert_eq!(body["dim"], 2);
    assert_eq!(body["metric"], "euclidean");
    assert_eq!(body["vector_count"], 0);

    let list_resp = client
        .get(format!("{}/collections", base_url))
        .send()
        .await
        .expect("list request failed");

    assert_eq!(list_resp.status(), reqwest::StatusCode::OK);
    let list_body: Vec<serde_json::Value> = list_resp.json().await.expect("json parse failed");
    assert_eq!(list_body.len(), 1);
    assert_eq!(list_body[0]["name"], "test_flat");
}

#[tokio::test]
async fn test_03_create_duplicate_collection_returns_409() {
    let base_url = start_test_server().await;
    let client = reqwest::Client::new();

    let payload = serde_json::json!({
        "name": "dup_col",
        "dim": 4,
        "index_type": "flat",
        "metric": "cosine"
    });

    let resp1 = client
        .post(format!("{}/collections", base_url))
        .json(&payload)
        .send()
        .await
        .expect("first create failed");
    assert_eq!(resp1.status(), reqwest::StatusCode::CREATED);

    let resp2 = client
        .post(format!("{}/collections", base_url))
        .json(&payload)
        .send()
        .await
        .expect("second create failed");
    assert_eq!(resp2.status(), reqwest::StatusCode::CONFLICT);

    let err_body: serde_json::Value = resp2.json().await.expect("json parse failed");
    assert!(err_body["error"].as_str().unwrap().contains("already exists"));
}

#[tokio::test]
async fn test_04_create_invalid_index_type_returns_400() {
    let base_url = start_test_server().await;
    let client = reqwest::Client::new();

    let payload = serde_json::json!({
        "name": "invalid_col",
        "dim": 4,
        "index_type": "quantum_hyper_tree",
        "metric": "euclidean"
    });

    let resp = client
        .post(format!("{}/collections", base_url))
        .json(&payload)
        .send()
        .await
        .expect("request failed");
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

    let err_body: serde_json::Value = resp.json().await.expect("json parse failed");
    let err_msg = err_body["error"].as_str().unwrap();
    assert!(err_msg.contains("invalid index_type"));
    assert!(err_msg.contains("expected 'flat', 'ivf', 'hnsw', or 'ivfpq'"));
}

#[tokio::test]
async fn test_05_flat_full_flow_hand_verified() {
    let base_url = start_test_server().await;
    let client = reqwest::Client::new();

    // 1. Create collection
    let create_payload = serde_json::json!({
        "name": "canonical_flat",
        "dim": 2,
        "index_type": "flat",
        "metric": "euclidean"
    });
    let resp = client
        .post(format!("{}/collections", base_url))
        .json(&create_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::CREATED);

    // 2. Insert 5 canonical hand-verified vectors
    //   v0 (id=0): [2.0, 1.0]
    //   v1 (id=1): [0.5, 0.0]
    //   v2 (id=2): [1.0, 3.0]
    //   v3 (id=3): [-1.0, 3.0]
    //   v4 (id=4): [4.0, -3.0]
    let test_points: Vec<(u64, Vec<f32>)> = vec![
        (0, vec![2.0, 1.0]),
        (1, vec![0.5, 0.0]),
        (2, vec![1.0, 3.0]),
        (3, vec![-1.0, 3.0]),
        (4, vec![4.0, -3.0]),
    ];

    for (id, vec) in test_points {
        let insert_payload = serde_json::json!({
            "id": id,
            "vector": vec
        });
        let ins_resp = client
            .post(format!("{}/collections/canonical_flat/points", base_url))
            .json(&insert_payload)
            .send()
            .await
            .unwrap();
        assert_eq!(ins_resp.status(), reqwest::StatusCode::CREATED);
    }

    // Confirm collection shows vector_count = 5
    let detail_resp = client
        .get(format!("{}/collections/canonical_flat", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(detail_resp.status(), reqwest::StatusCode::OK);
    let detail_body: serde_json::Value = detail_resp.json().await.unwrap();
    assert_eq!(detail_body["vector_count"], 5);

    // 3. Search query: [1.0, 1.0], k = 3
    // Hand-verified Euclidean distances:
    // v0: dist = 1.0 (id=0)
    // v1: dist = sqrt(1.25) ≈ 1.1180 (id=1)
    // v2: dist = 2.0 (id=2)
    let search_payload = serde_json::json!({
        "vector": [1.0, 1.0],
        "k": 3
    });
    let search_resp = client
        .post(format!("{}/collections/canonical_flat/search", base_url))
        .json(&search_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(search_resp.status(), reqwest::StatusCode::OK);

    let search_body: serde_json::Value = search_resp.json().await.unwrap();
    let results = search_body["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);

    assert_eq!(results[0]["id"], 0);
    assert!((results[0]["score"].as_f64().unwrap() - 1.0).abs() < 1e-4);

    assert_eq!(results[1]["id"], 1);
    assert!((results[1]["score"].as_f64().unwrap() - 1.25_f64.sqrt()).abs() < 1e-4);

    assert_eq!(results[2]["id"], 2);
    assert!((results[2]["score"].as_f64().unwrap() - 2.0).abs() < 1e-4);
}

#[tokio::test]
async fn test_06_search_nonexistent_collection_returns_404() {
    let base_url = start_test_server().await;
    let client = reqwest::Client::new();

    let search_payload = serde_json::json!({
        "vector": [1.0, 2.0, 3.0],
        "k": 5
    });

    let resp = client
        .post(format!("{}/collections/ghost_col/search", base_url))
        .json(&search_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    let err_body: serde_json::Value = resp.json().await.unwrap();
    assert!(err_body["error"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_07_insert_dimension_mismatch_returns_400_and_survives() {
    let base_url = start_test_server().await;
    let client = reqwest::Client::new();

    // Create 2D collection
    let create_payload = serde_json::json!({
        "name": "resilient_col",
        "dim": 2,
        "index_type": "flat",
        "metric": "euclidean"
    });
    client
        .post(format!("{}/collections", base_url))
        .json(&create_payload)
        .send()
        .await
        .unwrap();

    // Send BAD request with 4D vector instead of 2D
    let bad_payload = serde_json::json!({
        "id": 1,
        "vector": [1.0, 2.0, 3.0, 4.0]
    });
    let bad_resp = client
        .post(format!("{}/collections/resilient_col/points", base_url))
        .json(&bad_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(bad_resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let bad_err: serde_json::Value = bad_resp.json().await.unwrap();
    assert!(bad_err["error"].as_str().unwrap().contains("dimension mismatch"));

    // Send a SECOND request (valid 2D vector) to prove server did NOT crash
    let valid_payload = serde_json::json!({
        "id": 1,
        "vector": [1.0, 2.0]
    });
    let valid_resp = client
        .post(format!("{}/collections/resilient_col/points", base_url))
        .json(&valid_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(valid_resp.status(), reqwest::StatusCode::CREATED);

    // Verify search works immediately after
    let search_payload = serde_json::json!({
        "vector": [1.0, 2.0],
        "k": 1
    });
    let search_resp = client
        .post(format!("{}/collections/resilient_col/search", base_url))
        .json(&search_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(search_resp.status(), reqwest::StatusCode::OK);
    let search_body: serde_json::Value = search_resp.json().await.unwrap();
    assert_eq!(search_body["results"].as_array().unwrap().len(), 1);
    assert_eq!(search_body["results"][0]["id"], 1);
}

#[tokio::test]
async fn test_08_hnsw_full_flow_end_to_end() {
    let base_url = start_test_server().await;
    let client = reqwest::Client::new();

    // Create HNSW collection
    let create_payload = serde_json::json!({
        "name": "hnsw_col",
        "dim": 2,
        "index_type": "hnsw",
        "metric": "euclidean"
    });
    let resp = client
        .post(format!("{}/collections", base_url))
        .json(&create_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::CREATED);

    // Insert points
    let points: Vec<(u64, Vec<f32>)> = vec![
        (10, vec![1.0, 1.0]),
        (20, vec![1.0, 1.5]),
        (30, vec![9.0, 9.0]),
    ];

    for (id, vec) in points {
        let insert_payload = serde_json::json!({
            "id": id,
            "vector": vec
        });
        let ins_resp = client
            .post(format!("{}/collections/hnsw_col/points", base_url))
            .json(&insert_payload)
            .send()
            .await
            .unwrap();
        assert_eq!(ins_resp.status(), reqwest::StatusCode::CREATED);
    }

    // Search with ef_search parameter
    let search_payload = serde_json::json!({
        "vector": [1.0, 1.1],
        "k": 2,
        "ef_search": 64
    });
    let search_resp = client
        .post(format!("{}/collections/hnsw_col/search", base_url))
        .json(&search_payload)
        .send()
        .await
        .unwrap();

    assert_eq!(search_resp.status(), reqwest::StatusCode::OK);
    let search_body: serde_json::Value = search_resp.json().await.unwrap();
    let results = search_body["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    // Nearest to [1.0, 1.1] is id 10 ([1.0, 1.0])
    assert_eq!(results[0]["id"], 10);
}

#[tokio::test]
async fn test_09_delete_collection_and_verify_404() {
    let base_url = start_test_server().await;
    let client = reqwest::Client::new();

    // Create collection
    let create_payload = serde_json::json!({
        "name": "to_delete",
        "dim": 3,
        "index_type": "flat",
        "metric": "euclidean"
    });
    client
        .post(format!("{}/collections", base_url))
        .json(&create_payload)
        .send()
        .await
        .unwrap();

    // Verify it exists
    let get_resp = client
        .get(format!("{}/collections/to_delete", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(get_resp.status(), reqwest::StatusCode::OK);

    // Delete it
    let del_resp = client
        .delete(format!("{}/collections/to_delete", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(del_resp.status(), reqwest::StatusCode::OK);

    // Subsequent GET returns 404
    let get_after = client
        .get(format!("{}/collections/to_delete", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(get_after.status(), reqwest::StatusCode::NOT_FOUND);

    // Subsequent DELETE returns 404
    let del_after = client
        .delete(format!("{}/collections/to_delete", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(del_after.status(), reqwest::StatusCode::NOT_FOUND);
}
