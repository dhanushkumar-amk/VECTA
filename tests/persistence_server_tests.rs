//! Integration tests for Phase 43: Server Persistence Wiring & WAL Crash Recovery.

use std::fs::create_dir_all;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

use vecta::server::{create_router, AppState};

/// Helper: start a test server instance listening on an ephemeral port,
/// bound to a specific test data directory.
async fn start_server_with_data_dir(data_dir: PathBuf) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind ephemeral port");
    let addr = listener.local_addr().expect("failed to get local addr");
    let state = Arc::new(AppState::new(data_dir));
    let app = create_router(state);

    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("server failed to serve");
    });

    format!("http://{}", addr)
}

#[tokio::test]
async fn test_01_manual_checkpoint_and_restart_restores_data() {
    let test_dir = PathBuf::from("./target/test_persistence_manual_checkpoint");
    let _ = std::fs::remove_dir_all(&test_dir);
    create_dir_all(&test_dir).unwrap();

    let client = reqwest::Client::new();

    // 1. Start Server Instance 1
    let base_url1 = start_server_with_data_dir(test_dir.clone()).await;

    // 2. Create Flat collection
    let create_payload = serde_json::json!({
        "name": "persistent_flat",
        "dim": 2,
        "index_type": "flat",
        "metric": "euclidean"
    });
    let resp = client
        .post(format!("{}/collections", base_url1))
        .json(&create_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::CREATED);

    // 3. Insert points
    let points: Vec<(u64, Vec<f32>)> = vec![
        (100, vec![1.0, 2.0]),
        (200, vec![3.0, 4.0]),
    ];
    for (id, vec) in points {
        client
            .post(format!("{}/collections/persistent_flat/points", base_url1))
            .json(&serde_json::json!({ "id": id, "vector": vec }))
            .send()
            .await
            .unwrap();
    }

    // 4. Call checkpoint endpoint
    let cp_resp = client
        .post(format!("{}/collections/persistent_flat/checkpoint", base_url1))
        .send()
        .await
        .unwrap();
    assert_eq!(cp_resp.status(), reqwest::StatusCode::OK);

    // 5. Start Server Instance 2 pointing to the SAME data directory (simulates restart)
    let base_url2 = start_server_with_data_dir(test_dir.clone()).await;

    // 6. Verify collection exists on Instance 2
    let get_resp = client
        .get(format!("{}/collections/persistent_flat", base_url2))
        .send()
        .await
        .unwrap();
    assert_eq!(get_resp.status(), reqwest::StatusCode::OK);
    let info: serde_json::Value = get_resp.json().await.unwrap();
    assert_eq!(info["vector_count"], 2);

    // 7. Search query [1.0, 2.0] k=1 returns id 100
    let search_resp = client
        .post(format!("{}/collections/persistent_flat/search", base_url2))
        .json(&serde_json::json!({ "vector": [1.0, 2.0], "k": 1 }))
        .send()
        .await
        .unwrap();
    assert_eq!(search_resp.status(), reqwest::StatusCode::OK);
    let search_body: serde_json::Value = search_resp.json().await.unwrap();
    let results = search_body["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["id"], 100);
}

#[tokio::test]
async fn test_02_wal_crash_recovery_without_checkpoint() {
    let test_dir = PathBuf::from("./target/test_persistence_wal_crash");
    let _ = std::fs::remove_dir_all(&test_dir);
    create_dir_all(&test_dir).unwrap();

    let client = reqwest::Client::new();

    // 1. Start Server Instance 1
    let base_url1 = start_server_with_data_dir(test_dir.clone()).await;

    // 2. Create Flat collection (initial empty snapshot saved)
    let create_payload = serde_json::json!({
        "name": "wal_col",
        "dim": 2,
        "index_type": "flat",
        "metric": "euclidean"
    });
    client
        .post(format!("{}/collections", base_url1))
        .json(&create_payload)
        .send()
        .await
        .unwrap();

    // 3. Insert points WITHOUT checkpointing (live WAL logging only!)
    let points: Vec<(u64, Vec<f32>)> = vec![
        (1, vec![10.0, 10.0]),
        (2, vec![20.0, 20.0]),
        (3, vec![30.0, 30.0]),
    ];
    for (id, vec) in points {
        client
            .post(format!("{}/collections/wal_col/points", base_url1))
            .json(&serde_json::json!({ "id": id, "vector": vec }))
            .send()
            .await
            .unwrap();
    }

    // Verify .wal file exists on disk and is non-empty
    let wal_path = test_dir.join("wal_col.wal");
    assert!(wal_path.exists());
    assert!(std::fs::metadata(&wal_path).unwrap().len() > 0);

    // 4. Ungraceful restart: Start Server Instance 2 WITHOUT having called /checkpoint
    let base_url2 = start_server_with_data_dir(test_dir.clone()).await;

    // 5. Confirm WAL replay restored all 3 points on startup
    let get_resp = client
        .get(format!("{}/collections/wal_col", base_url2))
        .send()
        .await
        .unwrap();
    assert_eq!(get_resp.status(), reqwest::StatusCode::OK);
    let info: serde_json::Value = get_resp.json().await.unwrap();
    assert_eq!(info["vector_count"], 3);

    // 6. Search on the recovered collection
    let search_resp = client
        .post(format!("{}/collections/wal_col/search", base_url2))
        .json(&serde_json::json!({ "vector": [20.0, 20.0], "k": 1 }))
        .send()
        .await
        .unwrap();
    assert_eq!(search_resp.status(), reqwest::StatusCode::OK);
    let search_body: serde_json::Value = search_resp.json().await.unwrap();
    let results = search_body["results"].as_array().unwrap();
    assert_eq!(results[0]["id"], 2);
}

#[tokio::test]
async fn test_03_mixed_collection_types_restored_via_peek_index_type() {
    let test_dir = PathBuf::from("./target/test_persistence_mixed");
    let _ = std::fs::remove_dir_all(&test_dir);
    create_dir_all(&test_dir).unwrap();

    let client = reqwest::Client::new();

    // 1. Start Server Instance 1
    let base_url1 = start_server_with_data_dir(test_dir.clone()).await;

    // 2. Create Flat collection and insert point
    client
        .post(format!("{}/collections", base_url1))
        .json(&serde_json::json!({
            "name": "mixed_flat",
            "dim": 2,
            "index_type": "flat",
            "metric": "euclidean"
        }))
        .send()
        .await
        .unwrap();

    client
        .post(format!("{}/collections/mixed_flat/points", base_url1))
        .json(&serde_json::json!({ "id": 1, "vector": [1.0, 1.0] }))
        .send()
        .await
        .unwrap();

    // 3. Create HNSW collection and insert point
    client
        .post(format!("{}/collections", base_url1))
        .json(&serde_json::json!({
            "name": "mixed_hnsw",
            "dim": 2,
            "index_type": "hnsw",
            "metric": "euclidean"
        }))
        .send()
        .await
        .unwrap();

    client
        .post(format!("{}/collections/mixed_hnsw/points", base_url1))
        .json(&serde_json::json!({ "id": 2, "vector": [2.0, 2.0] }))
        .send()
        .await
        .unwrap();

    // 4. Checkpoint both collections
    client
        .post(format!("{}/collections/mixed_flat/checkpoint", base_url1))
        .send()
        .await
        .unwrap();
    client
        .post(format!("{}/collections/mixed_hnsw/checkpoint", base_url1))
        .send()
        .await
        .unwrap();

    // 5. Restart server pointing to same directory
    let base_url2 = start_server_with_data_dir(test_dir.clone()).await;

    // 6. List collections to confirm both types are restored with correct types
    let list_resp = client
        .get(format!("{}/collections", base_url2))
        .send()
        .await
        .unwrap();
    let list: Vec<serde_json::Value> = list_resp.json().await.unwrap();
    assert_eq!(list.len(), 2);

    let flat_col = list.iter().find(|c| c["name"] == "mixed_flat").unwrap();
    assert_eq!(flat_col["index_type"], "flat");
    assert_eq!(flat_col["vector_count"], 1);

    let hnsw_col = list.iter().find(|c| c["name"] == "mixed_hnsw").unwrap();
    assert_eq!(hnsw_col["index_type"], "hnsw");
    assert_eq!(hnsw_col["vector_count"], 1);
}
