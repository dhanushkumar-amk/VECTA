"""Integration tests for VectaClient SDK against a running server."""

import os
import time
import pytest
import requests
from vecta_client import VectaClient, VectaAPIError

SERVER_URL = os.environ.get("VECTA_SERVER_URL", "http://127.0.0.1:6333")


@pytest.fixture(scope="module")
def client():
    """Ensure server is reachable, then yield client."""
    c = VectaClient(base_url=SERVER_URL, timeout=10.0)
    # Check if server is accessible
    try:
        c.health()
    except Exception as exc:
        pytest.skip(f"Vecta server is not running at {SERVER_URL}: {exc}")
    return c


def test_01_health(client):
    res = client.health()
    assert isinstance(res, dict)
    assert res.get("status") == "ok"


def test_02_full_round_trip(client):
    col_name = f"pyclient_test_{int(time.time())}"
    create_res = client.create_collection(
        name=col_name,
        dim=2,
        index_type="flat",
        metric="euclidean",
    )
    assert create_res["name"] == col_name
    assert create_res["dim"] == 2
    assert create_res["vector_count"] == 0

    # Insert points
    client.insert(col_name, id=1, vector=[1.0, 1.0])
    client.insert(col_name, id=2, vector=[2.0, 2.0])
    client.insert(col_name, id=3, vector=[9.0, 9.0])

    # Detail
    detail = client.get_collection(col_name)
    assert detail["vector_count"] == 3

    # Search
    results = client.search(col_name, vector=[1.0, 1.1], k=2)
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[1]["id"] == 2

    # Checkpoint
    client.checkpoint(col_name)

    # Cleanup
    client.delete_collection(col_name)
    collections = client.list_collections()
    assert not any(c["name"] == col_name for c in collections)


def test_03_error_handling(client):
    with pytest.raises(VectaAPIError) as exc_info:
        client.search("non_existent_collection_xyz", vector=[1.0, 2.0], k=1)

    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.message.lower()


def test_04_delete_and_verify(client):
    col_name = f"to_delete_{int(time.time())}"
    client.create_collection(name=col_name, dim=3, index_type="flat")

    collections = [c["name"] for c in client.list_collections()]
    assert col_name in collections

    client.delete_collection(col_name)
    collections_after = [c["name"] for c in client.list_collections()]
    assert col_name not in collections_after

    with pytest.raises(VectaAPIError) as exc_info:
        client.get_collection(col_name)
    assert exc_info.value.status_code == 404


def test_05_timeout_behavior():
    # Pass an extremely short timeout against an unreachable/blackhole IP to verify Timeout exception
    slow_client = VectaClient(base_url="http://10.255.255.1:6333", timeout=0.001)
    with pytest.raises((requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        slow_client.health()
