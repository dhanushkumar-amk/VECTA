"""Integration tests for Vecta's LangChain VectorStore integration."""

import os
import time
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from vecta_client import VectaClient
from vecta_client.langchain_integration import VectaVectorStore

SERVER_URL = os.environ.get("VECTA_SERVER_URL", "http://127.0.0.1:6333")


class DeterministicFakeEmbeddings(Embeddings):
    """Deterministic embeddings returning fixed coordinates for predictable nearest neighbor testing."""

    def __init__(self, mapping: dict, default_dim: int = 2):
        self.mapping = mapping
        self.default_dim = default_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping.get(t, [0.0] * self.default_dim) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.mapping.get(text, [0.0] * self.default_dim)


@pytest.fixture(scope="module")
def client():
    c = VectaClient(base_url=SERVER_URL, timeout=10.0)
    try:
        c.health()
    except Exception as exc:
        pytest.skip(f"Vecta server is not running at {SERVER_URL}: {exc}")
    return c


@pytest.fixture
def fake_embeddings():
    mapping = {
        "apple is a fruit": [1.0, 1.0],
        "banana is yellow": [1.0, 1.5],
        "heavy cargo diesel truck": [9.0, 9.0],
        "query: fruit": [1.0, 1.1],
        "new document orange": [1.2, 1.0],
    }
    return DeterministicFakeEmbeddings(mapping=mapping, default_dim=2)


def test_01_from_texts_and_similarity_search(client, fake_embeddings):
    col_name = f"langchain_test_{int(time.time())}"
    texts = [
        "apple is a fruit",
        "banana is yellow",
        "heavy cargo diesel truck",
    ]
    metadatas = [
        {"category": "fruit", "source": "produce"},
        {"category": "fruit", "source": "produce"},
        {"category": "vehicle", "source": "transport"},
    ]

    # 1. from_texts factory constructor
    vectorstore = VectaVectorStore.from_texts(
        texts=texts,
        embedding=fake_embeddings,
        metadatas=metadatas,
        client=client,
        collection=col_name,
        index_type="flat",
        metric="euclidean",
    )

    assert isinstance(vectorstore, VectaVectorStore)
    assert vectorstore.collection == col_name

    # 2. similarity_search
    # Query vector is [1.0, 1.1] -> closest is "apple is a fruit" ([1.0, 1.0], dist ≈ 0.1)
    # followed by "banana is yellow" ([1.0, 1.5], dist ≈ 0.4)
    docs = vectorstore.similarity_search("query: fruit", k=2)

    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "apple is a fruit"
    assert docs[0].metadata["category"] == "fruit"
    assert docs[1].page_content == "banana is yellow"

    # Cleanup
    client.delete_collection(col_name)


def test_02_add_texts_subsequent(client, fake_embeddings):
    col_name = f"langchain_add_{int(time.time())}"
    initial_texts = ["apple is a fruit"]
    initial_meta = [{"item": "apple"}]

    vectorstore = VectaVectorStore.from_texts(
        texts=initial_texts,
        embedding=fake_embeddings,
        metadatas=initial_meta,
        client=client,
        collection=col_name,
    )

    # Add more texts to the existing store
    new_ids = vectorstore.add_texts(
        texts=["new document orange"],
        metadatas=[{"item": "orange"}],
    )
    assert len(new_ids) == 1

    # Search query: fruit should find both fruit docs
    results = vectorstore.similarity_search("query: fruit", k=2)
    assert len(results) == 2
    contents = [d.page_content for d in results]
    assert "apple is a fruit" in contents
    assert "new document orange" in contents

    # Cleanup
    client.delete_collection(col_name)
