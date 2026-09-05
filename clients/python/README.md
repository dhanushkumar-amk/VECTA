# Vecta Python Client SDK

Pure-Python HTTP client SDK for interacting with a running Vecta vector database REST API server.

## Installation

```bash
pip install vecta-client
# Or for local development:
pip install -e clients/python/
```

## Quickstart

```python
from vecta_client import VectaClient

# Initialize client
client = VectaClient("http://localhost:6333")

# Check health
health = client.health()
print("Health:", health)

# Create collection
client.create_collection("docs", dim=128, index_type="flat", metric="euclidean")

# Insert point
client.insert("docs", id=1, vector=[0.1] * 128)

# Search
results = client.search("docs", vector=[0.1] * 128, k=5)
print("Search results:", results)
```
