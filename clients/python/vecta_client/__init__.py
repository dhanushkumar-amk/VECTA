"""Vecta Python Client SDK.

Provides an HTTP client for interacting with a running Vecta vector database server,
plus optional integrations for LangChain.
"""

from .client import VectaClient, VectaAPIError

try:
    from .langchain_integration import VectaVectorStore
    __all__ = ["VectaClient", "VectaAPIError", "VectaVectorStore"]
except ImportError:
    __all__ = ["VectaClient", "VectaAPIError"]
