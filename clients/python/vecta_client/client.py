"""Pure-Python HTTP client for the Vecta REST API server."""

from typing import Any, Dict, List, Optional
import requests


class VectaAPIError(Exception):
    """Exception raised for non-2xx HTTP responses from the Vecta server."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Vecta API Error [{status_code}]: {message}")


class VectaClient:
    """Client for interacting with a running Vecta vector database server over HTTP.

    Parameters
    ----------
    base_url : str
        The base URL of the running Vecta server (default: ``"http://localhost:6333"``).
    api_key : Optional[str]
        Optional API authentication token for future server auth integration.
    timeout : float
        Request timeout in seconds (default: 10.0).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _handle_response(self, response: requests.Response) -> Any:
        if 200 <= response.status_code < 300:
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError:
                return response.text

        # Extract structured error message if available
        try:
            err_data = response.json()
            error_message = err_data.get("error", response.text)
        except Exception:
            error_message = response.text or f"HTTP status {response.status_code}"

        raise VectaAPIError(status_code=response.status_code, message=error_message)

    def health(self) -> Dict[str, str]:
        """Check the liveness and health of the Vecta server.

        Returns
        -------
        dict
            Health response dict (e.g. ``{"status": "ok"}``).
        """
        resp = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def create_collection(
        self,
        name: str,
        dim: int,
        index_type: str,
        metric: str = "euclidean",
    ) -> Dict[str, Any]:
        """Create a new collection on the Vecta server.

        Parameters
        ----------
        name : str
            Unique name for the collection.
        dim : int
            Dimensionality of stored vectors.
        index_type : str
            Index architecture (``"flat"``, ``"ivf"``, ``"hnsw"``, ``"ivfpq"``).
        metric : str
            Similarity metric (``"euclidean"``, ``"cosine"``, ``"dot_product"``).

        Returns
        -------
        dict
            Metadata descriptor of the created collection.
        """
        payload = {
            "name": name,
            "dim": dim,
            "index_type": index_type,
            "metric": metric,
        }
        resp = self.session.post(
            f"{self.base_url}/collections",
            json=payload,
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections registered on the server.

        Returns
        -------
        list of dict
            List of collection metadata dictionaries.
        """
        resp = self.session.get(
            f"{self.base_url}/collections",
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def get_collection(self, name: str) -> Dict[str, Any]:
        """Retrieve detailed metadata for a single collection.

        Parameters
        ----------
        name : str
            Name of the collection.

        Returns
        -------
        dict
            Collection metadata dictionary.
        """
        resp = self.session.get(
            f"{self.base_url}/collections/{name}",
            timeout=self.timeout,
        )
        return self._handle_response(resp)

    def delete_collection(self, name: str) -> None:
        """Delete a collection and its associated on-disk state.

        Parameters
        ----------
        name : str
            Name of the collection to delete.
        """
        resp = self.session.delete(
            f"{self.base_url}/collections/{name}",
            timeout=self.timeout,
        )
        self._handle_response(resp)

    def insert(self, collection: str, id: int, vector: List[float]) -> None:
        """Insert a vector with its external ID into a collection.

        Parameters
        ----------
        collection : str
            Target collection name.
        id : int
            Unique external vector identifier.
        vector : list of float
            Coordinates matching collection dimensionality.
        """
        payload = {
            "id": id,
            "vector": vector,
        }
        resp = self.session.post(
            f"{self.base_url}/collections/{collection}/points",
            json=payload,
            timeout=self.timeout,
        )
        self._handle_response(resp)

    def search(
        self,
        collection: str,
        vector: List[float],
        k: int,
        nprobe: Optional[int] = None,
        ef_search: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a k-nearest-neighbor search query on a collection.

        Parameters
        ----------
        collection : str
            Target collection name.
        vector : list of float
            Query vector coordinates.
        k : int
            Number of nearest neighbors to retrieve.
        nprobe : Optional[int]
            Number of clusters to probe (IVF/IVF-PQ only).
        ef_search : Optional[int]
            Dynamic candidate list size (HNSW only).

        Returns
        -------
        list of dict
            Candidate results, each with ``{"id": int, "score": float}``.
        """
        payload: Dict[str, Any] = {
            "vector": vector,
            "k": k,
        }
        if nprobe is not None:
            payload["nprobe"] = nprobe
        if ef_search is not None:
            payload["ef_search"] = ef_search

        resp = self.session.post(
            f"{self.base_url}/collections/{collection}/search",
            json=payload,
            timeout=self.timeout,
        )
        data = self._handle_response(resp)
        return data.get("results", [])

    def checkpoint(self, collection: str) -> None:
        """Manually trigger a snapshot checkpoint and WAL truncation for a collection.

        Parameters
        ----------
        collection : str
            Target collection name.
        """
        resp = self.session.post(
            f"{self.base_url}/collections/{collection}/checkpoint",
            timeout=self.timeout,
        )
        self._handle_response(resp)
