"""LangChain VectorStore integration for Vecta.

Allows Vecta to be used as a drop-in vector store in LangChain applications.
"""

from typing import Any, Iterable, List, Optional
import uuid

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as e:
    raise ImportError(
        "langchain-core is required to use VectaVectorStore. "
        "Install it with `pip install langchain-core` or `pip install 'vecta-client[langchain]'`."
    ) from e

from .client import VectaClient


class VectaVectorStore(VectorStore):
    """LangChain VectorStore implementation backed by Vecta vector database.

    Parameters
    ----------
    client : VectaClient
        An active VectaClient instance configured with server URL.
    collection : str
        Target collection name on the Vecta server.
    embedding : Embeddings
        A LangChain Embeddings instance used to convert text to vectors.
    """

    def __init__(
        self,
        client: VectaClient,
        collection: str,
        embedding: Embeddings,
    ):
        self.client = client
        self.collection = collection
        self.embedding = embedding

        # In-memory document and metadata storage for this v1 integration.
        # Note: Core metadata filtering exists at the Rust engine level (Phase 29),
        # but metadata REST endpoints are planned for subsequent phases.
        self._text_store: dict[int, str] = {}
        self._metadata_store: dict[int, dict] = {}
        self._next_id: int = 1

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run texts through embeddings and insert them into the Vecta collection.

        Parameters
        ----------
        texts : Iterable[str]
            Texts to add to the vector store.
        metadatas : Optional[List[dict]]
            Optional metadata associated with each text.
        ids : Optional[List[str]]
            Optional IDs for the documents.

        Returns
        -------
        List[str]
            List of IDs of the added texts.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        embeddings = self.embedding.embed_documents(texts_list)
        returned_ids = []

        for idx, (text, vector) in enumerate(zip(texts_list, embeddings)):
            if ids and idx < len(ids):
                # Try parsing as int or hash to int
                try:
                    point_id = int(ids[idx])
                except ValueError:
                    point_id = abs(hash(ids[idx])) % (2**63 - 1)
                str_id = ids[idx]
            else:
                point_id = self._next_id
                self._next_id += 1
                str_id = str(point_id)

            self.client.insert(self.collection, id=point_id, vector=vector)
            self._text_store[point_id] = text

            if metadatas and idx < len(metadatas):
                self._metadata_store[point_id] = metadatas[idx]
            else:
                self._metadata_store[point_id] = {}

            returned_ids.append(str_id)

        return returned_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to query text.

        Parameters
        ----------
        query : str
            Query string to compare.
        k : int
            Number of nearest documents to return (default: 4).

        Returns
        -------
        List[Document]
            List of matched documents ordered by similarity.
        """
        query_vector = self.embedding.embed_query(query)
        results = self.client.search(
            collection=self.collection,
            vector=query_vector,
            k=k,
            nprobe=kwargs.get("nprobe"),
            ef_search=kwargs.get("ef_search"),
        )

        documents = []
        for match in results:
            doc_id = match["id"]
            content = self._text_store.get(doc_id, "")
            metadata = dict(self._metadata_store.get(doc_id, {}))
            metadata["_id"] = doc_id
            metadata["_score"] = match.get("score")
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "VectaVectorStore":
        """Factory constructor: embed texts and create a new VectaVectorStore.

        Parameters
        ----------
        texts : List[str]
            Texts to add to the vector store.
        embedding : Embeddings
            Text embedding model.
        metadatas : Optional[List[dict]]
            Optional metadata associated with each text.

        Keyword Arguments
        -----------------
        client : Optional[VectaClient]
            An existing client instance, or created using ``base_url``.
        base_url : Optional[str]
            Server URL if constructing a new client (default: ``"http://localhost:6333"``).
        collection : Optional[str]
            Name of the collection to create (default: auto-generated UUID).
        index_type : Optional[str]
            Index architecture (default: ``"flat"``).
        metric : Optional[str]
            Distance metric (default: ``"euclidean"``).

        Returns
        -------
        VectaVectorStore
            Initialized vector store populated with embedded texts.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        client = kwargs.get("client")
        if client is None:
            base_url = kwargs.get("base_url", "http://localhost:6333")
            client = VectaClient(base_url=base_url)

        collection_name = kwargs.get("collection", f"langchain_{uuid.uuid4().hex[:8]}")
        index_type = kwargs.get("index_type", "flat")
        metric = kwargs.get("metric", "euclidean")

        # Probe dimension using first text embedding
        sample_vec = embedding.embed_query(texts[0])
        dim = len(sample_vec)

        # Create collection if it doesn't already exist
        try:
            client.create_collection(
                name=collection_name,
                dim=dim,
                index_type=index_type,
                metric=metric,
            )
        except Exception:
            # If collection already exists, proceed to insert
            pass

        store = cls(client=client, collection=collection_name, embedding=embedding)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store
