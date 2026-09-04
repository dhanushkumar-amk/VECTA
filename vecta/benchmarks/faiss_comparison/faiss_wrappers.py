"""
FAISS Index Factory Wrappers (Phase 35).

Provides factory functions to construct FAISS CPU indexes matching vecta's
architectural parameters, lifecycle, and metric semantics.

Lifecycle Symmetry Note:
FAISS approximate indexes (IndexIVFFlat, IndexIVFPQ) require an explicit
training phase (`index.train(vectors)`) prior to ingestion (`index.add(vectors)`),
exactly mirroring vecta's train-then-add lifecycle established in Phase 11.
HNSW indexes in both libraries do not require training, but begin empty and
construct graphs incrementally during insertion.
"""

from typing import Optional, Union
import faiss
import numpy as np


def set_faiss_threads(num_threads: int = 1) -> None:
    """
    Explicitly set OpenMP thread count used by FAISS.

    Crucial for apples-to-apples comparison: FAISS defaults to using all logical CPU cores,
    while vecta core indexes run single-threaded per query.
    """
    faiss.omp_set_num_threads(num_threads)


def get_faiss_threads() -> int:
    """Return currently configured OpenMP thread count in FAISS."""
    return faiss.omp_get_max_threads()


def _resolve_metric(metric: str) -> int:
    """Translate user metric string to FAISS metric integer constant."""
    m = metric.lower()
    if m in ("euclidean", "l2"):
        return faiss.METRIC_L2
    elif m in ("dot_product", "dot", "ip"):
        return faiss.METRIC_INNER_PRODUCT
    elif m in ("cosine", "cos"):
        # Note: For Cosine similarity, vectors must be L2-normalized upfront.
        # Once normalized, Inner Product equals Cosine Similarity.
        return faiss.METRIC_INNER_PRODUCT
    else:
        raise ValueError(
            f"Unsupported FAISS metric '{metric}': expected 'euclidean', 'dot_product', or 'cosine'"
        )


def build_faiss_flat(dim: int, metric: str = "euclidean") -> faiss.Index:
    """
    Construct a FAISS brute-force flat index equivalent to `vecta.FlatIndex(dim, metric)`.

    Args:
        dim: Dimensionality of vectors.
        metric: Distance metric ('euclidean', 'dot_product', 'cosine').

    Returns:
        faiss.IndexFlatL2 or faiss.IndexFlatIP.
    """
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")

    metric_type = _resolve_metric(metric)
    if metric_type == faiss.METRIC_L2:
        return faiss.IndexFlatL2(dim)
    else:
        return faiss.IndexFlatIP(dim)


def build_faiss_ivf(
    dim: int,
    nlist: int = 100,
    metric: str = "euclidean",
) -> faiss.IndexIVFFlat:
    """
    Construct a FAISS Inverted File index equivalent to `vecta.IVFIndex(dim, num_clusters=nlist, metric)`.

    Args:
        dim: Dimensionality of vectors.
        nlist: Number of Voronoi coarse centroids (num_clusters in vecta).
        metric: Distance metric ('euclidean', 'dot_product', 'cosine').

    Returns:
        Untrained faiss.IndexIVFFlat instance.
    """
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    if nlist <= 0:
        raise ValueError(f"nlist must be positive, got {nlist}")

    metric_type = _resolve_metric(metric)
    quantizer = (
        faiss.IndexFlatL2(dim)
        if metric_type == faiss.METRIC_L2
        else faiss.IndexFlatIP(dim)
    )

    index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric_type)
    return index


def build_faiss_hnsw(
    dim: int,
    m: int = 16,
    ef_construction: int = 100,
    metric: str = "euclidean",
) -> faiss.IndexHNSWFlat:
    """
    Construct a FAISS HNSW graph index equivalent to `vecta.HnswIndex(dim, m=M, ef_construction)`.

    Args:
        dim: Dimensionality of vectors.
        m: Bi-directional link degree per node (M in FAISS / vecta).
        ef_construction: Size of dynamic candidate list during graph construction.
        metric: Distance metric ('euclidean', 'dot_product', 'cosine').

    Returns:
        faiss.IndexHNSWFlat instance.
    """
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    if m <= 1:
        raise ValueError(f"m must be > 1, got {m}")
    if ef_construction <= 0:
        raise ValueError(f"ef_construction must be positive, got {ef_construction}")

    metric_type = _resolve_metric(metric)
    index = faiss.IndexHNSWFlat(dim, m, metric_type)
    index.hnsw.efConstruction = ef_construction
    return index


def build_faiss_ivfpq(
    dim: int,
    nlist: int = 100,
    m: int = 8,
    nbits: int = 8,
    metric: str = "euclidean",
) -> faiss.IndexIVFPQ:
    """
    Construct a FAISS IVFPQ index equivalent to `vecta.IVFPQIndex(dim, num_clusters=nlist, m, k_per_subvector=2^nbits)`.

    Parameter Mapping:
        - dim: Vector dimension (must be divisible by m).
        - nlist: Coarse cluster count (num_clusters in vecta).
        - m: Number of sub-quantizer subvectors.
        - nbits: Bits per sub-vector code. nbits=8 corresponds to k_per_subvector=256 in vecta.
        - metric: 'euclidean' (vecta IVFPQ is Euclidean-only).

    Returns:
        Untrained faiss.IndexIVFPQ instance.
    """
    if dim <= 0 or dim % m != 0:
        raise ValueError(f"Dimension {dim} must be divisible by m={m}")
    if nlist <= 0:
        raise ValueError(f"nlist must be positive, got {nlist}")
    if nbits not in (4, 8, 16):
        raise ValueError(f"nbits must be 4, 8, or 16, got {nbits}")

    metric_type = _resolve_metric(metric)
    quantizer = (
        faiss.IndexFlatL2(dim)
        if metric_type == faiss.METRIC_L2
        else faiss.IndexFlatIP(dim)
    )

    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, metric_type)
    return index


def set_search_params(
    index: faiss.Index,
    nprobe: Optional[int] = None,
    ef_search: Optional[int] = None,
) -> None:
    """
    Helper to set approximate search hyperparameters on FAISS index objects.

    - If index is IndexIVF / IndexIVFFlat / IndexIVFPQ: sets `index.nprobe`.
    - If index is IndexHNSWFlat: sets `index.hnsw.efSearch`.
    """
    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = nprobe
    if ef_search is not None and hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef_search
