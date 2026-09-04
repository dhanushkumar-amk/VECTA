"""
Matched configuration definitions between vecta and FAISS (Phase 35).

Defines parameter pairs and sweep schedules to guarantee fair, apples-to-apples
comparisons across all four core index types:
1. Flat:   vecta.FlatIndex       <-> faiss.IndexFlatL2 / IndexFlatIP
2. IVF:    vecta.IVFIndex        <-> faiss.IndexIVFFlat
3. HNSW:   vecta.HnswIndex       <-> faiss.IndexHNSWFlat
4. IVFPQ:  vecta.IVFPQIndex      <-> faiss.IndexIVFPQ

Parameter Conversion Notes:
- IVFPQ Quantizer Resolution:
  vecta specifies `k_per_subvector` (number of centroids per subvector, default 256).
  FAISS specifies `nbits` (number of bits used to index centroids per subquantizer).
  Relation: k_per_subvector = 2^nbits. For k_per_subvector = 256, nbits = 8.
- HNSW Search Depth:
  vecta accepts `ef_search` as an argument to `search(query, k, ef_search=...)`.
  FAISS exposes `efSearch` as a mutable attribute on the internal HNSW structure:
  `index.hnsw.efSearch = ef_search`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FlatComparisonConfig:
    """Matched configuration for Flat (brute-force) search."""
    dim: int = 128
    metric: str = "euclidean"
    k: int = 10

    # Notes on metric mapping:
    # - "euclidean": vecta.FlatIndex(dim, "euclidean") <-> faiss.IndexFlatL2(dim)
    #   NOTE ON SCORES: FAISS IndexFlatL2 returns SQUARED L2 distances (sum((x_i - y_i)^2)),
    #   whereas vecta returns standard Euclidean distances (sqrt(sum((x_i - y_i)^2))).
    #   Ranking order is monotonically identical; score comparison requires sqrt(faiss_dist).
    # - "dot_product": vecta.FlatIndex(dim, "dot_product") <-> faiss.IndexFlatIP(dim)
    # - "cosine": vecta supports native Cosine; FAISS expects vectors to be L2-normalized
    #   before insertion into IndexFlatIP.


@dataclass(frozen=True)
class IVFComparisonConfig:
    """Matched configuration for Inverted File (IVF) search."""
    dim: int = 128
    nlist: int = 100  # Number of coarse Voronoi clusters (num_clusters in vecta)
    metric: str = "euclidean"
    k: int = 10

    # Probe sweep matching Phase 14 recall-vs-latency curve
    nprobe_sweep: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 100]
    )
    default_nprobe: int = 10


@dataclass(frozen=True)
class HNSWComparisonConfig:
    """Matched configuration for Hierarchical Navigable Small World (HNSW) search."""
    dim: int = 128
    m: int = 16                     # Bi-directional link degree per node
    ef_construction: int = 100     # Construction beam search depth
    metric: str = "euclidean"
    k: int = 10
    seed: int = 42

    # efSearch sweep matching Phase 19 recall-vs-latency curve
    ef_search_sweep: List[int] = field(
        default_factory=lambda: [10, 20, 40, 80, 160, 320]
    )
    default_ef_search: int = 50

    # API Difference Note:
    # vecta: index.search(query, k=k, ef_search=ef)
    # FAISS: index.hnsw.efSearch = ef; index.search(query, k=k)


@dataclass(frozen=True)
class IVFPQComparisonConfig:
    """
    Matched configuration for Inverted File with Product Quantization (IVFPQ).

    Parameter Alignment:
    - nlist: Coarse IVF clusters (num_clusters in vecta).
    - m: Number of sub-vector partitions (dim must be divisible by m).
    - nbits <-> k_per_subvector:
      FAISS takes `nbits = 8`.
      vecta takes `k_per_subvector = 256` (since 2^8 = 256).
    """
    dim: int = 128
    nlist: int = 100
    m: int = 8                    # Sub-vectors (128 / 8 = 16-dim subvectors)
    nbits: int = 8                # FAISS bits per subquantizer code
    k_per_subvector: int = 256    # vecta centroids per subvector (2^8 = 256)
    metric: str = "euclidean"     # vecta IVFPQ is Euclidean-only per Phase 23
    k: int = 10

    # Probe sweep matching Phase 24 recall-vs-latency curve
    nprobe_sweep: List[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50, 100]
    )
    default_nprobe: int = 10


# Master matched parameter map documenting exact parameter equivalents
MATCHED_PARAMETER_MAP: Dict[str, Dict[str, Any]] = {
    "FlatIndex": {
        "vecta_class": "vecta.FlatIndex",
        "faiss_class": "faiss.IndexFlatL2 (or IndexFlatIP)",
        "parameters": {
            "dim": "dim",
            "metric": "metric",
        },
        "query_parameters": {
            "query": "x (np.ndarray float32)",
            "k": "k",
        },
    },
    "IVFIndex": {
        "vecta_class": "vecta.IVFIndex",
        "faiss_class": "faiss.IndexIVFFlat",
        "parameters": {
            "dim": "d",
            "num_clusters": "nlist",
            "metric": "metric_type",
        },
        "query_parameters": {
            "nprobe": "index.nprobe (attribute in FAISS; argument in vecta)",
            "k": "k",
        },
    },
    "HnswIndex": {
        "vecta_class": "vecta.HnswIndex",
        "faiss_class": "faiss.IndexHNSWFlat",
        "parameters": {
            "dim": "d",
            "m": "M",
            "ef_construction": "index.hnsw.efConstruction",
            "metric": "metric_type",
        },
        "query_parameters": {
            "ef_search": "index.hnsw.efSearch (attribute in FAISS; argument in vecta)",
            "k": "k",
        },
    },
    "IVFPQIndex": {
        "vecta_class": "vecta.IVFPQIndex",
        "faiss_class": "faiss.IndexIVFPQ",
        "parameters": {
            "dim": "d",
            "num_clusters": "nlist",
            "m": "M",
            "k_per_subvector=256": "nbits=8 (2^nbits = 256)",
            "metric": "metric_type (Euclidean-only in vecta)",
        },
        "query_parameters": {
            "nprobe": "index.nprobe (attribute in FAISS; argument in vecta)",
            "k": "k",
        },
    },
}

# Standard defaults for benchmark runs
DEFAULT_FLAT_CONFIG = FlatComparisonConfig()
DEFAULT_IVF_CONFIG = IVFComparisonConfig()
DEFAULT_HNSW_CONFIG = HNSWComparisonConfig()
DEFAULT_IVFPQ_CONFIG = IVFPQComparisonConfig()
