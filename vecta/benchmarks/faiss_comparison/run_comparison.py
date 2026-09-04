"""
Head-to-head comparison benchmark runner between vecta and FAISS.

Phase 35: Harness skeleton and verification stub.
Full comparative evaluation logic is implemented in subsequent phases (36-50).

Usage:
    python benchmarks/faiss_comparison/run_comparison.py --index-type all --threads 1
    python benchmarks/faiss_comparison/run_comparison.py --index-type flat --dry-run
"""

import argparse
import os
import sys
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import vecta
    import faiss
except ImportError as e:
    print(f"Error importing required benchmark packages: {e}")
    sys.exit(1)

from benchmarks.faiss_comparison.config import (
    MATCHED_PARAMETER_MAP,
    DEFAULT_FLAT_CONFIG,
    DEFAULT_IVF_CONFIG,
    DEFAULT_HNSW_CONFIG,
    DEFAULT_IVFPQ_CONFIG,
)
from benchmarks.faiss_comparison.faiss_wrappers import (
    set_faiss_threads,
    get_faiss_threads,
    build_faiss_flat,
    build_faiss_ivf,
    build_faiss_hnsw,
    build_faiss_ivfpq,
)


def verify_side_by_side_instantiation(dim: int = 128) -> bool:
    """
    Verify that all four index types in both vecta and FAISS can be instantiated
    in the same Python process without symbol collisions or memory conflicts.
    """
    print("\n[Phase 35 Verification] Instantiating matched index pairs...")

    # 1. Flat Index
    v_flat = vecta.FlatIndex(dim, "euclidean")
    f_flat = build_faiss_flat(dim, "euclidean")
    assert v_flat.dim() == f_flat.d == dim
    print(f"  [1/4] Flat:   vecta.FlatIndex (dim={v_flat.dim()}) <-> faiss.{type(f_flat).__name__} (d={f_flat.d})")

    # 2. IVF Index
    v_ivf = vecta.IVFIndex(dim, num_clusters=100, metric="euclidean")
    f_ivf = build_faiss_ivf(dim, nlist=100, metric="euclidean")
    assert v_ivf.dim() == f_ivf.d == dim
    assert f_ivf.nlist == 100
    print(f"  [2/4] IVF:    vecta.IVFIndex (nlist=100) <-> faiss.{type(f_ivf).__name__} (nlist={f_ivf.nlist})")

    # 3. HNSW Index
    v_hnsw = vecta.HnswIndex(dim, m=16, ef_construction=100, ef_search=50, metric="euclidean", seed=42)
    f_hnsw = build_faiss_hnsw(dim, m=16, ef_construction=100, metric="euclidean")
    assert v_hnsw.dim() == f_hnsw.d == dim
    assert f_hnsw.hnsw.efConstruction == 100
    print(f"  [3/4] HNSW:   vecta.HnswIndex (M=16, efC=100) <-> faiss.{type(f_hnsw).__name__} (M={f_hnsw.hnsw.max_level})")

    # 4. IVFPQ Index
    v_ivfpq = vecta.IVFPQIndex(dim, num_clusters=100, m=8, k_per_subvector=256)
    f_ivfpq = build_faiss_ivfpq(dim, nlist=100, m=8, nbits=8, metric="euclidean")
    assert v_ivfpq.dim() == f_ivfpq.d == dim
    assert f_ivfpq.nlist == 100
    print(f"  [4/4] IVFPQ:  vecta.IVFPQIndex (k_sub=256) <-> faiss.{type(f_ivfpq).__name__} (nbits=8)")

    print("All four matched index pairs instantiated cleanly in one process!\n")
    return True


def run_comparison_stub(
    index_type: str = "all",
    threads: int = 1,
    k: int = 10,
    dataset: str = "siftsmall",
    dry_run: bool = False,
) -> None:
    """
    Comparison harness runner stub.

    Sets threading parameters and executes validation check.
    Full benchmarking loops across Phase 36+ will populate the respective sections.
    """
    print("=" * 70)
    print(" VECTA vs. FAISS COMPARISON HARNESS (PHASE 35)")
    print(f" Target Index: {index_type.upper()} | Threads: {threads} | Target k: {k}")
    print("=" * 70)

    # Set FAISS threading mode
    set_faiss_threads(threads)
    actual_threads = get_faiss_threads()
    print(f"FAISS OpenMP Threads set to: {actual_threads}")

    # Verify both libraries coexist side-by-side
    verify_side_by_side_instantiation(dim=128)

    if dry_run:
        print("Dry-run requested: parameter validation and instantiation successful.")
        return

    # TODO (Phase 36+): Implement full head-to-head FlatIndex evaluation
    if index_type in ("all", "flat"):
        print("[TODO Phase 36] FlatIndex vs faiss.IndexFlatL2 benchmark loop scheduled.")

    # TODO (Phase 37+): Implement full head-to-head IVFIndex evaluation
    if index_type in ("all", "ivf"):
        print("[TODO Phase 37] IVFIndex vs faiss.IndexIVFFlat benchmark loop scheduled.")

    # TODO (Phase 38+): Implement full head-to-head HnswIndex evaluation
    if index_type in ("all", "hnsw"):
        print("[TODO Phase 38] HnswIndex vs faiss.IndexHNSWFlat benchmark loop scheduled.")

    # TODO (Phase 39+): Implement full head-to-head IVFPQIndex evaluation
    if index_type in ("all", "ivfpq"):
        print("[TODO Phase 39] IVFPQIndex vs faiss.IndexIVFPQ benchmark loop scheduled.")


def main():
    parser = argparse.ArgumentParser(
        description="Vecta vs. FAISS Head-to-Head Comparison Harness"
    )
    parser.add_argument(
        "--index-type",
        choices=["all", "flat", "ivf", "hnsw", "ivfpq"],
        default="all",
        help="Index type to compare (default: all)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Thread count for FAISS OpenMP (default: 1 for single-threaded parity)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve (default: 10)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="siftsmall",
        help="Dataset name (default: siftsmall)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate and validate configs without running benchmark loops",
    )
    args = parser.parse_args()

    run_comparison_stub(
        index_type=args.index_type,
        threads=args.threads,
        k=args.k,
        dataset=args.dataset,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
