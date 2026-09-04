"""
Tests for FAISS comparison benchmarking harness (Phase 35).

Covers:
1. FAISS installation and trivial smoke test.
2. Factory functions in faiss_wrappers.py building correct types and attributes.
3. Coexistence of vecta and faiss in the same Python process.
4. run_comparison.py CLI and execution stub.
"""

import subprocess
import sys
import numpy as np
import pytest

import faiss
import vecta

from benchmarks.faiss_comparison.config import (
    MATCHED_PARAMETER_MAP,
    DEFAULT_FLAT_CONFIG,
    DEFAULT_IVF_CONFIG,
    DEFAULT_HNSW_CONFIG,
    DEFAULT_IVFPQ_CONFIG,
)
from benchmarks.faiss_comparison.faiss_wrappers import (
    build_faiss_flat,
    build_faiss_ivf,
    build_faiss_hnsw,
    build_faiss_ivfpq,
    set_faiss_threads,
    get_faiss_threads,
    set_search_params,
)
from benchmarks.faiss_comparison.run_comparison import (
    verify_side_by_side_instantiation,
    run_comparison_stub,
)


class TestFaissInstallation:
    """Test 1: FAISS CPU package installation and basic functionality."""

    def test_faiss_import_and_version(self):
        assert hasattr(faiss, "__version__")
        assert len(faiss.__version__) > 0

    def test_trivial_flat_smoke(self):
        dim = 4
        index = faiss.IndexFlatL2(dim)
        data = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
        )
        index.add(data)
        assert index.ntotal == 2

        distances, indices = index.search(data[:1], k=1)
        assert indices.shape == (1, 1)
        assert indices[0, 0] == 0
        assert abs(distances[0, 0] - 0.0) < 1e-6


class TestFaissWrappers:
    """Test 2: Factory functions construct expected index types and parameters."""

    def test_build_faiss_flat(self):
        idx_l2 = build_faiss_flat(dim=64, metric="euclidean")
        assert isinstance(idx_l2, faiss.IndexFlatL2)
        assert idx_l2.d == 64
        assert idx_l2.metric_type == faiss.METRIC_L2

        idx_ip = build_faiss_flat(dim=32, metric="dot_product")
        assert isinstance(idx_ip, faiss.IndexFlatIP)
        assert idx_ip.d == 32
        assert idx_ip.metric_type == faiss.METRIC_INNER_PRODUCT

    def test_build_faiss_ivf(self):
        idx = build_faiss_ivf(dim=128, nlist=50, metric="euclidean")
        assert isinstance(idx, faiss.IndexIVFFlat)
        assert idx.d == 128
        assert idx.nlist == 50
        assert not idx.is_trained

        set_search_params(idx, nprobe=8)
        assert idx.nprobe == 8

    def test_build_faiss_hnsw(self):
        idx = build_faiss_hnsw(dim=128, m=16, ef_construction=120, metric="euclidean")
        assert isinstance(idx, faiss.IndexHNSWFlat)
        assert idx.d == 128
        assert idx.hnsw.efConstruction == 120

        set_search_params(idx, ef_search=60)
        assert idx.hnsw.efSearch == 60

    def test_build_faiss_ivfpq(self):
        idx = build_faiss_ivfpq(dim=128, nlist=100, m=8, nbits=8, metric="euclidean")
        assert isinstance(idx, faiss.IndexIVFPQ)
        assert idx.d == 128
        assert idx.nlist == 100
        assert not idx.is_trained

        set_search_params(idx, nprobe=12)
        assert idx.nprobe == 12

    def test_threading_control(self):
        set_faiss_threads(1)
        assert get_faiss_threads() == 1
        set_faiss_threads(4)
        assert get_faiss_threads() == 4
        # Restore single-thread baseline
        set_faiss_threads(1)


class TestCoexistenceVectaAndFaiss:
    """Test 3: Vecta and FAISS coexisting in the same Python process."""

    def test_side_by_side_instantiation(self):
        assert verify_side_by_side_instantiation(dim=64)

    def test_cross_check_search_on_same_data(self):
        dim = 8
        n_vectors = 50
        np.random.seed(123)

        # Generate vectors
        data = np.random.randn(n_vectors, dim).astype(np.float32)
        query = np.random.randn(1, dim).astype(np.float32)

        # Build vecta index
        v_idx = vecta.FlatIndex(dim, "euclidean")
        v_idx.add_batch(list(range(n_vectors)), data.tolist())

        # Build faiss index
        f_idx = build_faiss_flat(dim, "euclidean")
        f_idx.add(data)

        # Search top 5
        v_res = v_idx.search(query[0].tolist(), k=5)
        f_dists, f_ids = f_idx.search(query, k=5)

        v_top_ids = [item[0] for item in v_res]
        f_top_ids = f_ids[0].tolist()

        assert (
            v_top_ids == f_top_ids
        ), f"Vecta {v_top_ids} vs FAISS {f_top_ids} top-k candidate mismatch!"

        # FAISS IndexFlatL2 returns squared Euclidean distance (sum((x_i - y_i)^2)),
        # whereas vecta returns standard Euclidean distance (sqrt(sum((x_i - y_i)^2))).
        # Candidate IDs must match 100%, and sqrt(faiss_dist) must match vecta score.
        for rank, ((v_id, v_score), f_dist) in enumerate(
            zip(v_res, f_dists[0].tolist())
        ):
            assert (
                abs(v_score - np.sqrt(f_dist)) < 1e-4
            ), f"Score mismatch at rank {rank}: vecta={v_score}, faiss_sqrt={np.sqrt(f_dist)}"

    def test_cross_check_dot_product_search(self):
        dim = 8
        n_vectors = 50
        np.random.seed(456)

        data = np.random.randn(n_vectors, dim).astype(np.float32)
        query = np.random.randn(1, dim).astype(np.float32)

        v_idx = vecta.FlatIndex(dim, "dot_product")
        v_idx.add_batch(list(range(n_vectors)), data.tolist())

        f_idx = build_faiss_flat(dim, "dot_product")
        f_idx.add(data)

        v_res = v_idx.search(query[0].tolist(), k=5)
        f_dists, f_ids = f_idx.search(query, k=5)

        v_top_ids = [item[0] for item in v_res]
        f_top_ids = f_ids[0].tolist()

        assert v_top_ids == f_top_ids

        for rank, ((v_id, v_score), f_dist) in enumerate(
            zip(v_res, f_dists[0].tolist())
        ):
            assert (
                abs(v_score - f_dist) < 1e-4
            ), f"Score mismatch at rank {rank}: vecta={v_score}, faiss={f_dist}"


class TestRunComparisonStub:
    """Test 4: run_comparison.py stub runs without errors."""

    def test_run_comparison_stub_dry_run(self):
        run_comparison_stub(index_type="all", threads=1, k=10, dry_run=True)

    def test_cli_execution(self):
        result = subprocess.run(
            [
                sys.executable,
                "benchmarks/faiss_comparison/run_comparison.py",
                "--index-type",
                "flat",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "VECTA vs. FAISS COMPARISON HARNESS" in result.stdout
        assert "Dry-run requested" in result.stdout
