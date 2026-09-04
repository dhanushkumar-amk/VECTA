"""
End-to-end benchmark for IVFPQIndex on standard SIFT dataset (Phase 24).

Measures:
- IVFPQIndex build & training time (coarse k-means + PQ codebook training + vector compression)
- Memory footprint in bytes (and compression ratio vs full-precision IVFIndex)
- For nprobe in [1, 5, 10, 20, 50]:
  - Query latency percentiles (mean, p50, p95)
  - Search throughput (QPS)
  - Recall@10 against exact ground truth
- Direct side-by-side comparison table vs IVFIndex (loaded from previous benchmark artifacts)
- JSON artifact persistence in benchmarks/results/
"""

import argparse
import glob
import json
import os
import sys
import time
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import vecta
from benchmarks.datasets.download_sift1m import load_siftsmall
from benchmarks.utils.recall import recall_at_k
from benchmarks.utils.timer import Timer, compute_qps, save_benchmark_result


def load_latest_ivf_result(results_dir: str) -> dict:
    """Load the latest saved IVFIndex benchmark result JSON, if available."""
    pattern = os.path.join(results_dir, "ivf_index_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    try:
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not load {files[0]}: {e}")
        return None


def run_ivf_pq_index_benchmark(
    k_clusters: int = 100,
    m: int = 8,
    k_per_subvector: int = 256,
    max_iterations: int = 25,
    top_k: int = 10,
    nprobe_list: list = None,
    dataset_name: str = "siftsmall",
    save_json: bool = True,
) -> dict:
    """
    Run full speed vs recall vs memory benchmark on IVFPQIndex across varying nprobe values.
    """
    if nprobe_list is None:
        nprobe_list = [1, 5, 10, 20, 50]

    print("=" * 80)
    print(" VECTA BENCHMARK: IVFPQIndex (Inverted File with Product Quantization)")
    print(f" Dataset: {dataset_name} | Clusters (k): {k_clusters} | Subquantizers (m): {m} | k_per_sub: {k_per_subvector}")
    print("=" * 80)

    # 1. Load Dataset
    print("\n[1/5] Loading dataset...")
    with Timer("Load Dataset") as t_load:
        base, query, gt = load_siftsmall()
    num_base, dim = base.shape
    num_query = query.shape[0]
    print(f"  Base vectors:   {num_base:,} vectors (dim={dim})")
    print(f"  Query vectors:  {num_query:,} vectors")
    print(f"  Ground truth:   {gt.shape[0]:,} queries with {gt.shape[1]} nearest neighbors")
    print(f"  Load time:      {t_load.elapsed_ms:.2f} ms")

    # 2. Build and Train IVFPQ Index
    print(f"\n[2/5] Training and building IVFPQIndex (k={k_clusters}, m={m}, k_sub={k_per_subvector})...")
    ivf_pq = vecta.IVFPQIndex(
        dim=dim,
        num_clusters=k_clusters,
        m=m,
        k_per_subvector=k_per_subvector,
        max_iterations=max_iterations,
    )
    ids = list(range(num_base))
    vectors_list = base.tolist()

    with Timer("IVFPQ Training") as t_train:
        # Train coarse centroids and PQ codebooks
        ivf_pq.train(vectors_list, ivf_seed=42, pq_seed=42)
    train_time_sec = t_train.elapsed_sec
    print(f"  Training time:  {t_train.elapsed_ms:.2f} ms ({train_time_sec:.4f} s)")

    with Timer("IVFPQ Ingestion (PQ Encoding)") as t_add:
        ivf_pq.add_batch(ids, vectors_list)
    add_time_sec = t_add.elapsed_sec
    total_build_sec = train_time_sec + add_time_sec
    print(f"  Ingestion time: {t_add.elapsed_ms:.2f} ms ({add_time_sec:.4f} s)")
    print(f"  Total build:    {total_build_sec * 1000.0:.2f} ms ({total_build_sec:.4f} s)")
    assert len(ivf_pq) == num_base

    # 3. Memory Footprint Diagnostics
    print("\n[3/5] Memory footprint analysis...")
    ivfpq_mem_bytes = ivf_pq.memory_footprint_bytes()
    ivfpq_vector_bytes = num_base * m
    full_precision_vector_bytes = num_base * dim * 4
    full_precision_total_bytes = full_precision_vector_bytes + (k_clusters * dim * 4)

    vector_compression = full_precision_vector_bytes / max(ivfpq_vector_bytes, 1)
    total_compression = full_precision_total_bytes / max(ivfpq_mem_bytes, 1)

    print(f"  Full-precision vector storage (IVF): {full_precision_vector_bytes:,} bytes ({full_precision_vector_bytes / 1024.0:.1f} KB)")
    print(f"  Compressed vector storage (IVFPQ):   {ivfpq_vector_bytes:,} bytes ({ivfpq_vector_bytes / 1024.0:.1f} KB)")
    print(f"  Vector data compression ratio:       {vector_compression:.1f}x")
    print(f"  Total IVFPQ footprint:               {ivfpq_mem_bytes:,} bytes ({ivfpq_mem_bytes / 1024.0:.1f} KB)")
    print(f"  Total index compression ratio:       {total_compression:.1f}x (including codebooks & centroids)")

    # 4. Benchmark Across nprobe values
    print(f"\n[4/5] Benchmarking search throughput and recall across nprobe values...")
    query_lists = query.tolist()
    gt_list = gt.tolist()

    nprobe_results = []

    for nprobe in nprobe_list:
        predicted_ids = []
        latencies_ms = []

        with Timer(f"Search nprobe={nprobe}") as t_search:
            for q_vec in query_lists:
                t0 = time.perf_counter()
                results = ivf_pq.search(q_vec, k=top_k, nprobe=nprobe)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
                predicted_ids.append([item[0] for item in results])

        search_sec = t_search.elapsed_sec
        qps = compute_qps(search_sec, num_query)
        recall = recall_at_k(predicted_ids, gt_list, k=top_k)
        mean_lat = float(np.mean(latencies_ms))
        p50_lat = float(np.median(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))

        nprobe_results.append({
            "nprobe": nprobe,
            "qps": qps,
            "recall_at_10": recall,
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "search_time_sec": search_sec,
            "memory_footprint_bytes": ivfpq_mem_bytes,
        })

    # Summary table for IVFPQIndex
    print("\nIVFPQIndex Results Summary Table:")
    print("-" * 80)
    print(f"{'nprobe':>8} | {'Recall@10':>10} | {'Throughput (QPS)':>18} | {'Latency Mean':>14} | {'Footprint':>12}")
    print("-" * 80)
    for r in nprobe_results:
        print(
            f"{r['nprobe']:>8} | "
            f"{r['recall_at_10'] * 100.0:>9.2f}% | "
            f"{r['qps']:>18,.1f} | "
            f"{r['mean_latency_ms']:>11.3f} ms | "
            f"{r['memory_footprint_bytes'] / 1024.0:>9.1f} KB"
        )
    print("-" * 80)

    # 5. Direct Comparison Table: IVFIndex vs IVFPQIndex
    print("\n[5/5] Direct Comparison: IVFIndex (Full Precision) vs IVFPQIndex (PQ Compressed):")
    results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    ivf_prev = load_latest_ivf_result(results_dir)

    print("=" * 100)
    print(f"{'nprobe':>6} | {'IVF Recall':>11} | {'IVFPQ Recall':>13} | {'IVF QPS':>12} | {'IVFPQ QPS':>12} | {'IVF Mem':>10} | {'IVFPQ Mem':>10} | {'Mem Save':>9}")
    print("=" * 100)

    ivf_nprobe_map = {}
    if ivf_prev and "nprobe_benchmarks" in ivf_prev:
        for item in ivf_prev["nprobe_benchmarks"]:
            ivf_nprobe_map[item["nprobe"]] = item

    for r in nprobe_results:
        np_val = r["nprobe"]
        ivfpq_rec = f"{r['recall_at_10'] * 100.0:.1f}%"
        ivfpq_qps = f"{r['qps']:,.0f}"
        ivfpq_mem_kb = f"{ivfpq_mem_bytes / 1024.0:.1f} KB"

        if np_val in ivf_nprobe_map:
            ivf_item = ivf_nprobe_map[np_val]
            ivf_rec = f"{ivf_item['recall_at_10'] * 100.0:.1f}%"
            ivf_qps = f"{ivf_item['qps']:,.0f}"
        else:
            ivf_rec = "N/A"
            ivf_qps = "N/A"

        ivf_mem_kb = f"{full_precision_total_bytes / 1024.0:.1f} KB"
        save_ratio = f"{total_compression:.1f}x"

        print(
            f"{np_val:>6} | "
            f"{ivf_rec:>11} | "
            f"{ivfpq_rec:>13} | "
            f"{ivf_qps:>12} | "
            f"{ivfpq_qps:>12} | "
            f"{ivf_mem_kb:>10} | "
            f"{ivfpq_mem_kb:>10} | "
            f"{save_ratio:>9}"
        )
    print("=" * 100)

    benchmark_data = {
        "index_type": "IVFPQIndex",
        "dataset": dataset_name,
        "num_vectors": num_base,
        "dim": dim,
        "num_queries": num_query,
        "k_clusters": k_clusters,
        "m": m,
        "k_per_subvector": k_per_subvector,
        "top_k": top_k,
        "train_time_sec": train_time_sec,
        "add_time_sec": add_time_sec,
        "total_build_time_sec": total_build_sec,
        "memory_footprint_bytes": ivfpq_mem_bytes,
        "vector_compression_ratio": vector_compression,
        "total_compression_ratio": total_compression,
        "nprobe_benchmarks": nprobe_results,
    }

    if save_json:
        filepath = save_benchmark_result(
            benchmark_name=f"ivf_pq_index_{dataset_name}",
            metrics=benchmark_data,
            results_dir=results_dir,
        )
        print(f"\nBenchmark artifact saved to: {filepath}")

    return benchmark_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vecta IVFPQIndex Benchmark")
    parser.add_argument("--k-clusters", type=int, default=100, help="Number of IVF clusters (centroids)")
    parser.add_argument("--m", type=int, default=8, help="Number of subquantizers")
    parser.add_argument("--k-per-subvector", type=int, default=256, help="Centroids per subquantizer")
    parser.add_argument("--max-iterations", type=int, default=25, help="Lloyd iterations")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k nearest neighbors to retrieve")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON results")
    args = parser.parse_args()

    run_ivf_pq_index_benchmark(
        k_clusters=args.k_clusters,
        m=args.m,
        k_per_subvector=args.k_per_subvector,
        max_iterations=args.max_iterations,
        top_k=args.top_k,
        save_json=not args.no_save,
    )
