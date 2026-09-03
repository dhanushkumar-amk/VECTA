"""
End-to-end benchmark for HnswIndex on standard SIFT dataset (Phase 19).

Measures:
- HnswIndex build time (sequential graph insertion with M=16, ef_construction=200)
- Layer distribution histogram diagnostics
- For ef_search in [10, 50, 100, 200, 500]:
  - Query latency percentiles (mean, p50, p95, p99)
  - Search throughput (QPS)
  - Recall@10 against exact ground truth
- Table printout and JSON artifact persistence in benchmarks/results/
"""

import argparse
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


def run_hnsw_index_benchmark(
    m: int = 16,
    ef_construction: int = 200,
    top_k: int = 10,
    ef_search_list: list[int] = None,
    dataset_name: str = "siftsmall",
    save_json: bool = True,
) -> dict:
    """
    Run full speed vs recall benchmark on HnswIndex across varying ef_search values.
    """
    if ef_search_list is None:
        ef_search_list = [10, 50, 100, 200, 500]

    print("=" * 70)
    print(" VECTA BENCHMARK: HnswIndex (Hierarchical Navigable Small World)")
    print(f" Dataset: {dataset_name} | M: {m} | ef_c: {ef_construction} | Target Top-k: {top_k}")
    print("=" * 70)

    # 1. Load Dataset
    print("\n[1/4] Loading dataset...")
    with Timer("Load Dataset") as t_load:
        base, query, gt = load_siftsmall()
    num_base, dim = base.shape
    num_query = query.shape[0]
    print(f"  Base vectors:   {num_base:,} vectors (dim={dim})")
    print(f"  Query vectors:  {num_query:,} vectors")
    print(f"  Ground truth:   {gt.shape[0]:,} queries with {gt.shape[1]} nearest neighbors")
    print(f"  Load time:      {t_load.elapsed_ms:.2f} ms")

    # 2. Build HNSW Index
    print(f"\n[2/4] Building HnswIndex (M={m}, ef_construction={ef_construction})...")
    hnsw_index = vecta.HnswIndex(
        dim=dim,
        metric="euclidean",
        m=m,
        ef_construction=ef_construction,
        ef_search=50,
        seed=42,
    )
    ids = list(range(num_base))
    vectors_list = base.tolist()

    with Timer("HNSW Ingestion") as t_build:
        hnsw_index.add_batch(ids, vectors_list)
    build_time_sec = t_build.elapsed_sec
    print(f"  Build time:     {t_build.elapsed_ms:.2f} ms ({build_time_sec:.4f} s)")
    assert len(hnsw_index) == num_base

    # Layer distribution diagnostics
    distribution = hnsw_index.max_layer_distribution()
    print("  Layer distribution:")
    for l in sorted(distribution.keys()):
        cnt = distribution[l]
        pct = (cnt / num_base) * 100.0
        print(f"    Layer {l}: {cnt:>5} nodes ({pct:>5.1f}%)")

    # 3. Benchmark Across ef_search values
    print(f"\n[3/4] Benchmarking search throughput and recall across ef_search values...")
    query_lists = query.tolist()
    gt_list = gt.tolist()

    ef_results = []

    for ef_search in ef_search_list:
        predicted_ids = []
        latencies_ms = []

        with Timer(f"Search ef_search={ef_search}") as t_search:
            for q_vec in query_lists:
                t0 = time.perf_counter()
                results = hnsw_index.search(q_vec, k=top_k, ef_search=ef_search)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
                predicted_ids.append([item[0] for item in results])

        qps = compute_qps(t_search.elapsed_sec, num_query)
        recall = recall_at_k(predicted_ids, gt_list, k=top_k)

        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        p99 = float(np.percentile(latencies_ms, 99))
        mean_lat = float(np.mean(latencies_ms))

        ef_results.append({
            "ef_search": ef_search,
            "qps": round(qps, 2),
            "recall_at_10": round(recall, 4),
            "latency_mean_ms": round(mean_lat, 4),
            "latency_p50_ms": round(p50, 4),
            "latency_p95_ms": round(p95, 4),
            "latency_p99_ms": round(p99, 4),
            "total_search_sec": round(t_search.elapsed_sec, 4),
        })

    # 4. Print Summary Table
    print("\n" + "=" * 80)
    print(" HnswIndex BENCHMARK RESULTS (M=16, ef_construction=200, N=10,000 SIFT)")
    print("=" * 80)
    print(f"{'ef_search':>10} | {'QPS':>10} | {'Recall@10':>10} | {'Mean (ms)':>10} | {'p50 (ms)':>10} | {'p95 (ms)':>10}")
    print("-" * 80)
    for r in ef_results:
        print(
            f"{r['ef_search']:>10} | "
            f"{r['qps']:>10.1f} | "
            f"{r['recall_at_10'] * 100:>9.2f}% | "
            f"{r['latency_mean_ms']:>10.4f} | "
            f"{r['latency_p50_ms']:>10.4f} | "
            f"{r['latency_p95_ms']:>10.4f}"
        )
    print("=" * 80)

    # 5. Persist JSON artifact
    benchmark_payload = {
        "benchmark": "HnswIndex",
        "dataset": dataset_name,
        "num_vectors": num_base,
        "dimension": dim,
        "metric": "euclidean",
        "m": m,
        "ef_construction": ef_construction,
        "build_time_sec": round(build_time_sec, 4),
        "layer_distribution": distribution,
        "top_k": top_k,
        "results": ef_results,
    }

    if save_json:
        json_path = save_benchmark_result("hnsw_index", benchmark_payload)
        print(f"\nSaved benchmark payload to: {json_path}")

    return benchmark_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vecta HnswIndex Benchmark on SIFT1M")
    parser.add_argument("--m", type=int, default=16, help="Max neighbors per node")
    parser.add_argument("--ef-construction", type=int, default=200, help="ef_construction")
    parser.add_argument("--top-k", type=int, default=10, help="Target top-k neighbors")
    parser.add_argument("--no-save", action="store_true", help="Disable saving JSON results")
    args = parser.parse_args()

    run_hnsw_index_benchmark(
        m=args.m,
        ef_construction=args.ef_construction,
        top_k=args.top_k,
        save_json=not args.no_save,
    )
