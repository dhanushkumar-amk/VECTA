"""
End-to-end benchmark for FlatIndex on standard SIFT dataset.

Measures:
- Index build time and ingestion rate (vectors/sec)
- Total search time and Queries Per Second (QPS)
- Query latency percentiles (mean, p50, p95, p99)
- Recall@k against standard SIFT ground truth
- JSON artifact persistence in benchmarks/results/
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


def run_flat_index_benchmark(
    k: int = 10,
    dataset_name: str = "siftsmall",
    save_json: bool = True,
) -> dict:
    """
    Run full benchmark suite on FlatIndex.
    """
    print("=" * 65)
    print(f" VECTA BENCHMARK: FlatIndex (Euclidean / L2)")
    print(f" Dataset: {dataset_name} | Target k: {k}")
    print("=" * 65)

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

    # 2. Build Index
    print("\n[2/4] Building FlatIndex...")
    index = vecta.FlatIndex(dim=dim, metric="euclidean")
    ids = list(range(num_base))
    vectors_list = base.tolist()

    with Timer("Build Index") as t_build:
        index.add_batch(ids, vectors_list)

    build_time_sec = t_build.elapsed_sec
    build_rate = num_base / build_time_sec if build_time_sec > 0 else 0.0
    print(f"  Build time:     {t_build.elapsed_ms:.2f} ms ({build_time_sec:.4f} s)")
    print(f"  Ingestion rate: {build_rate:,.1f} vectors/sec")
    assert index.len() == num_base

    # 3. Search Queries
    print(f"\n[3/4] Searching {num_query} queries with k={k}...")
    predicted_ids = []
    latencies_ms = []

    # Prepare query vectors as lists for FFI
    query_lists = query.tolist()

    with Timer("Total Search Time") as t_search:
        for q_vec in query_lists:
            t0 = time.perf_counter()
            results = index.search(q_vec, k=k)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            predicted_ids.append([item[0] for item in results])

    search_time_sec = t_search.elapsed_sec
    qps = compute_qps(search_time_sec, num_query)
    mean_lat = float(np.mean(latencies_ms))
    p50_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))

    print(f"  Total search:   {t_search.elapsed_ms:.2f} ms ({search_time_sec:.4f} s)")
    print(f"  Throughput:     {qps:,.1f} QPS")
    print(f"  Latency (mean): {mean_lat:.3f} ms")
    print(f"  Latency (p50):  {p50_lat:.3f} ms")
    print(f"  Latency (p95):  {p95_lat:.3f} ms")
    print(f"  Latency (p99):  {p99_lat:.3f} ms")

    # 4. Compute Recall
    print(f"\n[4/4] Evaluating Recall against SIFT ground truth...")
    rec_k = recall_at_k(predicted_ids, gt.tolist(), k=k)
    rec_1 = recall_at_k(predicted_ids, gt.tolist(), k=1) if k >= 1 else None

    print(f"  Recall@{k}:      {rec_k * 100:.2f}%")
    if rec_1 is not None:
        print(f"  Recall@1:       {rec_1 * 100:.2f}%")

    print("\n" + "=" * 65)
    print(" SUMMARY")
    print("=" * 65)
    print(f"  Dataset:         {dataset_name} (N={num_base:,}, D={dim})")
    print(f"  Metric:          Euclidean (L2)")
    print(f"  Build Time:      {build_time_sec:.4f} s ({build_rate:,.1f} vec/s)")
    print(f"  Search Time:     {search_time_sec:.4f} s for {num_query} queries")
    print(f"  Throughput:      {qps:,.1f} QPS")
    print(f"  Latency (p50):   {p50_lat:.3f} ms")
    print(f"  Recall@{k}:       {rec_k * 100:.2f}%")
    print("=" * 65)

    benchmark_data = {
        "dataset": dataset_name,
        "num_vectors": num_base,
        "dimension": dim,
        "num_queries": num_query,
        "k": k,
        "metric": "euclidean",
        "build_time_sec": build_time_sec,
        "build_rate_vec_per_sec": build_rate,
        "search_time_sec": search_time_sec,
        "qps": qps,
        "latency_ms": {
            "mean": mean_lat,
            "p50": p50_lat,
            "p95": p95_lat,
            "p99": p99_lat,
        },
        "recall": {
            f"recall@{k}": rec_k,
        },
    }
    if rec_1 is not None:
        benchmark_data["recall"]["recall@1"] = rec_1

    if save_json:
        saved_path = save_benchmark_result("flat_index", benchmark_data)
        print(f"\nResults saved to: {saved_path}")

    return benchmark_data


def main():
    parser = argparse.ArgumentParser(description="Vecta FlatIndex Benchmark")
    parser.add_argument("--k", type=int, default=10, help="Top-k nearest neighbors (default: 10)")
    parser.add_argument("--no-save", action="store_true", help="Do not save results to JSON")
    args = parser.parse_args()

    run_flat_index_benchmark(k=args.k, save_json=not args.no_save)


if __name__ == "__main__":
    main()
