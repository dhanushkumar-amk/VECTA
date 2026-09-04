"""
End-to-end benchmark for ShardedFlatIndex on standard SIFT dataset (Phase 34).

Measures:
- Ingestion rate and shard distribution across N shards
- Sequential search (parallel=False) throughput (QPS) and latency percentiles
- Parallel fan-out search (parallel=True) throughput (QPS) and latency percentiles
- Parallel speedup factor
- Recall@10 sanity check against SIFT ground truth (exact search = ~100% recall)
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


def run_sharded_index_benchmark(
    k: int = 10,
    num_shards: int = 4,
    dataset_name: str = "siftsmall",
    save_json: bool = True,
) -> dict:
    """
    Run full benchmark suite on ShardedFlatIndex comparing sequential vs parallel search.
    """
    print("=" * 70)
    print(f" VECTA BENCHMARK: ShardedFlatIndex (Euclidean / L2)")
    print(f" Dataset: {dataset_name} | Shards: {num_shards} | Target k: {k}")
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

    # 2. Build Sharded Index
    print(f"\n[2/4] Building ShardedFlatIndex with {num_shards} shards...")
    index = vecta.ShardedFlatIndex(dim=dim, num_shards=num_shards, metric="euclidean")
    ids = list(range(num_base))
    vectors_list = base.tolist()

    with Timer("Build Index") as t_build:
        index.add_batch(ids, vectors_list)

    build_time_sec = t_build.elapsed_sec
    build_rate = num_base / build_time_sec if build_time_sec > 0 else 0.0
    shard_sizes = index.shard_sizes()
    expected_avg = num_base / num_shards

    print(f"  Build time:     {t_build.elapsed_ms:.2f} ms ({build_time_sec:.4f} s)")
    print(f"  Ingestion rate: {build_rate:,.1f} vectors/sec")
    print(f"  Total count:    {len(index):,} vectors")
    print("  Shard sizes:   ", shard_sizes)
    for s, size in enumerate(shard_sizes):
        dev = (size - expected_avg) / expected_avg * 100.0
        print(f"    Shard {s}: {size:,} ({dev:+.2f}% from mean {expected_avg:.0f})")

    query_lists = query.tolist()

    # 3. Sequential Search (parallel=False)
    print(f"\n[3/4] Searching {num_query} queries with k={k} (Sequential: parallel=False)...")
    seq_predicted_ids = []
    seq_latencies_ms = []

    with Timer("Sequential Search") as t_seq_search:
        for q_vec in query_lists:
            t0 = time.perf_counter()
            results = index.search(q_vec, k=k, parallel=False)
            t1 = time.perf_counter()
            seq_latencies_ms.append((t1 - t0) * 1000.0)
            seq_predicted_ids.append([item[0] for item in results])

    seq_time_sec = t_seq_search.elapsed_sec
    seq_qps = compute_qps(seq_time_sec, num_query)
    seq_mean_lat = float(np.mean(seq_latencies_ms))
    seq_p50_lat = float(np.median(seq_latencies_ms))
    seq_p95_lat = float(np.percentile(seq_latencies_ms, 95))
    seq_p99_lat = float(np.percentile(seq_latencies_ms, 99))
    seq_rec_k = recall_at_k(seq_predicted_ids, gt.tolist(), k=k)

    print(f"  Total search:   {t_seq_search.elapsed_ms:.2f} ms ({seq_time_sec:.4f} s)")
    print(f"  Throughput:     {seq_qps:,.1f} QPS")
    print(f"  Latency (mean): {seq_mean_lat:.3f} ms")
    print(f"  Latency (p50):  {seq_p50_lat:.3f} ms")
    print(f"  Recall@{k}:      {seq_rec_k * 100:.2f}%")

    # 4. Parallel Fan-Out Search (parallel=True)
    print(f"\n[4/4] Searching {num_query} queries with k={k} (Parallel: parallel=True)...")
    par_predicted_ids = []
    par_latencies_ms = []

    with Timer("Parallel Search") as t_par_search:
        for q_vec in query_lists:
            t0 = time.perf_counter()
            results = index.search(q_vec, k=k, parallel=True)
            t1 = time.perf_counter()
            par_latencies_ms.append((t1 - t0) * 1000.0)
            par_predicted_ids.append([item[0] for item in results])

    par_time_sec = t_par_search.elapsed_sec
    par_qps = compute_qps(par_time_sec, num_query)
    par_mean_lat = float(np.mean(par_latencies_ms))
    par_p50_lat = float(np.median(par_latencies_ms))
    par_p95_lat = float(np.percentile(par_latencies_ms, 95))
    par_p99_lat = float(np.percentile(par_latencies_ms, 99))
    par_rec_k = recall_at_k(par_predicted_ids, gt.tolist(), k=k)

    print(f"  Total search:   {t_par_search.elapsed_ms:.2f} ms ({par_time_sec:.4f} s)")
    print(f"  Throughput:     {par_qps:,.1f} QPS")
    print(f"  Latency (mean): {par_mean_lat:.3f} ms")
    print(f"  Latency (p50):  {par_p50_lat:.3f} ms")
    print(f"  Recall@{k}:      {par_rec_k * 100:.2f}%")

    # Verify equivalence between sequential and parallel search results
    exact_matches = sum(
        1 for s_ids, p_ids in zip(seq_predicted_ids, par_predicted_ids) if s_ids == p_ids
    )
    assert (
        exact_matches == num_query
    ), f"Parallel results differed from sequential for {num_query - exact_matches} queries!"

    speedup = par_qps / seq_qps if seq_qps > 0 else 1.0

    print("\n" + "=" * 70)
    print(" SUMMARY: SEQUENTIAL VS. PARALLEL SHARDED SEARCH")
    print("=" * 70)
    print(f"  Dataset:             {dataset_name} (N={num_base:,}, D={dim})")
    print(f"  Num Shards:          {num_shards}")
    print(f"  Queries:             {num_query:,}")
    print(f"  Sequential QPS:      {seq_qps:,.1f} QPS (p50: {seq_p50_lat:.3f} ms)")
    print(f"  Parallel QPS:        {par_qps:,.1f} QPS (p50: {par_p50_lat:.3f} ms)")
    print(f"  Parallel Speedup:    {speedup:.2f}x")
    print(f"  Recall@{k} (Seq):     {seq_rec_k * 100:.2f}% (exact search sanity check)")
    print(f"  Recall@{k} (Par):     {par_rec_k * 100:.2f}% (exact search sanity check)")
    print(f"  Equivalence:         100% identical ({exact_matches}/{num_query} queries match)")
    print("=" * 70)

    benchmark_data = {
        "dataset": dataset_name,
        "num_vectors": num_base,
        "dimension": dim,
        "num_shards": num_shards,
        "shard_sizes": shard_sizes,
        "num_queries": num_query,
        "k": k,
        "metric": "euclidean",
        "build_time_sec": build_time_sec,
        "build_rate_vec_per_sec": build_rate,
        "sequential": {
            "search_time_sec": seq_time_sec,
            "qps": seq_qps,
            "latency_ms": {
                "mean": seq_mean_lat,
                "p50": seq_p50_lat,
                "p95": seq_p95_lat,
                "p99": seq_p99_lat,
            },
            "recall_at_k": seq_rec_k,
        },
        "parallel": {
            "search_time_sec": par_time_sec,
            "qps": par_qps,
            "latency_ms": {
                "mean": par_mean_lat,
                "p50": par_p50_lat,
                "p95": par_p95_lat,
                "p99": par_p99_lat,
            },
            "recall_at_k": par_rec_k,
        },
        "speedup_factor": speedup,
        "result_equivalence": exact_matches == num_query,
    }

    if save_json:
        saved_path = save_benchmark_result("sharded_index", benchmark_data)
        print(f"\nResults saved to: {saved_path}")

    return benchmark_data


def main():
    parser = argparse.ArgumentParser(description="Vecta ShardedFlatIndex Benchmark")
    parser.add_argument("--k", type=int, default=10, help="Top-k nearest neighbors (default: 10)")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of shards (default: 4)")
    parser.add_argument("--no-save", action="store_true", help="Do not save results to JSON")
    args = parser.parse_args()

    run_sharded_index_benchmark(k=args.k, num_shards=args.num_shards, save_json=not args.no_save)


if __name__ == "__main__":
    main()
