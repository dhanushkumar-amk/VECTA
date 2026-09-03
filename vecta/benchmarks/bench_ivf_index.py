"""
End-to-end benchmark for IVFIndex on standard SIFT dataset (Phase 14).

Measures:
- IVFIndex build & training time (k-means clustering + vector insertion)
- Cluster distribution and imbalance diagnostics
- For nprobe in [1, 5, 10, 20, 50]:
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


def run_ivf_index_benchmark(
    k_clusters: int = 100,
    top_k: int = 10,
    nprobe_list: list[int] = None,
    dataset_name: str = "siftsmall",
    save_json: bool = True,
) -> dict:
    """
    Run full speed vs recall benchmark on IVFIndex across varying nprobe values.
    """
    if nprobe_list is None:
        nprobe_list = [1, 5, 10, 20, 50]

    print("=" * 70)
    print(" VECTA BENCHMARK: IVFIndex (Inverted File Index)")
    print(f" Dataset: {dataset_name} | Clusters (k): {k_clusters} | Target Top-k: {top_k}")
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

    # 2. Build and Train IVF Index
    print(f"\n[2/4] Training and building IVFIndex (k={k_clusters} clusters)...")
    ivf_index = vecta.IVFIndex(dim=dim, num_clusters=k_clusters, metric="euclidean")
    ids = list(range(num_base))
    vectors_list = base.tolist()

    with Timer("IVF Training") as t_train:
        # Train k-means coarse quantizer on the base dataset
        ivf_index.train(vectors_list, k=k_clusters, max_iterations=25, seed=42)
    train_time_sec = t_train.elapsed_sec
    print(f"  Training time:  {t_train.elapsed_ms:.2f} ms ({train_time_sec:.4f} s)")

    with Timer("IVF Ingestion") as t_add:
        ivf_index.add_batch(ids, vectors_list)
    add_time_sec = t_add.elapsed_sec
    total_build_sec = train_time_sec + add_time_sec
    print(f"  Ingestion time: {t_add.elapsed_ms:.2f} ms ({add_time_sec:.4f} s)")
    print(f"  Total build:    {total_build_sec * 1000.0:.2f} ms ({total_build_sec:.4f} s)")
    assert len(ivf_index) == num_base

    # Cluster diagnostics
    sizes = ivf_index.cluster_sizes()
    min_size = min(sizes)
    max_size = max(sizes)
    mean_size = float(np.mean(sizes))
    print(f"  Cluster sizes:  min={min_size}, max={max_size}, mean={mean_size:.1f}")

    # 3. Benchmark Across nprobe values
    print(f"\n[3/4] Benchmarking search throughput and recall across nprobe values...")
    query_lists = query.tolist()
    gt_list = gt.tolist()

    nprobe_results = []

    for nprobe in nprobe_list:
        predicted_ids = []
        latencies_ms = []

        with Timer(f"Search nprobe={nprobe}") as t_search:
            for q_vec in query_lists:
                t0 = time.perf_counter()
                results = ivf_index.search(q_vec, k=top_k, nprobe=nprobe)
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
        })

    # 4. Print Summary Table
    print("\n[4/4] Summary Results Table:")
    print("-" * 75)
    print(f"{'nprobe':>8} | {'Recall@10':>10} | {'Throughput (QPS)':>18} | {'Latency Mean':>14} | {'Latency p95':>12}")
    print("-" * 75)
    for r in nprobe_results:
        print(
            f"{r['nprobe']:>8} | "
            f"{r['recall_at_10'] * 100.0:>9.2f}% | "
            f"{r['qps']:>18,.1f} | "
            f"{r['mean_latency_ms']:>11.3f} ms | "
            f"{r['p95_latency_ms']:>9.3f} ms"
        )
    print("-" * 75)

    benchmark_data = {
        "index_type": "IVFIndex",
        "dataset": dataset_name,
        "num_vectors": num_base,
        "dim": dim,
        "num_queries": num_query,
        "k_clusters": k_clusters,
        "top_k": top_k,
        "train_time_sec": train_time_sec,
        "add_time_sec": add_time_sec,
        "total_build_time_sec": total_build_sec,
        "nprobe_benchmarks": nprobe_results,
    }

    if save_json:
        results_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
        filepath = save_benchmark_result(
            benchmark_name=f"ivf_index_{dataset_name}",
            metrics=benchmark_data,
            results_dir=results_dir,
        )
        print(f"\nArtifact saved to: {filepath}")

    return benchmark_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vecta IVFIndex Benchmark")
    parser.add_argument("--k-clusters", type=int, default=100, help="Number of IVF clusters (centroids)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k nearest neighbors to retrieve")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON results")
    args = parser.parse_args()

    run_ivf_index_benchmark(
        k_clusters=args.k_clusters,
        top_k=args.top_k,
        save_json=not args.no_save,
    )
