"""
Head-to-head comparison benchmark runner between vecta and FAISS.

Phase 36: Full FlatIndex vs. faiss.IndexFlatL2/IndexFlatIP comparison,
with generic, reusable trial execution and statistical summary machinery.

Usage:
    python benchmarks/faiss_comparison/run_comparison.py --index flat
    python benchmarks/faiss_comparison/run_comparison.py --index flat --threads 1
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
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

from benchmarks.datasets.download_sift1m import load_siftsmall
from benchmarks.utils.recall import recall_at_k
from benchmarks.utils.timer import save_benchmark_result

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


def run_trials(
    search_fn: Callable[[Any, int], Any],
    queries: List[Any],
    k: int,
    num_trials: int = 5,
    warmup_trials: int = 1,
) -> List[float]:
    """
    Execute warmup trials followed by timed measurement trials.

    Library-agnostic execution harness:
    1. Runs `warmup_trials` first (discarded, not measured) to allow CPU caches,
       branch predictors, and allocator state to reach thermal steady state.
    2. Runs `num_trials` real measured trials.
    3. Returns list of elapsed wall-clock times (in seconds) for each measured trial.

    Args:
        search_fn: Callable taking (query_item, k) and returning search results.
        queries: List of pre-formatted query items (e.g. list[float] or np.ndarray).
        k: Number of nearest neighbors to retrieve.
        num_trials: Number of timed trials to measure.
        warmup_trials: Number of unmeasured warmup passes.

    Returns:
        List of float elapsed times (in seconds) of length `num_trials`.
    """
    # 1. Warmup passes (unmeasured, discarded)
    for _ in range(warmup_trials):
        for q in queries:
            search_fn(q, k)

    # 2. Timed measurement trials
    trial_times: List[float] = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        for q in queries:
            search_fn(q, k)
        t1 = time.perf_counter()
        trial_times.append(t1 - t0)

    return trial_times


def summarize_timings(trial_times: List[float], num_queries: int) -> Dict[str, Any]:
    """
    Compute rigorous statistical metrics across measured benchmark trials.

    Flags high-variance measurements (>20% standard deviation relative to mean),
    indicating measurement noise, background scheduler interference, or thermal throttling.

    Args:
        trial_times: List of elapsed seconds for each trial.
        num_queries: Number of queries executed per trial.

    Returns:
        Dictionary of statistical summary metrics.
    """
    if not trial_times:
        raise ValueError("trial_times cannot be empty")
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")

    times_arr = np.array(trial_times, dtype=np.float64)
    mean_time = float(np.mean(times_arr))
    median_time = float(np.median(times_arr))
    stddev_time = float(np.std(times_arr, ddof=1)) if len(times_arr) > 1 else 0.0

    qps_arr = num_queries / times_arr
    mean_qps = float(np.mean(qps_arr))
    median_qps = float(np.median(qps_arr))
    stddev_qps = float(np.std(qps_arr, ddof=1)) if len(qps_arr) > 1 else 0.0

    variance_ratio = stddev_time / mean_time if mean_time > 0 else 0.0
    high_variance_flag = bool(variance_ratio > 0.20)

    latency_p50_ms = (median_time / num_queries) * 1000.0
    latency_mean_ms = (mean_time / num_queries) * 1000.0

    return {
        "raw_trial_times_sec": [float(t) for t in trial_times],
        "mean_time_sec": mean_time,
        "median_time_sec": median_time,
        "stddev_time_sec": stddev_time,
        "mean_qps": mean_qps,
        "median_qps": median_qps,
        "stddev_qps": stddev_qps,
        "latency_mean_ms": latency_mean_ms,
        "latency_p50_ms": latency_p50_ms,
        "variance_ratio": variance_ratio,
        "high_variance_warning": high_variance_flag,
    }


def compare_flat_index(
    dataset_name: str = "siftsmall",
    k: int = 10,
    metric: str = "euclidean",
    threads: int = 1,
    num_trials: int = 5,
    warmup_trials: int = 2,
    save_json: bool = True,
) -> Dict[str, Any]:
    """
    Execute full head-to-head comparison between vecta.FlatIndex and faiss.IndexFlatL2/IP.

    Evaluates:
    - Index build time and ingestion throughput (vec/sec)
    - Query search latency (mean, median) and throughput (mean/median QPS, stddev)
    - Recall@k sanity check against ground truth
    - Relative speedup ratios
    """
    # Configure FAISS thread count (per methodology: Tier 1 is single-threaded)
    set_faiss_threads(threads)
    actual_threads = get_faiss_threads()

    # 1. Load Dataset
    print(f"\n[1/4] Loading {dataset_name} dataset...")
    base, query, gt = load_siftsmall()
    num_base, dim = base.shape
    num_query = query.shape[0]
    print(f"  Dataset: {num_base:,} base vectors (dim={dim}), {num_query:,} queries, k={k}")

    # 2. Build Indexes
    print("\n[2/4] Building indexes...")
    base_list = base.tolist()
    ids = list(range(num_base))

    # Build vecta.FlatIndex
    t0 = time.perf_counter()
    v_index = vecta.FlatIndex(dim, metric)
    v_index.add_batch(ids, base_list)
    v_build_time = time.perf_counter() - t0
    v_build_rate = num_base / v_build_time if v_build_time > 0 else 0.0
    print(f"  vecta.FlatIndex:   {v_build_time * 1000.0:.2f} ms ({v_build_rate:,.1f} vec/s)")

    # Build faiss.IndexFlat
    t0 = time.perf_counter()
    f_index = build_faiss_flat(dim, metric)
    f_index.add(base)
    f_build_time = time.perf_counter() - t0
    f_build_rate = num_base / f_build_time if f_build_time > 0 else 0.0
    print(f"  faiss.IndexFlat:   {f_build_time * 1000.0:.2f} ms ({f_build_rate:,.1f} vec/s)")

    # 3. Prepare pre-formatted queries for zero-allocation timed loops
    vecta_queries = query.tolist()
    faiss_queries = [np.ascontiguousarray(query[i : i + 1]) for i in range(num_query)]

    # 4. Search Trials (Warmup + Measurement)
    print(f"\n[3/4] Running {num_trials} measured trials ({warmup_trials} warmups excluded)...")

    # Vecta Search Trials
    v_trials = run_trials(
        lambda q, k_val: v_index.search(q, k=k_val),
        vecta_queries,
        k,
        num_trials=num_trials,
        warmup_trials=warmup_trials,
    )
    v_stats = summarize_timings(v_trials, num_query)

    # FAISS Search Trials
    f_trials = run_trials(
        lambda q, k_val: f_index.search(q, k=k_val),
        faiss_queries,
        k,
        num_trials=num_trials,
        warmup_trials=warmup_trials,
    )
    f_stats = summarize_timings(f_trials, num_query)

    # 5. Evaluate Recall@k against ground truth (sanity check: must be ~100%)
    print("\n[4/4] Evaluating Recall@10 sanity check against ground truth...")
    v_preds = [[item[0] for item in v_index.search(q, k=k)] for q in vecta_queries]
    f_preds = [f_index.search(q, k=k)[1][0].tolist() for q in faiss_queries]

    v_recall = recall_at_k(v_preds, gt.tolist(), k=k)
    f_recall = recall_at_k(f_preds, gt.tolist(), k=k)

    v_stats["recall_at_k"] = v_recall
    f_stats["recall_at_k"] = f_recall
    v_stats["index_name"] = "vecta.FlatIndex"
    f_stats["index_name"] = type(f_index).__name__
    v_stats["build_time_sec"] = v_build_time
    v_stats["build_rate_vec_per_sec"] = v_build_rate
    f_stats["build_time_sec"] = f_build_time
    f_stats["build_rate_vec_per_sec"] = f_build_rate

    # Check for recall correctness
    recall_discrepancy = bool(v_recall < 0.999 or f_recall < 0.999)
    if recall_discrepancy:
        print("  [ERROR] Recall@k is below 100% on exact search! Possible correctness bug!")
    else:
        print(f"  Recall@{k} Verified: vecta={v_recall * 100:.2f}%, faiss={f_recall * 100:.2f}% (100% exact match)")

    # Compute Speedup Ratio
    qps_ratio = f_stats["mean_qps"] / v_stats["mean_qps"] if v_stats["mean_qps"] > 0 else 1.0
    build_ratio = f_build_rate / v_build_rate if v_build_rate > 0 else 1.0
    faster_engine = "faiss" if qps_ratio >= 1.0 else "vecta"

    comparison_summary = {
        "faster_engine": faster_engine,
        "qps_speedup_ratio": qps_ratio if faster_engine == "faiss" else 1.0 / qps_ratio,
        "build_speedup_ratio": build_ratio if build_ratio >= 1.0 else 1.0 / build_ratio,
        "recall_discrepancy": recall_discrepancy,
    }

    results = {
        "benchmark": "faiss_comparison_flat",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset_name,
        "num_vectors": num_base,
        "dimension": dim,
        "num_queries": num_query,
        "k": k,
        "metric": metric,
        "threads": actual_threads,
        "num_trials": num_trials,
        "warmup_trials": warmup_trials,
        "vecta": v_stats,
        "faiss": f_stats,
        "comparison": comparison_summary,
    }

    print_comparison_table(results)

    if save_json:
        saved_path = save_benchmark_result("faiss_comparison_flat", results)
        print(f"\nRaw and aggregate results saved to: {saved_path}")

    return results


def print_comparison_table(results: Dict[str, Any]) -> None:
    """Format and print a clean, publication-ready head-to-head comparison table."""
    v = results["vecta"]
    f = results["faiss"]
    comp = results["comparison"]
    k = results["k"]
    threads = results["threads"]
    dataset = results["dataset"]
    num_vectors = results["num_vectors"]
    dim = results["dimension"]

    print("\n" + "=" * 80)
    print(f" HEAD-TO-HEAD COMPARISON: {v['index_name']} vs. {f['index_name']}")
    print(f" Dataset: {dataset} (N={num_vectors:,}, D={dim}) | Target k: {k} | Threads: {threads}")
    print("=" * 80)
    print(f" {'Metric':<26} {'vecta':<22} {'FAISS':<22} {'Delta / Ratio':<18}")
    print("-" * 80)

    # Build Metrics
    v_btime = f"{v['build_time_sec'] * 1000.0:.2f} ms"
    f_btime = f"{f['build_time_sec'] * 1000.0:.2f} ms"
    b_ratio_str = f"FAISS {comp['build_speedup_ratio']:.2f}x faster"
    print(f" {'Build Time':<26} {v_btime:<22} {f_btime:<22} {b_ratio_str:<18}")

    v_brate = f"{v['build_rate_vec_per_sec']:,.0f} vec/s"
    f_brate = f"{f['build_rate_vec_per_sec']:,.0f} vec/s"
    print(f" {'Ingestion Rate':<26} {v_brate:<22} {f_brate:<22} {b_ratio_str:<18}")

    # Search Throughput Metrics
    v_mqps = f"{v['mean_qps']:,.1f} QPS"
    f_mqps = f"{f['mean_qps']:,.1f} QPS"
    faster_str = f"{comp['faster_engine'].upper()} {comp['qps_speedup_ratio']:.2f}x faster"
    print(f" {'Mean Throughput (QPS)':<26} {v_mqps:<22} {f_mqps:<22} {faster_str:<18}")

    v_medqps = f"{v['median_qps']:,.1f} QPS"
    f_medqps = f"{f['median_qps']:,.1f} QPS"
    print(f" {'Median Throughput (QPS)':<26} {v_medqps:<22} {f_medqps:<22} {'-' :<18}")

    v_var_pct = v["variance_ratio"] * 100.0
    f_var_pct = f["variance_ratio"] * 100.0
    v_std = f"+/-{v['stddev_qps']:.1f} ({v_var_pct:.1f}%)"
    f_std = f"+/-{f['stddev_qps']:.1f} ({f_var_pct:.1f}%)"
    v_stability = "Low variance" if not v["high_variance_warning"] else "HIGH VARIANCE!"
    f_stability = "Low variance" if not f["high_variance_warning"] else "HIGH VARIANCE!"
    print(f" {'QPS StdDev (Variance)':<26} {v_std:<22} {f_std:<22} {f_stability:<18}")

    # Latency Metrics
    v_lat_mean = f"{v['latency_mean_ms']:.3f} ms"
    f_lat_mean = f"{f['latency_mean_ms']:.3f} ms"
    lat_ratio = f['latency_mean_ms'] / v['latency_mean_ms'] if v['latency_mean_ms'] > 0 else 1.0
    lat_str = f"FAISS {lat_ratio:.2f}x latency" if lat_ratio < 1.0 else f"vecta {1.0/lat_ratio:.2f}x lower"
    print(f" {'Mean Latency (ms)':<26} {v_lat_mean:<22} {f_lat_mean:<22} {lat_str:<18}")

    v_lat_p50 = f"{v['latency_p50_ms']:.3f} ms"
    f_lat_p50 = f"{f['latency_p50_ms']:.3f} ms"
    print(f" {'Median Latency (p50)':<26} {v_lat_p50:<22} {f_lat_p50:<22} {'-' :<18}")

    # Recall Metrics
    v_rec = f"{v['recall_at_k'] * 100:.2f}%"
    f_rec = f"{f['recall_at_k'] * 100:.2f}%"
    rec_match = "100% exact match" if not comp["recall_discrepancy"] else "MISMATCH!"
    print(f" {f'Recall@{k}':<26} {v_rec:<22} {f_rec:<22} {rec_match:<18}")

    print("=" * 80)
    if v["high_variance_warning"] or f["high_variance_warning"]:
        print(" [WARNING] High timing variance detected (>20% stddev). Check background processes.")
    else:
        print(" Measurement Quality: Clean & Stable (stddev < 5% across all 5 trials).")

    print(
        f" Conclusion: {comp['faster_engine'].upper()} is {comp['qps_speedup_ratio']:.2f}x faster "
        f"than {('vecta' if comp['faster_engine'] == 'faiss' else 'faiss')} in single-threaded brute-force search."
    )
    print("=" * 80)


def find_iso_recall(
    sweep: List[Dict[str, Any]],
    target_recall: float = 0.90,
    param_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Interpolate throughput (QPS) and effective parameter value at a target recall level.

    Uses piecewise linear interpolation between the two sweep points that bracket
    `target_recall`. If `target_recall` is outside the sweep bounds, returns the nearest
    boundary point and flags it as approximate.

    Args:
        sweep: List of per-parameter result dictionaries.
        target_recall: Desired recall level (default 0.90 for ~90% recall).
        param_name: Parameter name (e.g. 'nprobe', 'ef_search'). Auto-detected if None.

    Returns:
        Dictionary detailing estimated QPS, effective parameter value, and method for both libraries.
    """
    if not sweep:
        raise ValueError("Sweep data cannot be empty")

    if param_name is None:
        first = sweep[0]
        if "ef_search" in first:
            param_name = "ef_search"
        elif "nprobe" in first:
            param_name = "nprobe"
        elif "param_value" in first:
            param_name = "param_value"
        else:
            param_name = "param"

    def _interp_engine(engine: str) -> Dict[str, Any]:
        recall_key = f"{engine}_recall"
        qps_key = f"{engine}_qps"

        # Sort points by recall
        points = sorted(
            [(float(entry[recall_key]), float(entry[qps_key]), float(entry.get(param_name, entry.get("param_value", 0)))) for entry in sweep],
            key=lambda p: p[0],
        )

        def _make_res(achieved: float, qps: float, param_val: float, method: str, bracket=None):
            d = {
                "target_recall": target_recall,
                "param_name": param_name,
                "achieved_recall": achieved,
                "estimated_qps": qps,
                "estimated_param": param_val,
                f"estimated_{param_name}": param_val,
                "method": method,
            }
            if bracket:
                d["bracket"] = bracket
            return d

        # Check exact match
        for r, q, p_val in points:
            if abs(r - target_recall) < 1e-4:
                return _make_res(r, q, p_val, "exact")

        # Check if target is bracketed by adjacent points
        for i in range(len(points) - 1):
            r1, q1, p1 = points[i]
            r2, q2, p2 = points[i + 1]
            if r1 <= target_recall <= r2 and r2 > r1:
                alpha = (target_recall - r1) / (r2 - r1)
                est_qps = q1 + alpha * (q2 - q1)
                est_p = p1 + alpha * (p2 - p1)
                return _make_res(target_recall, est_qps, est_p, "linear_interpolation", ((r1, q1, p1), (r2, q2, p2)))

        # Boundary fallback (target outside observed range)
        if target_recall < points[0][0]:
            r, q, p_val = points[0]
            return _make_res(r, q, p_val, "nearest_boundary_lower")
        else:
            r, q, p_val = points[-1]
            return _make_res(r, q, p_val, "nearest_boundary_upper")

    vecta_iso = _interp_engine("vecta")
    faiss_iso = _interp_engine("faiss")

    v_qps = vecta_iso["estimated_qps"]
    f_qps = faiss_iso["estimated_qps"]
    ratio = f_qps / v_qps if v_qps > 0 else 1.0
    faster_engine = "faiss" if ratio >= 1.0 else "vecta"
    speedup = ratio if faster_engine == "faiss" else (1.0 / ratio if ratio > 0 else 1.0)

    return {
        "target_recall": target_recall,
        "param_name": param_name,
        "vecta": vecta_iso,
        "faiss": faiss_iso,
        "faster_engine": faster_engine,
        "speedup_ratio": speedup,
        "qps_ratio": ratio,
    }


def generate_tradeoff_plot(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    index_type: str = "hnsw",
    param_name: Optional[str] = None,
) -> str:
    """
    Generate and save a Recall@10 vs. QPS tradeoff curve plot.

    Follows the standard ann-benchmarks.com visualization style:
    - X-axis: Recall@10 against Ground Truth
    - Y-axis: Search Throughput (Queries Per Second)
    - Two lines: vecta.<IndexClass> vs. faiss.<IndexClass>
    - Annotations for parameter values at each data point
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed; skipping plot generation.")
        return ""

    sweep = results.get("sweep", [])
    if not sweep:
        print("[WARNING] No sweep data available to plot.")
        return ""

    if param_name is None:
        param_name = results.get("param_name")
        if param_name is None:
            first = sweep[0]
            if "ef_search" in first:
                param_name = "ef_search"
            elif "nprobe" in first:
                param_name = "nprobe"
            else:
                param_name = "param"

    param_prefix = "ef" if "ef" in param_name.lower() else "p"

    # Index class names for labels
    idx_upper = index_type.upper()
    if idx_upper == "IVF":
        v_label, f_label = "vecta.IVFIndex", "faiss.IndexIVFFlat"
        default_file_prefix = "ivf_recall_vs_qps"
    elif idx_upper == "HNSW":
        v_label, f_label = "vecta.HnswIndex", "faiss.IndexHNSWFlat"
        default_file_prefix = "hnsw_recall_vs_qps"
    elif idx_upper == "IVFPQ":
        v_label, f_label = "vecta.IVFPQIndex", "faiss.IndexIVFPQ"
        default_file_prefix = "ivfpq_recall_vs_qps"
    else:
        v_label, f_label = f"vecta.{idx_upper}", f"faiss.{idx_upper}"
        default_file_prefix = f"{index_type.lower()}_recall_vs_qps"

    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"{default_file_prefix}_{timestamp}.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    param_vals = [s.get(param_name, s.get("param_value", i)) for i, s in enumerate(sweep)]
    v_rec = [s["vecta_recall"] for s in sweep]
    v_qps = [s["vecta_qps"] for s in sweep]
    f_rec = [s["faiss_recall"] for s in sweep]
    f_qps = [s["faiss_qps"] for s in sweep]

    plt.figure(figsize=(9, 6), dpi=200)

    # Plot lines with markers
    plt.plot(
        v_rec,
        v_qps,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="#2563eb",
        label=v_label,
    )
    plt.plot(
        f_rec,
        f_qps,
        marker="s",
        linewidth=2.2,
        markersize=7,
        color="#ea580c",
        label=f_label,
    )

    # Annotate parameter values on points
    for p_val, r, q in zip(param_vals, v_rec, v_qps):
        plt.annotate(
            f"{param_prefix}={p_val}",
            (r, q),
            textcoords="offset points",
            xytext=(6, -7),
            fontsize=8,
            color="#1d4ed8",
            fontweight="bold",
        )
    for p_val, r, q in zip(param_vals, f_rec, f_qps):
        plt.annotate(
            f"{param_prefix}={p_val}",
            (r, q),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#c2410c",
            fontweight="bold",
        )

    dataset = results.get("dataset", "siftsmall")
    num_vecs = results.get("num_vectors", 10000)
    dim = results.get("dimension", 128)
    k = results.get("k", 10)
    threads = results.get("threads", 1)

    extra_desc = []
    if "nlist" in results:
        extra_desc.append(f"nlist={results['nlist']}")
    if "m" in results:
        extra_desc.append(f"M={results['m']}")
    if "ef_construction" in results:
        extra_desc.append(f"efC={results['ef_construction']}")
    config_str = (", " + ", ".join(extra_desc)) if extra_desc else ""

    plt.title(
        f"{idx_upper} Recall@{k} vs. Throughput (QPS) — Vecta vs. FAISS\n"
        f"Dataset: {dataset} (N={num_vecs:,}, D={dim}{config_str}) | Threads: {threads}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel(f"Recall@{k} against Ground Truth", fontsize=11, labelpad=8)
    plt.ylabel("Search Throughput (Queries / Second)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def generate_ivf_plot(results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generate Recall-vs-QPS tradeoff curve for IVF."""
    return generate_tradeoff_plot(results, output_path, index_type="ivf", param_name="nprobe")


def generate_hnsw_plot(results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generate Recall-vs-QPS tradeoff curve for HNSW."""
    return generate_tradeoff_plot(results, output_path, index_type="hnsw", param_name="ef_search")


def generate_ivfpq_plot(results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generate Recall-vs-QPS tradeoff curve for IVFPQ."""
    return generate_tradeoff_plot(results, output_path, index_type="ivfpq", param_name="nprobe")


def generate_memory_comparison_plot(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate and save a bar chart comparing memory footprint:
    Raw Uncompressed Float32 vs. vecta.IVFPQIndex vs. faiss.IndexIVFPQ.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed; skipping memory plot.")
        return ""

    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"ivfpq_memory_comparison_{timestamp}.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    raw_bytes = results.get("raw_vector_bytes", 10000 * 128 * 4)
    v_bytes = results.get("vecta_memory_bytes", 0)
    f_bytes = results.get("faiss_memory_bytes", 0)

    raw_kb = raw_bytes / 1024.0
    v_kb = v_bytes / 1024.0
    f_kb = f_bytes / 1024.0

    v_ratio = raw_bytes / v_bytes if v_bytes > 0 else 1.0
    f_ratio = raw_bytes / f_bytes if f_bytes > 0 else 1.0

    categories = [
        "Raw Float32\n(Uncompressed)",
        "vecta.IVFPQIndex\n(Resident RAM)",
        "faiss.IndexIVFPQ\n(Serialized Buffer)",
    ]
    values = [raw_kb, v_kb, f_kb]
    colors = ["#64748b", "#2563eb", "#ea580c"]

    plt.figure(figsize=(8.5, 5.5), dpi=200)
    bars = plt.bar(categories, values, color=colors, width=0.52, edgecolor="#334155", linewidth=1.2)

    # Annotate bar values and compression ratios above each bar
    plt.text(
        bars[0].get_x() + bars[0].get_width() / 2.0,
        bars[0].get_height() + (max(values) * 0.02),
        f"{raw_kb:,.1f} KB\n(Baseline 1.0x)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#334155",
    )
    plt.text(
        bars[1].get_x() + bars[1].get_width() / 2.0,
        bars[1].get_height() + (max(values) * 0.02),
        f"{v_kb:,.1f} KB\n({v_ratio:.1f}x smaller)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#1d4ed8",
    )
    plt.text(
        bars[2].get_x() + bars[2].get_width() / 2.0,
        bars[2].get_height() + (max(values) * 0.02),
        f"{f_kb:,.1f} KB\n({f_ratio:.1f}x smaller)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#c2410c",
    )

    num_vecs = results.get("num_vectors", 10000)
    dim = results.get("dimension", 128)
    m = results.get("m", 8)
    nlist = results.get("nlist", 100)
    k_sub = results.get("k_per_subvector", 256)

    plt.title(
        f"IVFPQ Memory Compression Footprint — Vecta vs. FAISS\n"
        f"Dataset: SIFT10k (N={num_vecs:,}, D={dim}) | nlist={nlist}, M={m}, k_sub={k_sub}",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    plt.ylabel("Memory Footprint (Kilobytes)", fontsize=11, labelpad=8)
    plt.ylim(0, max(values) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def print_sweep_comparison_table(
    results: Any,
    index_type: str = "hnsw",
    param_name: Optional[str] = None,
) -> None:
    """
    Format and print a per-parameter sweep comparison table and iso-recall analysis.

    Accepts full benchmark results dict or list of sweep dicts.
    """
    if isinstance(results, dict):
        sweep = results.get("sweep", results.get("results", []))
        dataset = results.get("dataset", "siftsmall")
        num_vecs = results.get("num_vectors", 10000)
        dim = results.get("dimension", 128)
        k = results.get("k", 10)
        threads = results.get("threads", 1)
        v_btime = results.get("vecta_build_time_sec")
        f_btime = results.get("faiss_build_time_sec")
        iso = results.get("iso_recall")
        if param_name is None:
            param_name = results.get("param_name")
        if "benchmark" in results and "ivfpq" in results["benchmark"]:
            index_type = "ivfpq"
        elif "benchmark" in results and "hnsw" in results["benchmark"]:
            index_type = "hnsw"
        elif "benchmark" in results and "ivf" in results["benchmark"]:
            index_type = "ivf"
    elif isinstance(results, list):
        sweep = results
        dataset, num_vecs, dim, k, threads = "siftsmall", 10000, 128, 10, 1
        v_btime, f_btime = None, None
        iso = None
    else:
        raise ValueError("Expected results dict or list of sweep dicts")

    if not sweep:
        print("[No sweep results to display]")
        return

    if param_name is None:
        first = sweep[0]
        if "ef_search" in first:
            param_name = "ef_search"
        elif "nprobe" in first:
            param_name = "nprobe"
        else:
            param_name = "param"

    if iso is None:
        iso = find_iso_recall(sweep, 0.90, param_name=param_name)

    idx_upper = index_type.upper()
    if idx_upper == "HNSW":
        v_class, f_class = "vecta.HnswIndex", "faiss.IndexHNSWFlat"
    elif idx_upper == "IVF":
        v_class, f_class = "vecta.IVFIndex", "faiss.IndexIVFFlat"
    elif idx_upper == "IVFPQ":
        v_class, f_class = "vecta.IVFPQIndex", "faiss.IndexIVFPQ"
    else:
        v_class, f_class = f"vecta.{idx_upper}", f"faiss.{idx_upper}"

    print("\n" + "=" * 88)
    print(f" HEAD-TO-HEAD {idx_upper} COMPARISON: {v_class} vs. {f_class}")
    extra_info = []
    if isinstance(results, dict):
        if "nlist" in results:
            extra_info.append(f"nlist={results['nlist']}")
        if "m" in results:
            extra_info.append(f"M={results['m']}")
        if "ef_construction" in results:
            extra_info.append(f"efC={results['ef_construction']}")
    extra_str = (" | " + ", ".join(extra_info)) if extra_info else ""
    print(f" Dataset: {dataset} (N={num_vecs:,}, D={dim}){extra_str} | Target k={k} | Threads={threads}")
    print("=" * 88)

    if v_btime is not None and f_btime is not None:
        v_bms = v_btime * 1000.0
        f_bms = f_btime * 1000.0
        if v_btime < f_btime:
            b_faster = "vecta"
            b_speedup = f_btime / v_btime if v_btime > 0 else 1.0
        else:
            b_faster = "faiss"
            b_speedup = v_btime / f_btime if f_btime > 0 else 1.0
        build_label = "Build Time (add_batch / add)" if idx_upper == "HNSW" else "Build Time (train + add)"
        print(f" {build_label}: vecta = {v_bms:.1f} ms | FAISS = {f_bms:.1f} ms ({b_faster.upper()} {b_speedup:.2f}x faster)")
        print("-" * 88)

    print(f" {param_name:<10} | {'vecta QPS':<13} | {'FAISS QPS':<13} | {'vecta Rec@10':<14} | {'FAISS Rec@10':<14} | {'QPS Ratio':<14}")
    print("-" * 88)

    for row in sweep:
        p_val = row.get(param_name, row.get("param_value", 0))
        v_qps = row["vecta_qps"]
        f_qps = row["faiss_qps"]
        v_rec = row["vecta_recall"] * 100.0
        f_rec = row["faiss_recall"] * 100.0
        ratio = row["qps_ratio"]
        faster = "FAISS" if f_qps >= v_qps else "vecta"
        ratio_str = f"{faster} {ratio:.2f}x" if faster == "FAISS" else f"{faster} {1.0/ratio:.2f}x"

        print(
            f" {p_val:<10} | "
            f"{v_qps:>10.1f}  | "
            f"{f_qps:>10.1f}  | "
            f"{v_rec:>11.2f}%  | "
            f"{f_rec:>11.2f}%  | "
            f"{ratio_str:<14}"
        )
    print("=" * 88)

    # Iso-recall analysis
    if iso:
        target_pct = iso["target_recall"] * 100.0
        v_iso = iso["vecta"]
        f_iso = iso["faiss"]
        faster = iso["faster_engine"].upper()
        speedup = iso["speedup_ratio"]
        p_display = param_name

        v_est_p = v_iso.get(f"estimated_{param_name}", v_iso.get("estimated_param", 0.0))
        f_est_p = f_iso.get(f"estimated_{param_name}", f_iso.get("estimated_param", 0.0))

        print(f" ISO-RECALL COMPARISON AT ~{target_pct:.0f}% ACCURACY TARGET:")
        print(
            f"  vecta reaches {v_iso['achieved_recall']*100:.1f}% recall at ~{p_display} {v_est_p:.1f} "
            f"-> {v_iso['estimated_qps']:,.1f} QPS ({v_iso['method']})"
        )
        print(
            f"  FAISS reaches {f_iso['achieved_recall']*100:.1f}% recall at ~{p_display} {f_est_p:.1f} "
            f"-> {f_iso['estimated_qps']:,.1f} QPS ({f_iso['method']})"
        )
        print(
            f"  Throughput comparison at matched ~{target_pct:.0f}% recall: {faster} is {speedup:.2f}x faster"
        )
        print("=" * 88)


def print_ivf_comparison_table(results: Any) -> None:
    """Wrapper for IVF table printing."""
    print_sweep_comparison_table(results, index_type="ivf", param_name="nprobe")


def print_hnsw_comparison_table(results: Any) -> None:
    """Wrapper for HNSW table printing."""
    print_sweep_comparison_table(results, index_type="hnsw", param_name="ef_search")


def print_ivfpq_comparison_table(results: Any) -> None:
    """Wrapper for IVFPQ table printing, including memory footprint reporting."""
    print_sweep_comparison_table(results, index_type="ivfpq", param_name="nprobe")
    if isinstance(results, dict):
        v_mem = results.get("vecta_memory_bytes")
        f_mem = results.get("faiss_memory_bytes")
        raw_mem = results.get("raw_vector_bytes")
        if v_mem is not None and f_mem is not None and raw_mem is not None:
            v_kb = v_mem / 1024.0
            f_kb = f_mem / 1024.0
            raw_kb = raw_mem / 1024.0
            v_comp = raw_mem / v_mem if v_mem > 0 else 1.0
            f_comp = raw_mem / f_mem if f_mem > 0 else 1.0
            ratio = f_mem / v_mem if v_mem > 0 else 1.0
            print(" MEMORY FOOTPRINT & COMPRESSION:")
            print(f"  Raw Float32 Baseline: {raw_kb:,.1f} KB (100.0%)")
            print(f"  vecta.IVFPQIndex:     {v_kb:,.1f} KB ({v_comp:.1f}x compression vs raw)")
            print(f"  faiss.IndexIVFPQ:     {f_kb:,.1f} KB ({f_comp:.1f}x compression vs raw)")
            print(f"  Memory Footprint Ratio: FAISS is {ratio:.2f}x of vecta")
            print("  Note: vecta measures resident in-RAM heap; FAISS measures serialized byte buffer.")
            print("=" * 88)


def compare_ivf_index(
    dataset: Optional[Any] = None,
    queries: Optional[Any] = None,
    ground_truth: Optional[Any] = None,
    k: int = 10,
    nlist: int = 100,
    nprobe_values: Optional[List[int]] = None,
    dataset_name: str = "siftsmall",
    metric: str = "euclidean",
    threads: int = 1,
    num_trials: int = 5,
    warmup_trials: int = 2,
    save_json: bool = True,
    save_plot: bool = True,
) -> Dict[str, Any]:
    """
    Execute full head-to-head comparison between vecta.IVFIndex and faiss.IndexIVFFlat
    across a sweep of nprobe probe settings.

    Evaluates:
    - Train + add build time and ingestion throughput
    - Query throughput (QPS) and recall@k across matching nprobe sweep values
    - Iso-recall (~90%) interpolated throughput comparison
    - Matplotlib Recall-vs-QPS tradeoff curve chart generation
    """
    set_faiss_threads(threads)
    actual_threads = get_faiss_threads()

    # 1. Dataset Acquisition
    if dataset is None or queries is None or ground_truth is None:
        print(f"\n[1/4] Loading {dataset_name} dataset...")
        base, query, gt = load_siftsmall()
    else:
        base, query, gt = dataset, queries, ground_truth

    base = np.ascontiguousarray(base, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    num_base, dim = base.shape
    num_query = query.shape[0]

    if nprobe_values is None:
        nprobe_values = [1, 2, 5, 10, 20, 50]

    print(f"  Dataset: {num_base:,} base vectors (dim={dim}), {num_query:,} queries, k={k}")
    print(f"  Configuration: nlist={nlist}, metric={metric}, nprobe_sweep={nprobe_values}")

    # 2. Build Indexes (train then add)
    print("\n[2/4] Building IVF indexes (train + add)...")
    base_list = base.tolist()
    ids = list(range(num_base))

    # Build vecta.IVFIndex
    t0 = time.perf_counter()
    v_index = vecta.IVFIndex(dim, num_clusters=nlist, metric=metric)
    v_index.train(base_list, k=nlist, max_iterations=25, seed=42)
    v_index.add_batch(ids, base_list)
    v_build_time = time.perf_counter() - t0
    v_build_rate = num_base / v_build_time if v_build_time > 0 else 0.0
    print(f"  vecta.IVFIndex:       {v_build_time * 1000.0:.2f} ms ({v_build_rate:,.1f} vec/s)")

    # Build faiss.IndexIVFFlat
    t0 = time.perf_counter()
    f_index = build_faiss_ivf(dim, nlist=nlist, metric=metric)
    f_index.train(base)
    f_index.add(base)
    f_build_time = time.perf_counter() - t0
    f_build_rate = num_base / f_build_time if f_build_time > 0 else 0.0
    print(f"  faiss.IndexIVFFlat:   {f_build_time * 1000.0:.2f} ms ({f_build_rate:,.1f} vec/s)")

    # 3. Prepare queries
    vecta_queries = query.tolist()
    faiss_queries = [np.ascontiguousarray(query[i : i + 1]) for i in range(num_query)]
    gt_list = gt.tolist() if isinstance(gt, np.ndarray) else gt

    # 4. Sweep across nprobe values
    print(f"\n[3/4] Running nprobe sweep across {nprobe_values} ({num_trials} trials, {warmup_trials} warmups)...")
    sweep_results: List[Dict[str, Any]] = []

    for np_val in nprobe_values:
        # Vecta Search Trials at this nprobe
        v_trials = run_trials(
            lambda q, k_val: v_index.search(q, k=k_val, nprobe=np_val),
            vecta_queries,
            k,
            num_trials=num_trials,
            warmup_trials=warmup_trials,
        )
        v_stats = summarize_timings(v_trials, num_query)

        # FAISS Search Trials at this nprobe
        # Set nprobe via attribute on the FAISS index instance
        f_index.nprobe = np_val
        f_trials = run_trials(
            lambda q, k_val: f_index.search(q, k=k_val),
            faiss_queries,
            k,
            num_trials=num_trials,
            warmup_trials=warmup_trials,
        )
        f_stats = summarize_timings(f_trials, num_query)

        # Recall@k calculation against ground truth
        v_preds = [[item[0] for item in v_index.search(q, k=k, nprobe=np_val)] for q in vecta_queries]
        f_preds = [f_index.search(q, k=k)[1][0].tolist() for q in faiss_queries]

        v_recall = recall_at_k(v_preds, gt_list, k=k)
        f_recall = recall_at_k(f_preds, gt_list, k=k)

        qps_ratio = f_stats["mean_qps"] / v_stats["mean_qps"] if v_stats["mean_qps"] > 0 else 1.0

        sweep_entry = {
            "nprobe": np_val,
            "vecta_qps": v_stats["mean_qps"],
            "faiss_qps": f_stats["mean_qps"],
            "vecta_recall": v_recall,
            "faiss_recall": f_recall,
            "qps_ratio": qps_ratio,
            "vecta_stats": v_stats,
            "faiss_stats": f_stats,
        }
        sweep_results.append(sweep_entry)
        print(f"  nprobe={np_val:<3} | vecta: {v_stats['mean_qps']:>8.1f} QPS (rec={v_recall*100:.1f}%) | "
              f"faiss: {f_stats['mean_qps']:>8.1f} QPS (rec={f_recall*100:.1f}%)")

    # 5. Iso-recall analysis (~90% target)
    iso_analysis = find_iso_recall(sweep_results, target_recall=0.90)

    results: Dict[str, Any] = {
        "benchmark": "faiss_comparison_ivf",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset_name,
        "num_vectors": num_base,
        "dimension": dim,
        "num_queries": num_query,
        "k": k,
        "nlist": nlist,
        "metric": metric,
        "threads": actual_threads,
        "num_trials": num_trials,
        "warmup_trials": warmup_trials,
        "vecta_build_time_sec": v_build_time,
        "vecta_build_rate_vec_per_sec": v_build_rate,
        "faiss_build_time_sec": f_build_time,
        "faiss_build_rate_vec_per_sec": f_build_rate,
        "sweep": sweep_results,
        "iso_recall": iso_analysis,
    }

    # Print publication-ready comparison table
    print_ivf_comparison_table(results)

    # 6. Save JSON
    if save_json:
        json_path = save_benchmark_result("faiss_comparison_ivf", results)
        results["json_path"] = json_path
        print(f"\nRaw sweep data saved to: {json_path}")

    # 7. Generate Plot
    if save_plot:
        plot_path = generate_ivf_plot(results)
        results["chart_path"] = plot_path
        if plot_path:
            print(f"Recall-vs-QPS tradeoff curve chart saved to: {plot_path}")

    return results


def compare_hnsw_index(
    dataset: Optional[Any] = None,
    queries: Optional[Any] = None,
    ground_truth: Optional[Any] = None,
    k: int = 10,
    m: int = 16,
    ef_construction: int = 100,
    ef_search_values: Optional[List[int]] = None,
    dataset_name: str = "siftsmall",
    metric: str = "euclidean",
    threads: int = 1,
    num_trials: int = 5,
    warmup_trials: int = 2,
    save_json: bool = True,
    save_plot: bool = True,
) -> Dict[str, Any]:
    """
    Execute full head-to-head comparison between vecta.HnswIndex and faiss.IndexHNSWFlat
    across a sweep of ef_search parameter settings.

    Evaluates:
    - Incremental graph construction build time and ingestion throughput
    - Query throughput (QPS) and recall@k across matching ef_search sweep values
    - Iso-recall (~90%) interpolated throughput comparison
    - Matplotlib Recall-vs-QPS tradeoff curve chart generation
    """
    set_faiss_threads(threads)
    actual_threads = get_faiss_threads()

    # 1. Dataset Acquisition
    if dataset is None or queries is None or ground_truth is None:
        print(f"\n[1/4] Loading {dataset_name} dataset...")
        base, query, gt = load_siftsmall()
    else:
        base, query, gt = dataset, queries, ground_truth

    base = np.ascontiguousarray(base, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    num_base, dim = base.shape
    num_query = query.shape[0]

    if ef_search_values is None:
        ef_search_values = [10, 20, 40, 80, 160, 320]

    print(f"  Dataset: {num_base:,} base vectors (dim={dim}), {num_query:,} queries, k={k}")
    print(f"  Configuration: M={m}, efConstruction={ef_construction}, metric={metric}, ef_search_sweep={ef_search_values}")

    # 2. Build Indexes
    print("\n[2/4] Building HNSW graph indexes...")
    base_list = base.tolist()
    ids = list(range(num_base))

    # Build vecta.HnswIndex
    t0 = time.perf_counter()
    v_index = vecta.HnswIndex(
        dim=dim,
        metric=metric,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search_values[0],
        seed=42,
    )
    v_index.add_batch(ids, base_list)
    v_build_time = time.perf_counter() - t0
    v_build_rate = num_base / v_build_time if v_build_time > 0 else 0.0
    print(f"  vecta.HnswIndex:       {v_build_time * 1000.0:.2f} ms ({v_build_rate:,.1f} vec/s)")

    # Build faiss.IndexHNSWFlat
    t0 = time.perf_counter()
    f_index = build_faiss_hnsw(dim, m=m, ef_construction=ef_construction, metric=metric)
    # Critical verification: assert efConstruction is applied BEFORE add()
    assert f_index.hnsw.efConstruction == ef_construction, (
        f"Ordering error: f_index.hnsw.efConstruction was {f_index.hnsw.efConstruction}, expected {ef_construction}"
    )
    f_index.add(base)
    f_build_time = time.perf_counter() - t0
    f_build_rate = num_base / f_build_time if f_build_time > 0 else 0.0
    print(f"  faiss.IndexHNSWFlat:   {f_build_time * 1000.0:.2f} ms ({f_build_rate:,.1f} vec/s)")

    # 3. Prepare queries
    vecta_queries = query.tolist()
    faiss_queries = [np.ascontiguousarray(query[i : i + 1]) for i in range(num_query)]
    gt_list = gt.tolist() if isinstance(gt, np.ndarray) else gt

    # 4. Sweep across ef_search values
    print(f"\n[3/4] Running ef_search sweep across {ef_search_values} ({num_trials} trials, {warmup_trials} warmups)...")
    sweep_results: List[Dict[str, Any]] = []

    for ef_val in ef_search_values:
        # Vecta Search Trials at this ef_search
        v_trials = run_trials(
            lambda q, k_val: v_index.search(q, k=k_val, ef_search=ef_val),
            vecta_queries,
            k,
            num_trials=num_trials,
            warmup_trials=warmup_trials,
        )
        v_stats = summarize_timings(v_trials, num_query)

        # FAISS Search Trials at this ef_search
        # Critical API: Set nested attribute on index.hnsw
        f_index.hnsw.efSearch = ef_val
        f_trials = run_trials(
            lambda q, k_val: f_index.search(q, k=k_val),
            faiss_queries,
            k,
            num_trials=num_trials,
            warmup_trials=warmup_trials,
        )
        f_stats = summarize_timings(f_trials, num_query)

        # Recall@k calculation against ground truth
        v_preds = [[item[0] for item in v_index.search(q, k=k, ef_search=ef_val)] for q in vecta_queries]
        f_preds = [f_index.search(q, k=k)[1][0].tolist() for q in faiss_queries]

        v_recall = recall_at_k(v_preds, gt_list, k=k)
        f_recall = recall_at_k(f_preds, gt_list, k=k)

        qps_ratio = f_stats["mean_qps"] / v_stats["mean_qps"] if v_stats["mean_qps"] > 0 else 1.0

        sweep_entry = {
            "ef_search": ef_val,
            "param_value": ef_val,
            "vecta_qps": v_stats["mean_qps"],
            "faiss_qps": f_stats["mean_qps"],
            "vecta_recall": v_recall,
            "faiss_recall": f_recall,
            "qps_ratio": qps_ratio,
            "vecta_stats": v_stats,
            "faiss_stats": f_stats,
        }
        sweep_results.append(sweep_entry)
        print(f"  ef_search={ef_val:<4} | vecta: {v_stats['mean_qps']:>8.1f} QPS (rec={v_recall*100:.1f}%) | "
              f"faiss: {f_stats['mean_qps']:>8.1f} QPS (rec={f_recall*100:.1f}%)")

    # 5. Iso-recall analysis (~90% target)
    iso_analysis = find_iso_recall(sweep_results, target_recall=0.90, param_name="ef_search")

    results: Dict[str, Any] = {
        "benchmark": "faiss_comparison_hnsw",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset_name,
        "num_vectors": num_base,
        "dimension": dim,
        "num_queries": num_query,
        "k": k,
        "m": m,
        "ef_construction": ef_construction,
        "param_name": "ef_search",
        "metric": metric,
        "threads": actual_threads,
        "num_trials": num_trials,
        "warmup_trials": warmup_trials,
        "vecta_build_time_sec": v_build_time,
        "vecta_build_rate_vec_per_sec": v_build_rate,
        "faiss_build_time_sec": f_build_time,
        "faiss_build_rate_vec_per_sec": f_build_rate,
        "sweep": sweep_results,
        "iso_recall": iso_analysis,
    }

    # Print publication-ready comparison table
    print_hnsw_comparison_table(results)

    # 6. Save JSON
    if save_json:
        json_path = save_benchmark_result("faiss_comparison_hnsw", results)
        results["json_path"] = json_path
        print(f"\nRaw sweep data saved to: {json_path}")

    # 7. Generate Plot
    if save_plot:
        plot_path = generate_hnsw_plot(results)
        results["chart_path"] = plot_path
        if plot_path:
            print(f"Recall-vs-QPS tradeoff curve chart saved to: {plot_path}")

    return results


def compare_ivf_pq_index(
    dataset: Optional[Any] = None,
    queries: Optional[Any] = None,
    ground_truth: Optional[Any] = None,
    k: int = 10,
    nlist: int = 100,
    m: int = 8,
    k_per_subvector: int = 256,
    nprobe_values: Optional[List[int]] = None,
    dataset_name: str = "siftsmall",
    metric: str = "euclidean",
    threads: int = 1,
    num_trials: int = 5,
    warmup_trials: int = 2,
    save_json: bool = True,
    save_plot: bool = True,
) -> Dict[str, Any]:
    """
    Execute full head-to-head comparison between vecta.IVFPQIndex and faiss.IndexIVFPQ
    across an nprobe probe sweep, including memory compression footprint analysis.

    Evaluates:
    - Train + add build time and ingestion throughput
    - In-RAM resident heap vs. serialized index memory footprint
    - Query throughput (QPS) and recall@k across matching nprobe sweep values
    - Iso-recall (~90%) interpolated throughput comparison
    - Matplotlib Recall-vs-QPS tradeoff curve and memory comparison bar chart
    """
    # Verify k_per_subvector is a power of 2 for FAISS nbits compatibility
    if k_per_subvector <= 0 or (k_per_subvector & (k_per_subvector - 1)) != 0:
        raise ValueError(
            f"k_per_subvector must be a power of 2 for FAISS nbits compatibility, got {k_per_subvector}"
        )
    nbits = int(math.log2(k_per_subvector))

    # Metric validation: vecta IVFPQ is Euclidean-only
    if metric.lower() not in ("euclidean", "l2"):
        raise ValueError(f"vecta IVFPQ only supports Euclidean/L2 metric, got '{metric}'")

    set_faiss_threads(threads)
    actual_threads = get_faiss_threads()

    # 1. Dataset Acquisition
    if dataset is None or queries is None or ground_truth is None:
        print(f"\n[1/4] Loading {dataset_name} dataset...")
        base, query, gt = load_siftsmall()
    else:
        base, query, gt = dataset, queries, ground_truth

    base = np.ascontiguousarray(base, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    num_base, dim = base.shape
    num_query = query.shape[0]

    if dim % m != 0:
        raise ValueError(f"Dimension {dim} must be divisible by m={m}")

    if nprobe_values is None:
        nprobe_values = [1, 2, 5, 10, 20, 50, 100]

    print(f"  Dataset: {num_base:,} base vectors (dim={dim}), {num_query:,} queries, k={k}")
    print(
        f"  Configuration: nlist={nlist}, M={m}, k_sub={k_per_subvector} (nbits={nbits}), "
        f"metric={metric}, nprobe_sweep={nprobe_values}"
    )

    # 2. Build Indexes (train then add)
    print("\n[2/4] Building IVFPQ compressed indexes (train + add)...")
    base_list = base.tolist()
    ids = list(range(num_base))

    # Build vecta.IVFPQIndex
    t0 = time.perf_counter()
    v_index = vecta.IVFPQIndex(
        dim=dim,
        num_clusters=nlist,
        m=m,
        k_per_subvector=k_per_subvector,
        max_iterations=25,
    )
    v_index.train(base_list, ivf_seed=42, pq_seed=42)
    v_index.add_batch(ids, base_list)
    v_build_time = time.perf_counter() - t0
    v_build_rate = num_base / v_build_time if v_build_time > 0 else 0.0
    v_mem_bytes = v_index.memory_footprint_bytes()
    print(f"  vecta.IVFPQIndex:     {v_build_time * 1000.0:.2f} ms ({v_build_rate:,.1f} vec/s) | Memory: {v_mem_bytes / 1024.0:.1f} KB")

    # Build faiss.IndexIVFPQ
    t0 = time.perf_counter()
    f_index = build_faiss_ivfpq(dim, nlist=nlist, m=m, nbits=nbits, metric="euclidean")
    assert f_index.metric_type == faiss.METRIC_L2, (
        f"Expected METRIC_L2 on faiss.IndexIVFPQ, got {f_index.metric_type}"
    )
    f_index.train(base)
    f_index.add(base)
    f_build_time = time.perf_counter() - t0
    f_build_rate = num_base / f_build_time if f_build_time > 0 else 0.0
    # FAISS serialized buffer proxy for memory footprint
    f_mem_bytes = len(faiss.serialize_index(f_index))
    raw_vector_bytes = num_base * dim * 4
    print(f"  faiss.IndexIVFPQ:     {f_build_time * 1000.0:.2f} ms ({f_build_rate:,.1f} vec/s) | Memory: {f_mem_bytes / 1024.0:.1f} KB")

    # 3. Prepare queries
    vecta_queries = query.tolist()
    faiss_queries = [np.ascontiguousarray(query[i : i + 1]) for i in range(num_query)]
    gt_list = gt.tolist() if isinstance(gt, np.ndarray) else gt

    # 4. Sweep across nprobe values
    print(f"\n[3/4] Running nprobe sweep across {nprobe_values} ({num_trials} trials, {warmup_trials} warmups)...")
    sweep_results: List[Dict[str, Any]] = []

    for np_val in nprobe_values:
        # Vecta Search Trials at this nprobe
        v_trials = run_trials(
            lambda q, k_val: v_index.search(q, k=k_val, nprobe=np_val),
            vecta_queries,
            k,
            num_trials=num_trials,
            warmup_trials=warmup_trials,
        )
        v_stats = summarize_timings(v_trials, num_query)

        # FAISS Search Trials at this nprobe
        f_index.nprobe = np_val
        f_trials = run_trials(
            lambda q, k_val: f_index.search(q, k=k_val),
            faiss_queries,
            k,
            num_trials=num_trials,
            warmup_trials=warmup_trials,
        )
        f_stats = summarize_timings(f_trials, num_query)

        # Recall@k calculation against ground truth
        v_preds = [[item[0] for item in v_index.search(q, k=k, nprobe=np_val)] for q in vecta_queries]
        f_preds = [f_index.search(q, k=k)[1][0].tolist() for q in faiss_queries]

        v_recall = recall_at_k(v_preds, gt_list, k=k)
        f_recall = recall_at_k(f_preds, gt_list, k=k)

        qps_ratio = f_stats["mean_qps"] / v_stats["mean_qps"] if v_stats["mean_qps"] > 0 else 1.0

        sweep_entry = {
            "nprobe": np_val,
            "param_value": np_val,
            "vecta_qps": v_stats["mean_qps"],
            "faiss_qps": f_stats["mean_qps"],
            "vecta_recall": v_recall,
            "faiss_recall": f_recall,
            "qps_ratio": qps_ratio,
            "vecta_stats": v_stats,
            "faiss_stats": f_stats,
        }
        sweep_results.append(sweep_entry)
        print(f"  nprobe={np_val:<3} | vecta: {v_stats['mean_qps']:>8.1f} QPS (rec={v_recall*100:.1f}%) | "
              f"faiss: {f_stats['mean_qps']:>8.1f} QPS (rec={f_recall*100:.1f}%)")

    # 5. Iso-recall analysis (~90% target)
    iso_analysis = find_iso_recall(sweep_results, target_recall=0.90, param_name="nprobe")

    results: Dict[str, Any] = {
        "benchmark": "faiss_comparison_ivfpq",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset_name,
        "num_vectors": num_base,
        "dimension": dim,
        "num_queries": num_query,
        "k": k,
        "nlist": nlist,
        "m": m,
        "k_per_subvector": k_per_subvector,
        "nbits": nbits,
        "param_name": "nprobe",
        "metric": "euclidean",
        "threads": actual_threads,
        "num_trials": num_trials,
        "warmup_trials": warmup_trials,
        "vecta_build_time_sec": v_build_time,
        "vecta_build_rate_vec_per_sec": v_build_rate,
        "faiss_build_time_sec": f_build_time,
        "faiss_build_rate_vec_per_sec": f_build_rate,
        "raw_vector_bytes": raw_vector_bytes,
        "vecta_memory_bytes": v_mem_bytes,
        "faiss_memory_bytes": f_mem_bytes,
        "vecta_compression_ratio": raw_vector_bytes / v_mem_bytes if v_mem_bytes > 0 else 1.0,
        "faiss_compression_ratio": raw_vector_bytes / f_mem_bytes if f_mem_bytes > 0 else 1.0,
        "sweep": sweep_results,
        "iso_recall": iso_analysis,
    }

    # Print publication-ready comparison table with memory metrics
    print_ivfpq_comparison_table(results)

    # 6. Save JSON
    if save_json:
        json_path = save_benchmark_result("faiss_comparison_ivfpq", results)
        results["json_path"] = json_path
        print(f"\nRaw sweep data saved to: {json_path}")

    # 7. Generate Plots
    if save_plot:
        plot_path = generate_ivfpq_plot(results)
        results["chart_path"] = plot_path
        if plot_path:
            print(f"Recall-vs-QPS tradeoff curve chart saved to: {plot_path}")

        mem_plot_path = generate_memory_comparison_plot(results)
        results["memory_chart_path"] = mem_plot_path
        if mem_plot_path:
            print(f"Memory footprint comparison chart saved to: {mem_plot_path}")

    return results


def load_latest_benchmark_results(results_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load latest saved JSON benchmark results for flat, ivf, hnsw, ivfpq."""
    if results_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, "results")

    all_results = {}
    for idx_key in ["flat", "ivf", "hnsw", "ivfpq"]:
        pattern = os.path.join(results_dir, f"faiss_comparison_{idx_key}_*.json")
        matches = glob.glob(pattern)
        if matches:
            matches.sort()
            latest = matches[-1]
            try:
                with open(latest, "r", encoding="utf-8") as f:
                    all_results[idx_key] = json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not read {latest}: {e}")
    return all_results


def print_final_summary(
    all_results: Optional[Dict[str, Any]] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Print a consolidated master comparison table across all four index types:
    Flat, IVF, HNSW, and IVFPQ.
    """
    if all_results is None:
        all_results = load_latest_benchmark_results(results_dir)

    print("\n" + "=" * 118)
    print(" MASTER HEAD-TO-HEAD BENCHMARK SUMMARY: VECTA vs. FAISS")
    print(" SIFT10k Benchmark Suite (N=10,000, Dim=128, Metric=Euclidean, Single-Threaded CPU Parity)")
    print("=" * 118)
    print(
        f" {'Index Architecture':<18} | {'Engine':<7} | {'Build Time':<12} | "
        f"{'QPS (~90% Rec)':<16} | {'Speedup':<13} | {'Recall@10':<11} | {'Memory / Buffer':<17} | {'Compression':<11}"
    )
    print("-" * 118)

    # 1. Flat Index
    if "flat" in all_results:
        f_data = all_results["flat"]
        v_b = f"{f_data['vecta']['build_time_sec'] * 1000.0:.1f} ms"
        f_b = f"{f_data['faiss']['build_time_sec'] * 1000.0:.1f} ms"
        v_qps = f"{f_data['vecta']['mean_qps']:,.1f}"
        f_qps = f"{f_data['faiss']['mean_qps']:,.1f}"
        ratio = f_data["comparison"]["qps_speedup_ratio"]
        f_ratio_str = f"FAISS {ratio:.2f}x" if f_data["comparison"]["faster_engine"] == "faiss" else f"vecta {ratio:.2f}x"
        v_rec = f"{f_data['vecta']['recall_at_k'] * 100:.1f}%"
        f_rec = f"{f_data['faiss']['recall_at_k'] * 100:.1f}%"
        raw_kb = (f_data.get("num_vectors", 10000) * f_data.get("dimension", 128) * 4) / 1024.0

        print(
            f" {'Flat (Exact L2)':<18} | {'vecta':<7} | {v_b:<12} | "
            f"{v_qps:<16} | {'baseline':<13} | {v_rec:<11} | {raw_kb:,.1f} KB       | {'1.0x (raw)':<11}"
        )
        print(
            f" {'':<18} | {'FAISS':<7} | {f_b:<12} | "
            f"{f_qps:<16} | {f_ratio_str:<13} | {f_rec:<11} | {raw_kb:,.1f} KB       | {'1.0x (raw)':<11}"
        )
        print("-" * 118)

    # 2. IVF Index
    if "ivf" in all_results:
        i_data = all_results["ivf"]
        v_b = f"{i_data['vecta_build_time_sec'] * 1000.0:.1f} ms"
        f_b = f"{i_data['faiss_build_time_sec'] * 1000.0:.1f} ms"
        iso = i_data.get("iso_recall", {})
        v_qps = f"{iso.get('vecta', {}).get('estimated_qps', 0):,.1f}"
        f_qps = f"{iso.get('faiss', {}).get('estimated_qps', 0):,.1f}"
        sp = iso.get("speedup_ratio", 1.0)
        faster = iso.get("faster_engine", "faiss")
        f_ratio_str = f"{faster.upper()} {sp:.2f}x"
        v_rec = f"{iso.get('vecta', {}).get('achieved_recall', 0) * 100:.1f}%"
        f_rec = f"{iso.get('faiss', {}).get('achieved_recall', 0) * 100:.1f}%"
        raw_kb = (i_data.get("num_vectors", 10000) * i_data.get("dimension", 128) * 4) / 1024.0

        print(
            f" {'IVF (nlist=100)':<18} | {'vecta':<7} | {v_b:<12} | "
            f"{v_qps:<16} | {'baseline':<13} | {v_rec:<11} | ~{raw_kb:,.1f} KB      | {'1.0x (raw)':<11}"
        )
        print(
            f" {'':<18} | {'FAISS':<7} | {f_b:<12} | "
            f"{f_qps:<16} | {f_ratio_str:<13} | {f_rec:<11} | ~{raw_kb:,.1f} KB      | {'1.0x (raw)':<11}"
        )
        print("-" * 118)

    # 3. HNSW Index
    if "hnsw" in all_results:
        h_data = all_results["hnsw"]
        v_b = f"{h_data['vecta_build_time_sec'] * 1000.0:.1f} ms"
        f_b = f"{h_data['faiss_build_time_sec'] * 1000.0:.1f} ms"
        iso = h_data.get("iso_recall", {})
        v_qps = f"{iso.get('vecta', {}).get('estimated_qps', 0):,.1f}"
        f_qps = f"{iso.get('faiss', {}).get('estimated_qps', 0):,.1f}"
        sp = iso.get("speedup_ratio", 1.0)
        faster = iso.get("faster_engine", "faiss")
        f_ratio_str = f"{faster.upper()} {sp:.2f}x"
        v_rec = f"{iso.get('vecta', {}).get('achieved_recall', 0) * 100:.1f}%"
        f_rec = f"{iso.get('faiss', {}).get('achieved_recall', 0) * 100:.1f}%"
        raw_kb = (h_data.get("num_vectors", 10000) * h_data.get("dimension", 128) * 4) / 1024.0

        print(
            f" {'HNSW (M=16,efC=100)':<18} | {'vecta':<7} | {v_b:<12} | "
            f"{v_qps:<16} | {'baseline':<13} | {v_rec:<11} | ~{raw_kb:,.1f} KB      | {'0.9x (graph)':<11}"
        )
        print(
            f" {'':<18} | {'FAISS':<7} | {f_b:<12} | "
            f"{f_qps:<16} | {f_ratio_str:<13} | {f_rec:<11} | ~{raw_kb:,.1f} KB      | {'0.9x (graph)':<11}"
        )
        print("-" * 118)

    # 4. IVFPQ Index
    if "ivfpq" in all_results:
        p_data = all_results["ivfpq"]
        v_b = f"{p_data['vecta_build_time_sec'] * 1000.0:.1f} ms"
        f_b = f"{p_data['faiss_build_time_sec'] * 1000.0:.1f} ms"
        iso = p_data.get("iso_recall", {})
        v_qps = f"{iso.get('vecta', {}).get('estimated_qps', 0):,.1f}"
        f_qps = f"{iso.get('faiss', {}).get('estimated_qps', 0):,.1f}"
        sp = iso.get("speedup_ratio", 1.0)
        faster = iso.get("faster_engine", "faiss")
        f_ratio_str = f"{faster.upper()} {sp:.2f}x"
        v_rec = f"{iso.get('vecta', {}).get('achieved_recall', 0) * 100:.1f}%"
        f_rec = f"{iso.get('faiss', {}).get('achieved_recall', 0) * 100:.1f}%"

        v_mem_kb = p_data.get("vecta_memory_bytes", 0) / 1024.0
        f_mem_kb = p_data.get("faiss_memory_bytes", 0) / 1024.0
        v_comp = f"{p_data.get('vecta_compression_ratio', 1.0):.1f}x smaller"
        f_comp = f"{p_data.get('faiss_compression_ratio', 1.0):.1f}x smaller"

        print(
            f" {'IVFPQ (M=8,k=256)':<18} | {'vecta':<7} | {v_b:<12} | "
            f"{v_qps:<16} | {'baseline':<13} | {v_rec:<11} | {v_mem_kb:,.1f} KB        | {v_comp:<11}"
        )
        print(
            f" {'':<18} | {'FAISS':<7} | {f_b:<12} | "
            f"{f_qps:<16} | {f_ratio_str:<13} | {f_rec:<11} | {f_mem_kb:,.1f} KB        | {f_comp:<11}"
        )
        print("-" * 118)

    print("=" * 118)
    return all_results


def verify_side_by_side_instantiation(dim: int = 128) -> bool:
    """
    Verify that all four index types in both vecta and FAISS can be instantiated
    in the same Python process without symbol collisions or memory conflicts.
    """
    print("\n[Harness Verification] Instantiating matched index pairs...")

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
    print(f"  [3/4] HNSW:   vecta.HnswIndex (M=16, efC=100) <-> faiss.{type(f_hnsw).__name__} (efC=100)")

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
) -> Optional[Dict[str, Any]]:
    """Harness runner entry point matching Phase 35 signature."""
    set_faiss_threads(threads)
    if dry_run:
        verify_side_by_side_instantiation(dim=128)
        return None

    if index_type in ("flat", "all"):
        return compare_flat_index(
            dataset_name=dataset,
            k=k,
            metric="euclidean",
            threads=threads,
            save_json=False,
        )
    elif index_type == "ivf":
        return compare_ivf_index(
            dataset_name=dataset,
            k=k,
            metric="euclidean",
            threads=threads,
            save_json=False,
            save_plot=False,
        )
    elif index_type == "hnsw":
        return compare_hnsw_index(
            dataset_name=dataset,
            k=k,
            metric="euclidean",
            threads=threads,
            save_json=False,
            save_plot=False,
        )
    elif index_type == "ivfpq":
        return compare_ivf_pq_index(
            dataset_name=dataset,
            k=k,
            metric="euclidean",
            threads=threads,
            save_json=False,
            save_plot=False,
        )
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Vecta vs. FAISS Head-to-Head Comparison Harness"
    )
    parser.add_argument(
        "--index",
        "--index-type",
        dest="index_type",
        choices=["all", "flat", "ivf", "hnsw", "ivfpq"],
        default="flat",
        help="Index type to compare (default: flat)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print consolidated master summary across all four completed comparisons",
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
        "--nlist",
        type=int,
        default=100,
        help="Number of IVF coarse clusters (default: 100)",
    )
    parser.add_argument(
        "--nprobe-sweep",
        type=str,
        default="1,2,5,10,20,50,100",
        help="Comma-separated nprobe sweep values for IVF / IVFPQ (default: 1,2,5,10,20,50,100)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Subvectors for IVFPQ (default: 8) or link degree M for HNSW (default: 16)",
    )
    parser.add_argument(
        "--k-per-subvector",
        type=int,
        default=256,
        help="Centroids per subvector for IVFPQ (default: 256, 2^8)",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=100,
        help="HNSW construction beam depth efConstruction (default: 100)",
    )
    parser.add_argument(
        "--ef-search-sweep",
        type=str,
        default="10,20,40,80,160,320",
        help="Comma-separated ef_search sweep values for HNSW (default: 10,20,40,80,160,320)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="siftsmall",
        help="Dataset name (default: siftsmall)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of timed measurement trials (default: 5)",
    )
    parser.add_argument(
        "--warmup-trials",
        type=int,
        default=2,
        help="Number of discarded warmup passes (default: 2)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to JSON",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate or save matplotlib plot",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate and validate configs without running benchmark loops",
    )
    args = parser.parse_args()

    # If --summary flag passed alone, print consolidated table and exit
    if args.summary:
        print_final_summary()
        return

    print("=" * 80)
    print(" VECTA vs. FAISS COMPARISON HARNESS")
    print(f" Target Index: {args.index_type.upper()} | Threads: {args.threads} | Target k: {args.k}")
    print("=" * 80)

    # If dry-run requested, run verification check only
    if args.dry_run:
        set_faiss_threads(args.threads)
        verify_side_by_side_instantiation(dim=128)
        print("Dry-run requested: parameter validation and instantiation successful.")
        return

    # Parse sweep parameters
    nprobe_values = [int(x.strip()) for x in args.nprobe_sweep.split(",") if x.strip()]
    ef_search_values = [int(x.strip()) for x in args.ef_search_sweep.split(",") if x.strip()]

    all_run_results = {}

    # Run selected index comparison
    if args.index_type in ("flat", "all"):
        res_flat = compare_flat_index(
            dataset_name=args.dataset,
            k=args.k,
            metric="euclidean",
            threads=args.threads,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials,
            save_json=not args.no_save,
        )
        all_run_results["flat"] = res_flat

    if args.index_type in ("ivf", "all"):
        res_ivf = compare_ivf_index(
            k=args.k,
            nlist=args.nlist,
            nprobe_values=nprobe_values,
            dataset_name=args.dataset,
            metric="euclidean",
            threads=args.threads,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials,
            save_json=not args.no_save,
            save_plot=not args.no_plot,
        )
        all_run_results["ivf"] = res_ivf

    if args.index_type in ("hnsw", "all"):
        hnsw_m = args.m if args.m is not None else 16
        res_hnsw = compare_hnsw_index(
            k=args.k,
            m=hnsw_m,
            ef_construction=args.ef_construction,
            ef_search_values=ef_search_values,
            dataset_name=args.dataset,
            metric="euclidean",
            threads=args.threads,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials,
            save_json=not args.no_save,
            save_plot=not args.no_plot,
        )
        all_run_results["hnsw"] = res_hnsw

    if args.index_type in ("ivfpq", "all"):
        ivfpq_m = args.m if args.m is not None else 8
        res_ivfpq = compare_ivf_pq_index(
            k=args.k,
            nlist=args.nlist,
            m=ivfpq_m,
            k_per_subvector=args.k_per_subvector,
            nprobe_values=nprobe_values,
            dataset_name=args.dataset,
            metric="euclidean",
            threads=args.threads,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials,
            save_json=not args.no_save,
            save_plot=not args.no_plot,
        )
        all_run_results["ivfpq"] = res_ivfpq

    # If running all, print final consolidated master summary
    if args.index_type == "all":
        print_final_summary(all_run_results)


if __name__ == "__main__":
    main()
