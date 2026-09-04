# Vecta vs. FAISS Benchmarking Methodology & Protocol

*Phase 35: Standardized Comparison Framework for Vector Search Engines*

---

## 1. Executive Summary & Objective

The goal of this benchmarking harness is to conduct a **fair, scientifically rigorous, and peer-reviewable head-to-head comparison** between `vecta` (written from scratch in Rust with PyO3 bindings) and Facebook AI Similarity Search (`faiss-cpu`, written in C++ with OpenMP and BLAS/LAPACK optimizations).

In machine learning and vector systems engineering, superficial benchmarks often produce misleading claims by quietly varying hardware baselines, thread configurations, compiler flags, or approximate search hyperparameters. This document details the exact protocol, hardware specifications, parameter conversions, and threading policies governing all subsequent comparison phases (Phase 36 through Phase 50).

---

## 2. Hardware & Environment Baseline

All comparison numbers in this project are captured on a single fixed workstation to eliminate cross-machine variance, virtualization noise, and NUMA memory throttling:

| Specification | Workstation Value |
| :--- | :--- |
| **CPU Model** | AMD Ryzen 7 5800X 8-Core Processor |
| **Base / Boost Frequency** | 3.80 GHz base clock / up to 4.70 GHz boost |
| **Physical Cores** | 8 physical CPU cores |
| **Logical Processors (SMT)** | 16 threads |
| **L1 / L2 / L3 Cache** | 512 KB L1 / 4 MB L2 / 32 MB unified L3 Cache |
| **System Memory (RAM)** | 32.0 GB DDR4 (31.93 GB usable) |
| **Host Operating System** | Microsoft Windows 11 Pro 64-bit |
| **Python Environment** | CPython 3.14.7 64-bit |
| **Rust Toolchain** | Rustc 1.85+ (stable-x86_64-pc-windows-msvc) |
| **FAISS Package** | `faiss-cpu` version 1.15.0 (compiled for AMD64) |
| **Compiler Optimization** | `vecta`: `--release` (opt-level=3, LTO=true, codegen-units=1) |

> [!IMPORTANT]
> **Why CPU-Only Comparison (`faiss-cpu`)?**
> `vecta` is an in-memory, CPU-based vector database engine designed for commodity host hardware. Comparing `vecta` against GPU-accelerated FAISS (`faiss-gpu` using CUDA) would represent a fundamentally mismatched hardware baseline. Every test herein compares CPU instructions to CPU instructions on identical AMD Ryzen silicon.

---

## 3. Benchmark Datasets & Workloads

We benchmark against the canonical **SIFT (Scale-Invariant Feature Transform)** datasets established by Hervé Jégou et al. (INRIA), universally accepted as the standard vector search benchmark:

### Primary Dataset: SIFT-Small (`siftsmall`)
- **Base Vectors ($N$)**: 10,000 vectors
- **Dimensionality ($D$)**: 128 dimensions (`float32`)
- **Query Vectors ($Q$)**: 100 query vectors
- **Ground Truth**: Exact Euclidean 100 nearest neighbors per query pre-computed via brute force
- **Memory Footprint**: $10,000 \times 128 \times 4\text{ bytes} \approx 5.12\text{ MB}$ raw floats

### Scale Dataset: SIFT-1M (`sift1m`)
- **Base Vectors ($N$)**: 1,000,000 vectors
- **Dimensionality ($D$)**: 128 dimensions (`float32`)
- **Query Vectors ($Q$)**: 10,000 query vectors
- **Ground Truth**: Exact Euclidean 100 nearest neighbors per query
- **Memory Footprint**: $1,000,000 \times 128 \times 4\text{ bytes} \approx 512\text{ MB}$ raw floats

Target recall values are evaluated at:
- $k = 10$ (standard real-world retrieval threshold for RAG and semantic search)
- $k = 100$ (broad candidate retrieval)

---

## 4. Matched Architectural Configurations

To ensure parity, every index type in `vecta` is mapped to its exact structural equivalent in `FAISS`:

| Index Category | `vecta` Class | `FAISS` Class | Matched Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Flat** | `vecta.FlatIndex` | `faiss.IndexFlatL2` | Dimensionality $D=128$, Metric = Euclidean / L2 |
| **IVF** | `vecta.IVFIndex` | `faiss.IndexIVFFlat` | Centroid count $\text{nlist}=100$, coarse quantizer = FlatL2 |
| **HNSW** | `vecta.HnswIndex` | `faiss.IndexHNSWFlat` | Degree $M=16$, build depth $\text{efConstruction}=100$ |
| **IVFPQ** | `vecta.IVFPQIndex` | `faiss.IndexIVFPQ` | $\text{nlist}=100$, sub-vectors $m=8$, codebook $k=256 \iff \text{nbits}=8$ |

### Unit Conversion Note: `k_per_subvector` vs. `nbits`
- `vecta.IVFPQIndex` explicitly configures the number of centroids per sub-space via `k_per_subvector = 256`.
- `faiss.IndexIVFPQ` configures this via code bit-width `nbits = 8`.
- Since $2^{\text{nbits}} = 2^8 = 256$, these represent mathematically identical codebook resolutions ($256$ centroids per subvector).

### Parameter Configuration API Differences
- In `vecta`, search-time depth is passed dynamically to the query:
  ```python
  vecta_ivf.search(query, k=10, nprobe=16)
  vecta_hnsw.search(query, k=10, ef_search=80)
  ```
- In `FAISS`, these are configured as index attributes prior to calling `search`:
  ```python
  faiss_ivf.nprobe = 16
  faiss_ivf.search(query, k=10)

  faiss_hnsw.hnsw.efSearch = 80
  faiss_hnsw.search(query, k=10)
  ```
  The benchmark harness wraps these differences cleanly so callers execute queries identically.

---

## 5. Threading & Concurrency Policy

FAISS and `vecta` have distinct architectural threading defaults:
- **FAISS CPU** links against OpenMP and by default fans single-query distance loops across **all 16 logical cores**.
- **`vecta` core indexes** (`FlatIndex`, `IVFIndex`, `HnswIndex`, `IVFPQIndex`) are designed as single-threaded core data structures, delegating concurrency to explicit multi-tenant wrappers ([`ConcurrentFlatIndex`](file:///c:/Users/dhanu/Documents/PROJECTS/OWN%20VECTOR%20DB/vecta/src/core/concurrent_index.rs)) or horizontal partitioning ([`ShardedFlatIndex`](file:///c:/Users/dhanu/Documents/PROJECTS/OWN%20VECTOR%20DB/vecta/src/core/sharded_index.rs)).

> [!IMPORTANT]
> **Explicit Decision: Two-Tier Evaluation**
> Running FAISS multi-threaded against single-threaded `vecta` would misattribute OpenMP multi-core speedup to FAISS's algorithmic implementation. Conversely, testing only single-threaded would overlook FAISS's multi-core optimizations. We therefore measure and report **both tiers explicitly**:
>
> 1. **Tier 1: Single-Threaded Algorithmic Parity ($T = 1$)**:
>    - Set `faiss.omp_set_num_threads(1)` and run `vecta` single-threaded.
>    - Measures pure single-core algorithmic efficiency, SIMD vectorization, cache line locality, and heap traversal overhead.
>
> 2. **Tier 2: Multi-Threaded Scalability ($T = 16$)**:
>    - Set `faiss.omp_set_num_threads(16)` to utilize all available workstation threads.
>    - Compare against `vecta.ShardedFlatIndex(..., parallel=True)` and concurrent query workers (`threading.Thread` with GIL release).
>    - Measures multi-core throughput and thread-scaling efficiency.

---

## 6. Measurement Methodology & Metrics

To guarantee statistical reliability and isolate execution variance, benchmarks adhere to the following protocol:

### A. Warmup Discipline
- Before timing begins, the benchmark executes **2 complete dry-run passes** over the query dataset.
- Dry-run passes populate CPU L1/L2/L3 caches, warm memory controllers, and trigger any dynamic library loading overhead.
- All dry-run execution timings are strictly discarded from final metrics.

### B. Statistical Repetition
- Search benchmarks execute across $M = 5$ independent recorded trials.
- Latency is recorded per query using high-resolution monotonic timers (`time.perf_counter()`).
- Metrics reported:
  - **Mean Latency (ms)**: Arithmetic mean over all queries.
  - **Percentiles**: p50 (median), p95, p99 tail latency.
  - **Throughput (QPS)**: $\text{QPS} = \frac{\text{Total Queries}}{\text{Total Elapsed Time (seconds)}}$.

### C. Ground Truth & Recall Calculation
Recall@k measures the fraction of true top-$k$ Euclidean nearest neighbors captured in the retrieved candidate list:

$$\text{Recall@k} = \frac{1}{Q} \sum_{q=1}^Q \frac{|\mathcal{R}_k(q) \cap \mathcal{G}_k(q)|}{k}$$

Where $\mathcal{R}_k(q)$ is the retrieved set of candidate IDs, and $\mathcal{G}_k(q)$ is the ground-truth nearest neighbor set.

### D. Memory Footprint Measurement
Memory consumption is measured in two ways:
1. **Mathematical Theoretical Footprint**: Raw vector buffers + centroid matrices + graph link pointers + quantized code arrays.
2. **Process Resident Set Size (RSS)**: Delta in OS process memory before and after index population, measured via Windows system counters.

---

## 7. Known Asymmetries & Honest Technical Caveats

We document every known divergence honestly rather than presenting numbers as more directly comparable than they are:

1. **Metric Support**:
   - `vecta`'s `IVFPQIndex` is Euclidean-only (per Phase 23 design). FAISS supports both Inner Product and L2 for IVFPQ.
2. **K-Means Initialization**:
   - `vecta` implements k-means++ centroid initialization with deterministic seeding.
   - FAISS uses random subset initialization with internal iteration bounds. As a result, trained Voronoi centroids may differ slightly even on the same dataset, leading to minor variations in recall curves at low `nprobe`.
3. **Graph Construction Heuristics in HNSW**:
   - `vecta` implements standard HNSW neighbor selection with distance ranking.
   - FAISS includes optional heuristic neighbor pruning (shrinking neighbors to preserve directional diversity). For fair comparison, we compare base HNSW topologies.
4. **SIMD Vectorization**:
   - FAISS utilizes hand-written AVX2 / AVX-512 intrinsics in C++.
   - `vecta` uses contiguous memory buffers (`VectorBatch`) compiled with LLVM auto-vectorization (`target-cpu=native` / opt-level=3).
5. **Distance Metrics (Squared L2 vs. Linear Euclidean)**:
   - `FAISS`'s `IndexFlatL2` and other L2 indexes compute and return **squared Euclidean distances** ($\sum (x_i - y_i)^2$) to eliminate square root computation in distance loops.
   - `vecta`'s `Metric::Euclidean` computes and returns the **standard Euclidean distance** ($\sqrt{\sum (x_i - y_i)^2}$).
   - Because $f(x) = x^2$ is strictly monotonic for $x \ge 0$ ($d_1 < d_2 \iff d_1^2 < d_2^2$), the resulting nearest neighbor candidate order and recall@k are **100% identical**. However, direct score comparisons require taking $\sqrt{\text{faiss\_dist}}$.
