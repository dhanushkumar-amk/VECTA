# 🧵 X / Twitter Launch Thread: Vecta

---

### Tweet 1 (The Hook)
I spent the last 40 phases building a production-grade vector database engine from scratch in Rust, with PyO3 Python bindings.

Meet **Vecta** ⚡

Flat, IVF, HNSW, and IVFPQ — complete with WAL crash recovery, mmap, metadata filtering, and sharding.

Then, I benchmarked it head-to-head against Meta FAISS.

Here are the honest numbers 👇🧵 (1/7)

---

### Tweet 2 (The Architecture)
Most vector DB tutorials stop at brute-force cosine similarity.

Vecta implements all four foundational ANN architectures from first principles:
1️⃣ Flat: Exact L2/Cosine/IP (100% recall baseline)
2️⃣ IVF: Lloyd's k-means + inverted list multi-probing
3️⃣ HNSW: Hierarchical skip-graph with beam search
4️⃣ IVFPQ: Product Quantization + ADC table lookups

Zero external search dependencies. Pure Rust. (2/7)

---

### Tweet 3 (The Benchmark Setup)
A benchmark is useless if it isn't apples-to-apples:
- Standard SIFT10k dataset (N=10,000, Dim=128, Euclidean metric)
- Single-threaded CPU parity (OMP_NUM_THREADS=1, threads=1)
- Identical k=10 ground truth
- Monotonic sweeps across nprobe (IVF, IVFPQ) and ef_search (HNSW)
- Multi-trial timing with thermal warmups discarded

No synthetic vanity metrics. (3/7)

---

### Tweet 4 (The Results: Wins & Losses vs FAISS)
The master comparison table:

Flat (Exact):
• Vecta: 1,413 QPS
• FAISS: 5,573 QPS (FAISS 3.9x)

IVF (nlist=100, ~90% recall):
• Vecta: 17,849 QPS
• FAISS: 69,957 QPS (FAISS 3.9x)

HNSW (M=16, efC=100):
• Vecta: 5,988 QPS
• FAISS: 59,721 QPS (FAISS 10.0x)

IVFPQ (M=8, k=256):
• Vecta: 25,434 QPS
• FAISS: 8,978 QPS (Vecta 2.8x at ~90% boundary)

[Attach: Master Benchmark Summary Table / HNSW & IVF Recall vs QPS Plots] (4/7)

---

### Tweet 5 (The Big Win: IVFPQ Memory Compression)
Why use IVFPQ if HNSW has higher recall? **Compression.**

Raw float32 vectors for SIFT10k occupy 5.0 MB.
• Vecta IVFPQ resident in-RAM footprint: 256.1 KB
• Compression ratio: **19.5x smaller than raw!**
• FAISS serialized buffer: 335.2 KB (14.9x)

At nprobe=50, Vecta matches FAISS throughput (16.2k QPS), and at nprobe=100, Vecta's cache-friendly ADC lookup loop actually edges ahead (10.1k vs 9.0k QPS).

[Attach: ivfpq_memory_comparison.png & ivfpq_recall_vs_qps.png] (5/7)

---

### Tweet 6 (Where FAISS Still Wins & Why)
Engineering honesty matters:
1. FAISS's HNSW is ~10x faster because of aggressive software prefetching (`_mm_prefetch`), contiguous flat graph layouts, and AVX2 unrolled distance kernels.
2. Centroid training in FAISS is 7x–40x faster due to OpenMP multi-threading and optimized clustering routines.

Knowing *why* decades-optimized C++ code is faster is the best part of systems engineering. (6/7)

---

### Tweet 7 (The Journey & Code)
40 phases: from writing the first SIMD distance dot-product to handling Python GIL release for multithreaded queries, WAL replay, and sharding coordinators.

The entire codebase is open-source under MIT:
⭐ Code: https://github.com/dhanushkumar-amk/VECTA
📦 Release: v0.1.0 with CI matrix wheels (Linux, macOS, Windows)

If you're interested in systems programming, vector search, or Rust/PyO3, give it a star! 🦀🚀 (7/7)
