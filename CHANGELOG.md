# Changelog & 40-Phase Engineering Journey

All notable milestones and architectural developments for the **Vecta** vector database engine are documented below across its 40-phase construction.

---

## [v0.1.0] - Initial Release (The 40-Phase Capstone)

### Phase 1–3: Project Architecture & Tooling Foundation
- **Phase 1**: Initialized repository structure with pure Rust core (`src/core`), PyO3 bridge (`src/python.rs`), and Cargo/Maturin build configurations.
- **Phase 2**: Established smoke test harness verifying bidirectional Rust-Python FFI.
- **Phase 3**: Configured GitHub Actions CI matrix builds (Linux, macOS, Windows) and automated release wheel pipeline.

### Phase 4–8: Exact Vector Search (FlatIndex)
- **Phase 4**: Implemented core vector similarity metrics in Rust: Euclidean (L2), Cosine Similarity, and Dot Product.
- **Phase 5**: Implemented contiguous `VectorBatch` storage layout and min-heap / max-heap bounded top-k selection.
- **Phase 6**: Completed pure Rust `FlatIndex` brute-force search engine.
- **Phase 7**: Exposed `vecta.FlatIndex` to Python via PyO3 with NumPy array buffer protocol.
- **Phase 8**: Benchmarked `FlatIndex` on SIFT10k dataset ($N=10,000, D=128$), establishing 100% recall baseline.

### Phase 9–14: Coarse Quantization (IVFIndex)
- **Phase 9**: Implemented Lloyd's k-means clustering in Rust with k-means++ initialization.
- **Phase 10**: Designed inverted list data structures mapping cluster centroids to vector postings.
- **Phase 11**: Implemented multi-cluster probing (`nprobe`) search in pure Rust.
- **Phase 12**: Completed train, add, and search lifecycles for `IVFIndex`.
- **Phase 13**: Exposed `vecta.IVFIndex` to Python with clustering seed control and batch insertion.
- **Phase 14**: Benchmarked `IVFIndex` across `nprobe` sweep on SIFT10k, demonstrating the recall/speed tradeoff.

### Phase 15–19: Graph-Based Search (HnswIndex)
- **Phase 15**: Built multi-layer hierarchical skip-graph representation in Rust with geometric level distribution.
- **Phase 16**: Implemented greedy layer traversal and beam-search heuristics.
- **Phase 17**: Completed incremental insertion with `ef_construction` neighbor selection and edge pruning.
- **Phase 18**: Exposed `vecta.HnswIndex` to Python with dynamic `ef_search` query parameterization.
- **Phase 19**: Benchmarked `HnswIndex` on SIFT10k, achieving $>89\%$ recall at $>28,000$ QPS.

### Phase 20–24: Product Quantization (IVFPQIndex)
- **Phase 20**: Implemented subvector space decomposition and independent codebook training in Rust.
- **Phase 21**: Implemented Asymmetric Distance Computation (ADC) with precomputed query lookup tables.
- **Phase 22**: Built hybrid `IVFPQIndex` combining coarse centroid inverted lists with fine PQ quantization.
- **Phase 23**: Added resident memory footprint reporting (`memory_footprint_bytes`).
- **Phase 24**: Exposed `vecta.IVFPQIndex` to Python and validated up to $19.5\times$ memory compression on SIFT10k.

### Phase 25–26: Persistence & Zero-Copy Memory Mapping
- **Phase 25**: Implemented compact binary serialization (`save` / `load`) with magic bytes, versioning, and CRC validation for all four index types.
- **Phase 26**: Implemented zero-copy memory-mapped search (`mmap`) for instant cold-start index access without heap duplication.

### Phase 27–28: Write-Ahead Logging (WAL) & Crash Recovery
- **Phase 27**: Implemented append-only Write-Ahead Log in Rust with transactional record boundaries and checksum verification.
- **Phase 28**: Integrated WAL replay into index recovery, guaranteeing durability against process crashes.

### Phase 29–30: Metadata Filtering
- **Phase 29**: Built decoupled `MetadataStore` with support for integer, float, string, and boolean attributes, along with composable filter expressions (`Eq`, `Gt`, `Lt`, `And`, `Or`, `Not`).
- **Phase 30**: Exposed metadata filtering to Python with an ergonomic dictionary mini-syntax for post-filtered top-k queries.

### Phase 31–32: Concurrency & GIL Management
- **Phase 31**: Built `ConcurrentFlatIndex` in Rust with `RwLock` synchronizing multiple simultaneous readers and serialized writers.
- **Phase 32**: Exposed `ConcurrentFlatIndex` to Python with explicit GIL release (`Python::allow_threads`), unlocking true OS-thread parallelism in Python.

### Phase 33–34: Horizontal Sharding
- **Phase 33**: Built `ShardedFlatIndex` coordinator in Rust with deterministic hash-based vector routing, query fan-out across shards, and global top-k candidate merging.
- **Phase 34**: Exposed `ShardedFlatIndex` to Python, enabling multi-shard partitioned search within a single process.

### Phase 35–39: Rigorous Head-to-Head FAISS Benchmarking Suite
- **Phase 35**: Established fair benchmarking methodology (identical datasets, queries, ground truth, $k$, metrics, and single-threaded CPU parity).
- **Phase 36**: Flat vs. `faiss.IndexFlatL2` comparison, proving 100% exact recall on both engines.
- **Phase 37**: IVF vs. `faiss.IndexIVFFlat` comparison across `nprobe` sweep with iso-recall interpolation.
- **Phase 38**: HNSW vs. `faiss.IndexHNSWFlat` comparison across `ef_search` sweep with tradeoff curve plotting.
- **Phase 39**: IVFPQ vs. `faiss.IndexIVFPQ` comparison across `nprobe` sweep, memory footprint & compression analysis, and consolidated master benchmark summary.

### Phase 40: Public Packaging, Documentation & v0.1.0 Release
- Comprehensive documentation rewrite, runnable quickstarts for all 4 index types, honest performance reporting against FAISS, zero-warning Rustdoc audit, contributing guidelines, and v0.1.0 release tagging.
