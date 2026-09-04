# Contributing to Vecta

Thank you for your interest in contributing to **Vecta**! This guide details how to set up your local development environment, build the Rust-Python extension, run test suites, and execute benchmarks.

---

## Prerequisites

- **Rust**: Stable toolchain (1.75+ recommended) installed via [rustup](https://rustup.rs).
- **Python**: Python 3.10 to 3.14 with a virtual environment (`venv`).
- **Maturin**: Build tool for PyO3-based Rust extensions (`pip install maturin`).
- **FAISS (Optional)**: Needed only if running the `faiss_comparison` benchmark suite (`pip install faiss-cpu`).

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dhanushkumar-amk/VECTA.git
   cd VECTA
   ```

2. **Set up Python virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install maturin pytest numpy matplotlib requests
   ```

4. **Build and install Vecta in development mode**:
   ```bash
   # Debug build (faster compilation):
   maturin develop

   # Release build (with SIMD optimizations and inlining):
   maturin develop --release
   ```

---

## Running Tests

### 1. Pure Rust Test Suite
Runs all unit tests, property checks, and core index validation in Rust:
```bash
cargo test
```

### 2. Python Integration & Binding Tests
Runs all end-to-end Python binding tests across all index types, persistence, concurrency, and sharding:
```bash
pytest tests/python/ -v
```

### 3. Comparison Runner Tests
Verifies the FAISS comparison harness and statistical summarizers:
```bash
pytest tests/python/test_run_comparison.py -v
```

---

## Code Quality & Documentation

Before opening a pull request or tagging releases, verify formatting, linter clean-passes, and doc generation:

```bash
# Check Rust formatting
cargo fmt --check

# Run Clippy linter
cargo clippy -- -D warnings

# Build Rustdoc with zero warnings
cargo doc --no-deps
```

---

## Running Benchmarks

### Standalone Index Benchmarks
```bash
# Exact search baseline:
python benchmarks/bench_flat_index.py

# Inverted file index:
python benchmarks/bench_ivf_index.py

# HNSW graph:
python benchmarks/bench_hnsw_index.py

# IVFPQ compressed index:
python benchmarks/bench_ivf_pq_index.py

# Horizontal sharding:
python benchmarks/bench_sharded_index.py
```

### Head-to-Head FAISS Comparison Suite
```bash
# Compare individual index architectures:
python benchmarks/faiss_comparison/run_comparison.py --index flat
python benchmarks/faiss_comparison/run_comparison.py --index ivf
python benchmarks/faiss_comparison/run_comparison.py --index hnsw
python benchmarks/faiss_comparison/run_comparison.py --index ivfpq

# Print the consolidated 4-architecture master summary table:
python benchmarks/faiss_comparison/run_comparison.py --summary
```
