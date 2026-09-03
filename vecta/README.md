# vecta

A production-grade vector database built from scratch in Rust, benchmarked against FAISS.

## Architecture

```
Python layer  →  PyO3 bindings (src/python.rs)  →  Core engine (src/core/*)
```

The core engine is pure Rust with zero Python awareness, making it independently testable and portable to CLI, C ABI, or other language bindings.

## Quick Start

```bash
# Build and install into a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
maturin develop --release

# Verify
python -c "import vecta; print(vecta.hello_vecta())"
# → vecta engine initialized
```

## Development

```bash
cargo test          # Run core engine tests (no Python needed)
cargo build         # Build the Rust library
maturin develop     # Build + install into active venv (debug)
maturin build -r    # Build a release wheel in target/wheels/
```
