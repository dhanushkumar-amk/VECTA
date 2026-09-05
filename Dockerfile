# Stage 1: Build binary using official Rust image
FROM rust:slim AS builder

WORKDIR /app

# Install build dependencies required for compiling PyO3 bindings & OpenSSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    python3 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy full source tree
COPY . .

# Build the standalone REST API server binary in release mode
RUN cargo build --release --bin vecta-server

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim AS runtime

# Install minimal runtime certificates and SSL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled binary from builder stage
COPY --from=builder /app/target/release/vecta-server /usr/local/bin/vecta-server

# Expose standard REST API port
EXPOSE 6333

# Create persistent storage directory mount point
VOLUME /data

# Default environment configuration
ENV VECTA_PORT=6333
ENV VECTA_DATA_DIR=/data

# Execute binary
ENTRYPOINT ["vecta-server"]
