"use client";

import { useState, useEffect, useRef } from "react";

/* ───────────────── Navbar ───────────────── */
function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const links = [
    { label: "Features", href: "#features" },
    { label: "Benchmarks", href: "#benchmarks" },
    { label: "Architecture", href: "#architecture" },
    { label: "Quickstart", href: "#quickstart" },
    { label: "Docs", href: "/docs" },
  ];

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled
          ? "glass-strong shadow-lg shadow-black/20"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <a href="#" className="flex items-center gap-2.5 group">
          <span className="text-2xl">⚡</span>
          <span className="text-xl font-bold gradient-text-static tracking-tight">
            Vecta
          </span>
        </a>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-8">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors duration-300 relative after:absolute after:bottom-[-4px] after:left-0 after:w-0 after:h-[2px] after:bg-[var(--color-primary-light)] after:transition-all after:duration-300 hover:after:w-full"
            >
              {l.label}
            </a>
          ))}
          <a
            href="https://github.com/dhanushkumar-amk/VECTA"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--color-border)] text-sm text-[var(--color-text-secondary)] hover:border-[var(--color-primary)] hover:text-[var(--color-text-primary)] transition-all duration-300"
          >
            <GithubIcon />
            GitHub
          </a>
        </div>

        {/* Mobile toggle */}
        <button
          className="md:hidden text-[var(--color-text-secondary)]"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
            {menuOpen ? (
              <path d="M6 6l12 12M6 18L18 6" />
            ) : (
              <path d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden glass-strong border-t border-[var(--color-border)] px-6 py-4 space-y-3">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              onClick={() => setMenuOpen(false)}
              className="block text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
            >
              {l.label}
            </a>
          ))}
        </div>
      )}
    </nav>
  );
}

/* ───────────────── Hero ───────────────── */
function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20 noise-overlay">
      {/* Background orbs */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-[15%] left-[10%] w-[500px] h-[500px] rounded-full bg-[var(--color-primary)]/[0.07] blur-[120px] animate-pulse-glow" />
        <div className="absolute bottom-[10%] right-[10%] w-[400px] h-[400px] rounded-full bg-[var(--color-accent)]/[0.06] blur-[100px] animate-pulse-glow" style={{ animationDelay: "1.5s" }} />
        <div className="absolute top-[50%] left-[50%] w-[300px] h-[300px] rounded-full bg-[var(--color-primary-dark)]/[0.05] blur-[80px] animate-pulse-glow" style={{ animationDelay: "3s" }} />
      </div>

      {/* Grid pattern */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(99,102,241,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.3) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
        {/* Badge */}
        <div className="animate-fade-in-up inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-[var(--color-border)] bg-[var(--color-surface-card)]/60 mb-8">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs text-[var(--color-text-secondary)] font-medium tracking-wide uppercase">
            Open Source · MIT License
          </span>
        </div>

        {/* Title */}
        <h1 className="animate-fade-in-up-delay-1 text-5xl sm:text-6xl lg:text-7xl font-extrabold leading-[1.1] tracking-tight mb-6">
          Vector Search at{" "}
          <span className="gradient-text">Warp Speed</span>
        </h1>

        {/* Subtitle */}
        <p className="animate-fade-in-up-delay-2 text-lg sm:text-xl text-[var(--color-text-secondary)] max-w-2xl mx-auto mb-10 leading-relaxed">
          A production-grade vector database engine built from scratch in{" "}
          <span className="text-orange-400 font-semibold">pure Rust</span>.
          Four index architectures, Python bindings, REST API, and LangChain integration.
        </p>

        {/* CTA Buttons */}
        <div className="animate-fade-in-up-delay-3 flex flex-col sm:flex-row items-center justify-center gap-4">
          <a
            href="#quickstart"
            className="group relative px-7 py-3.5 rounded-xl font-semibold text-white bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-primary-dark)] shadow-lg shadow-[var(--color-primary)]/25 hover:shadow-[var(--color-primary)]/40 transition-all duration-300 hover:scale-[1.03]"
          >
            <span className="relative z-10 flex items-center gap-2">
              Get Started
              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" className="transition-transform duration-300 group-hover:translate-x-1">
                <path d="M3 8h10M9 4l4 4-4 4" />
              </svg>
            </span>
          </a>
          <a
            href="https://github.com/dhanushkumar-amk/VECTA"
            target="_blank"
            rel="noopener noreferrer"
            className="px-7 py-3.5 rounded-xl font-semibold border border-[var(--color-border)] text-[var(--color-text-primary)] hover:border-[var(--color-primary)] hover:bg-[var(--color-surface-hover)] transition-all duration-300"
          >
            <span className="flex items-center gap-2">
              <GithubIcon />
              View on GitHub
            </span>
          </a>
        </div>

        {/* Install command */}
        <div className="animate-fade-in-up-delay-3 mt-12">
          <CodeCopy text="pip install maturin && maturin develop --release" />
        </div>

        {/* Hero stats */}
        <div className="mt-16 grid grid-cols-2 sm:grid-cols-4 gap-6 animate-fade-in-up-delay-3">
          {[
            { value: "4", label: "Index Architectures" },
            { value: "45.7k", label: "QPS Peak" },
            { value: "19.5×", label: "Compression" },
            { value: "100%", label: "Pure Rust" },
          ].map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-2xl sm:text-3xl font-bold gradient-text-static">{s.value}</div>
              <div className="text-xs text-[var(--color-text-muted)] mt-1 uppercase tracking-wider">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ───────────────── Features ───────────────── */
function FeaturesSection() {
  const features = [
    {
      icon: "🧠",
      title: "4 Index Architectures",
      description:
        "Flat, IVF, HNSW, and IVFPQ — each built from first principles. Choose exact search, graph-based speed, or 19× compressed quantization.",
      tags: ["Flat", "IVF", "HNSW", "IVFPQ"],
    },
    {
      icon: "🦀",
      title: "Pure Rust Core",
      description:
        "Zero external C/C++ dependencies. The entire engine — k-means clustering, graph traversal, product quantization — is hand-written Rust.",
      tags: ["Memory Safe", "Zero Copy", "SIMD-Ready"],
    },
    {
      icon: "🐍",
      title: "Python Bindings (PyO3)",
      description:
        "Import as a native Python module with GIL-released concurrency. Sub-microsecond call latency via direct FFI.",
      tags: ["NumPy", "PyO3", "GIL-Free"],
    },
    {
      icon: "🌐",
      title: "REST API Server",
      description:
        "Production Axum + Tokio HTTP server with Bearer auth, WAL crash durability, and interactive Swagger UI at /docs.",
      tags: ["Axum", "Tokio", "OpenAPI 3.0"],
    },
    {
      icon: "🦜",
      title: "LangChain Integration",
      description:
        "First-class VectaVectorStore implementing LangChain's VectorStore interface. Drop into any RAG pipeline instantly.",
      tags: ["RAG", "Retriever", "Embeddings"],
    },
    {
      icon: "🐳",
      title: "Docker Ready",
      description:
        "One-command deployment with Docker or docker-compose. Persistent volumes, environment config, and fly.io ready.",
      tags: ["Docker", "Compose", "Fly.io"],
    },
  ];

  return (
    <section id="features" className="relative py-28 px-6 overflow-hidden">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[1px] bg-gradient-to-r from-transparent via-[var(--color-primary)]/30 to-transparent" />

      <div className="max-w-7xl mx-auto">
        <SectionHeader
          eyebrow="Core Capabilities"
          title="Built for Real Workloads"
          description="Every component is purpose-built for production vector search — no wrappers, no compromises."
        />

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-16">
          {features.map((f, i) => (
            <div
              key={f.title}
              className="hover-card glass rounded-2xl p-7 group"
              style={{ animationDelay: `${i * 0.1}s` }}
            >
              <div className="text-3xl mb-4">{f.icon}</div>
              <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">
                {f.title}
              </h3>
              <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed mb-4">
                {f.description}
              </p>
              <div className="flex flex-wrap gap-2">
                {f.tags.map((t) => (
                  <span
                    key={t}
                    className="text-[11px] px-2.5 py-1 rounded-full bg-[var(--color-primary)]/10 text-[var(--color-primary-light)] font-medium border border-[var(--color-primary)]/15"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ───────────────── Benchmarks ───────────────── */
function BenchmarksSection() {
  const benchmarks = [
    { engine: "Vecta Flat", qps: "1,413", recall: "100.0%", memory: "5,120 KB", compression: "1.0×" },
    { engine: "Vecta IVF", qps: "19,408", recall: "89.2%", memory: "5,170 KB", compression: "1.0×" },
    { engine: "Vecta HNSW", qps: "25,967", recall: "88.9%", memory: "6,144 KB", compression: "1.2×" },
    { engine: "Vecta IVFPQ", qps: "45,780", recall: "59.8%", memory: "262 KB", compression: "19.5×" },
  ];

  const comparisons = [
    { metric: "IVFPQ Throughput", vecta: "16,782 QPS", faiss: "16,152 QPS", winner: "vecta" },
    { metric: "IVFPQ Compression", vecta: "19.52×", faiss: "14.92×", winner: "vecta" },
    { metric: "IVFPQ Memory", vecta: "262 KB", faiss: "343 KB", winner: "vecta" },
    { metric: "IVF at ~90% Recall", vecta: "19,408 QPS", faiss: "68,569 QPS", winner: "faiss" },
    { metric: "HNSW at ~90% Recall", vecta: "7,252 QPS", faiss: "22,611 QPS", winner: "faiss" },
  ];

  return (
    <section id="benchmarks" className="relative py-28 px-6 bg-[var(--color-surface-alt)]">
      <div className="max-w-7xl mx-auto">
        <SectionHeader
          eyebrow="Performance"
          title="Benchmarked Against Meta FAISS"
          description="Head-to-head comparison on SIFT10k dataset under strict single-threaded CPU parity."
        />

        {/* Vecta Performance Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mt-16">
          {benchmarks.map((b) => (
            <div key={b.engine} className="hover-card glass rounded-2xl p-6 text-center">
              <h4 className="text-sm text-[var(--color-text-muted)] uppercase tracking-wider mb-3 font-medium">
                {b.engine}
              </h4>
              <div className="text-3xl font-bold gradient-text-static mb-1">{b.qps}</div>
              <div className="text-xs text-[var(--color-text-muted)] mb-4">Queries / Second</div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Recall</span>
                  <span className="text-[var(--color-text-primary)] font-medium">{b.recall}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Memory</span>
                  <span className="text-[var(--color-text-primary)] font-medium">{b.memory}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Compression</span>
                  <span className="text-emerald-400 font-medium">{b.compression}</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Vecta vs FAISS Table */}
        <div className="mt-14 glass rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-[var(--color-border)]">
            <h3 className="text-lg font-semibold">Vecta vs. Meta FAISS — Key Comparisons</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  <th className="text-left px-6 py-3 text-[var(--color-text-muted)] font-medium">Metric</th>
                  <th className="text-center px-6 py-3 text-[var(--color-primary-light)] font-medium">⚡ Vecta</th>
                  <th className="text-center px-6 py-3 text-[var(--color-text-muted)] font-medium">FAISS</th>
                  <th className="text-center px-6 py-3 text-[var(--color-text-muted)] font-medium">Winner</th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((c) => (
                  <tr key={c.metric} className="border-b border-[var(--color-border)]/50 hover:bg-[var(--color-surface-hover)] transition-colors">
                    <td className="px-6 py-3 text-[var(--color-text-secondary)]">{c.metric}</td>
                    <td className={`px-6 py-3 text-center font-medium ${c.winner === "vecta" ? "text-emerald-400" : "text-[var(--color-text-secondary)]"}`}>
                      {c.vecta}
                    </td>
                    <td className={`px-6 py-3 text-center font-medium ${c.winner === "faiss" ? "text-emerald-400" : "text-[var(--color-text-secondary)]"}`}>
                      {c.faiss}
                    </td>
                    <td className="px-6 py-3 text-center">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                        c.winner === "vecta"
                          ? "bg-emerald-400/10 text-emerald-400"
                          : "bg-blue-400/10 text-blue-400"
                      }`}>
                        {c.winner === "vecta" ? "⚡ Vecta" : "FAISS"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ───────────────── Architecture ───────────────── */
function ArchitectureSection() {
  return (
    <section id="architecture" className="relative py-28 px-6 overflow-hidden">
      <div className="max-w-5xl mx-auto">
        <SectionHeader
          eyebrow="Under the Hood"
          title="Architecture"
          description="Both the embedded Python module and standalone HTTP server consume the same Rust core."
        />

        <div className="mt-16 code-block p-6 sm:p-8 overflow-x-auto glow-purple">
          <pre className="text-[var(--color-text-secondary)] text-xs sm:text-sm leading-relaxed">
{`┌─────────────────────────────────────────────────────────────────────┐
│                       CLIENT APPLICATIONS                          │
│    Python Scripts       LangChain RAG        cURL / Web UI         │
└─────────┬──────────────────┬──────────────────────┬────────────────┘
          │                  │                      │
          │ Direct FFI       │ Python SDK           │ HTTP / JSON
          │ (PyO3)           │ (vecta_client)       │ (Bearer Auth)
          ▼                  ▼                      ▼
┌────────────────────┐  ┌────────────────────────────────────────────┐
│   src/python.rs    │  │       vecta-server (Axum + Tokio)          │
│  PyO3 Bindings     │  │  REST Routes · Auth · Swagger UI · WAL    │
└────────┬───────────┘  └──────────────────┬─────────────────────────┘
         │                                 │
         └────────────┬────────────────────┘
                      │  Shared Memory Calls
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        VECTA RUST CORE                              │
├──────────────────────────────┬──────────────────────────────────────┤
│  Index Architectures:        │  Durability & Storage:               │
│   • FlatIndex  (Exact SIMD)  │   • Write-Ahead Log (WAL + CRC32)   │
│   • IVFIndex   (k-means)     │   • Snapshot (Bincode)              │
│   • HnswIndex  (Graph)       │   • Memory-Mapped Zero-Copy         │
│   • IVFPQIndex (ADC Tables)  │   • Metadata Filter ASTs            │
├──────────────────────────────┴──────────────────────────────────────┤
│  Concurrency: ConcurrentFlatIndex (RwLock) + ShardedFlatIndex      │
└─────────────────────────────────────────────────────────────────────┘`}
          </pre>
        </div>

        {/* Execution modes */}
        <div className="mt-14 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="hover-card glass rounded-2xl p-7">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-[var(--color-primary)]/15 flex items-center justify-center text-xl">🐍</div>
              <h3 className="text-lg font-semibold">Embedded Mode</h3>
            </div>
            <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed mb-4">
              In-process CPython extension. Sub-microsecond latency via direct FFI calls. Best for local ML pipelines, notebooks, and edge inference.
            </p>
            <code className="text-xs text-[var(--color-accent-light)] font-mono bg-[var(--color-surface)]/50 px-3 py-1.5 rounded-lg">
              import vecta
            </code>
          </div>

          <div className="hover-card glass rounded-2xl p-7">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-[var(--color-accent)]/15 flex items-center justify-center text-xl">🌐</div>
              <h3 className="text-lg font-semibold">Server Mode</h3>
            </div>
            <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed mb-4">
              Async HTTP daemon with multi-threaded Tokio reactor. Any language via REST. Built for microservices, Kubernetes, and cloud deployments.
            </p>
            <code className="text-xs text-[var(--color-accent-light)] font-mono bg-[var(--color-surface)]/50 px-3 py-1.5 rounded-lg">
              localhost:6333/docs
            </code>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ───────────────── Quickstart ───────────────── */
function QuickstartSection() {
  const [activeTab, setActiveTab] = useState("python");

  const tabs = [
    { id: "python", label: "Python Embedded", icon: "🐍" },
    { id: "server", label: "REST API", icon: "🌐" },
    { id: "langchain", label: "LangChain", icon: "🦜" },
  ];

  const codeSnippets = {
    python: `import vecta

# Initialize 128-dimensional index with cosine similarity
index = vecta.HnswIndex(dim=128, metric="cosine")

# Insert vectors with unique IDs
index.add(0, [0.1, 0.2, 0.8, 0.5, ...])  # 128-dim vector
index.add(1, [0.9, 0.1, 0.2, 0.3, ...])
index.add(2, [0.15, 0.25, 0.75, 0.55, ...])

# Search top-k nearest neighbors
results = index.search(query=[0.12, 0.19, ...], k=5)
for vector_id, distance in results:
    print(f"ID: {vector_id}, Distance: {distance:.4f}")

# Save index to disk
index.save("my_index.bin")`,

    server: `# 1. Start the server
cargo run --release --bin vecta-server
# Or: docker run -d -p 6333:6333 vecta

# 2. Create a collection
curl -X POST http://localhost:6333/collections \\
  -H "Authorization: Bearer my_secret_key" \\
  -H "Content-Type: application/json" \\
  -d '{"name": "docs", "dim": 128, "index_type": "hnsw", "metric": "cosine"}'

# 3. Insert a vector
curl -X POST http://localhost:6333/collections/docs/points \\
  -H "Authorization: Bearer my_secret_key" \\
  -H "Content-Type: application/json" \\
  -d '{"id": 1, "vector": [0.1, 0.2, 0.8, 0.5, ...]}'

# 4. Search
curl -X POST http://localhost:6333/collections/docs/search \\
  -H "Authorization: Bearer my_secret_key" \\
  -H "Content-Type: application/json" \\
  -d '{"vector": [0.12, 0.19, ...], "k": 5, "ef_search": 64}'`,

    langchain: `from langchain_community.embeddings import OpenAIEmbeddings
from vecta_client.langchain import VectaVectorStore

# 1. Connect to Vecta with an embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = VectaVectorStore(
    collection_name="knowledge_base",
    embedding=embeddings,
    base_url="http://localhost:6333",
    api_key="my_secret_key"
)

# 2. Ingest documents
vector_store.add_texts(
    texts=["Vecta is a vector database built in Rust.",
           "HNSW provides fast approximate nearest neighbor search."],
    metadatas=[{"topic": "overview"}, {"topic": "hnsw"}]
)

# 3. Use as a retriever in your RAG pipeline
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("How does vector search work?")`,
  };

  return (
    <section id="quickstart" className="relative py-28 px-6 bg-[var(--color-surface-alt)]">
      <div className="max-w-4xl mx-auto">
        <SectionHeader
          eyebrow="Get Started"
          title="Up and Running in Minutes"
          description="Three ways to integrate Vecta into your application."
        />

        {/* Tabs */}
        <div className="mt-14 flex flex-wrap gap-2 justify-center">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 cursor-pointer ${
                activeTab === t.id
                  ? "bg-[var(--color-primary)] text-white shadow-lg shadow-[var(--color-primary)]/25"
                  : "glass text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-primary)]/40"
              }`}
            >
              <span>{t.icon}</span>
              {t.label}
            </button>
          ))}
        </div>

        {/* Code block */}
        <div className="mt-8 code-block overflow-hidden glow-purple">
          <div className="flex items-center justify-between px-5 py-3 border-b border-[var(--color-border)]">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500/60" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/60" />
              <div className="w-3 h-3 rounded-full bg-emerald-500/60" />
            </div>
            <span className="text-xs text-[var(--color-text-muted)]">
              {activeTab === "python" ? "example.py" : activeTab === "server" ? "terminal" : "rag_pipeline.py"}
            </span>
          </div>
          <div className="p-5 overflow-x-auto">
            <pre className="text-[var(--color-text-secondary)] text-sm leading-7">
              <code>{codeSnippets[activeTab]}</code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ───────────────── Index Comparison ───────────────── */
function IndexComparisonSection() {
  const indexes = [
    {
      name: "Flat",
      emoji: "📋",
      desc: "Exact exhaustive brute-force",
      recall: "100%",
      speed: "Baseline",
      memory: "1.0×",
      bestFor: "Ground truth, small datasets",
      color: "from-blue-500/20 to-blue-600/5",
    },
    {
      name: "IVF",
      emoji: "📊",
      desc: "Inverted file via k-means",
      recall: "80–98%",
      speed: "Fast",
      memory: "1.0×",
      bestFor: "Balanced speed & recall",
      color: "from-emerald-500/20 to-emerald-600/5",
    },
    {
      name: "HNSW",
      emoji: "🕸️",
      desc: "Hierarchical graph skip-lists",
      recall: "90–99%",
      speed: "Very Fast",
      memory: "1.1–1.3×",
      bestFor: "Low-latency, mission critical",
      color: "from-violet-500/20 to-violet-600/5",
    },
    {
      name: "IVFPQ",
      emoji: "🗜️",
      desc: "Product quantization + ADC",
      recall: "50–70%",
      speed: "Ultra Fast",
      memory: "0.05×",
      bestFor: "Huge datasets, tight RAM",
      color: "from-amber-500/20 to-amber-600/5",
    },
  ];

  return (
    <section className="relative py-28 px-6">
      <div className="max-w-7xl mx-auto">
        <SectionHeader
          eyebrow="Index Types"
          title="Choose Your Architecture"
          description="Each algorithm trades off between recall accuracy, query speed, and memory usage."
        />

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mt-16">
          {indexes.map((idx) => (
            <div key={idx.name} className="hover-card glass rounded-2xl overflow-hidden group">
              <div className={`h-1.5 bg-gradient-to-r ${idx.color}`} />
              <div className="p-6">
                <div className="text-3xl mb-3">{idx.emoji}</div>
                <h3 className="text-lg font-bold mb-1">{idx.name}</h3>
                <p className="text-xs text-[var(--color-text-muted)] mb-5">{idx.desc}</p>

                <div className="space-y-3 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-[var(--color-text-muted)]">Recall</span>
                    <span className="font-semibold text-[var(--color-text-primary)]">{idx.recall}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[var(--color-text-muted)]">Speed</span>
                    <span className="font-semibold text-[var(--color-text-primary)]">{idx.speed}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[var(--color-text-muted)]">Memory</span>
                    <span className="font-semibold text-emerald-400">{idx.memory}</span>
                  </div>
                </div>

                <div className="mt-5 pt-4 border-t border-[var(--color-border)]/50">
                  <p className="text-xs text-[var(--color-text-secondary)]">
                    <span className="text-[var(--color-primary-light)] font-medium">Best for: </span>
                    {idx.bestFor}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ───────────────── CTA / Footer ───────────────── */
function CTASection() {
  return (
    <section className="relative py-28 px-6 overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-[var(--color-primary)]/[0.06] blur-[120px]" />
      </div>

      <div className="relative z-10 max-w-3xl mx-auto text-center">
        <h2 className="text-3xl sm:text-4xl font-bold mb-5">
          Ready to Build with <span className="gradient-text-static">Vecta</span>?
        </h2>
        <p className="text-lg text-[var(--color-text-secondary)] mb-10">
          Open source, MIT licensed, and ready for production. Start building your vector-powered application today.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <a
            href="https://github.com/dhanushkumar-amk/VECTA"
            target="_blank"
            rel="noopener noreferrer"
            className="group px-8 py-4 rounded-xl font-semibold text-white bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-accent)] shadow-lg shadow-[var(--color-primary)]/20 hover:shadow-[var(--color-primary)]/40 transition-all duration-300 hover:scale-[1.03]"
          >
            <span className="flex items-center gap-2">
              <GithubIcon />
              Star on GitHub
              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" className="transition-transform duration-300 group-hover:translate-x-1">
                <path d="M3 8h10M9 4l4 4-4 4" />
              </svg>
            </span>
          </a>
          <a
            href="#quickstart"
            className="px-8 py-4 rounded-xl font-semibold border border-[var(--color-border)] text-[var(--color-text-primary)] hover:border-[var(--color-primary)] hover:bg-[var(--color-surface-hover)] transition-all duration-300"
          >
            Read the Docs
          </a>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-[var(--color-border)] py-10 px-6">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xl">⚡</span>
          <span className="font-semibold gradient-text-static">Vecta</span>
          <span className="text-sm text-[var(--color-text-muted)]">— Built with 🦀 Rust</span>
        </div>
        <div className="flex items-center gap-6">
          <a href="https://github.com/dhanushkumar-amk/VECTA" target="_blank" rel="noopener noreferrer" className="text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors">
            <GithubIcon />
          </a>
          <span className="text-sm text-[var(--color-text-muted)]">
            MIT License · © {new Date().getFullYear()} Dhanush Kumar
          </span>
        </div>
      </div>
    </footer>
  );
}

/* ───────────────── Shared Components ───────────────── */
function SectionHeader({ eyebrow, title, description }) {
  return (
    <div className="text-center max-w-2xl mx-auto">
      <span className="inline-block text-xs font-semibold uppercase tracking-[0.2em] text-[var(--color-primary-light)] mb-3">
        {eyebrow}
      </span>
      <h2 className="text-3xl sm:text-4xl font-bold mb-4">{title}</h2>
      <p className="text-[var(--color-text-secondary)] leading-relaxed">{description}</p>
    </div>
  );
}

function CodeCopy({ text }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  };

  return (
    <div className="inline-flex items-center gap-3 px-5 py-3 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border)] group">
      <span className="text-[var(--color-text-muted)] text-sm select-none">$</span>
      <code className="text-sm text-[var(--color-text-secondary)] font-mono">{text}</code>
      <button
        onClick={handleCopy}
        className="ml-2 text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors cursor-pointer"
        title="Copy to clipboard"
      >
        {copied ? (
          <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" className="text-emerald-400">
            <path d="M2 8l4 4 8-8" />
          </svg>
        ) : (
          <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="5" y="5" width="9" height="9" rx="1.5" />
            <path d="M2 10V3a1 1 0 011-1h7" />
          </svg>
        )}
      </button>
    </div>
  );
}

function GithubIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
    </svg>
  );
}

/* ───────────────── Page ───────────────── */
export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <HeroSection />
        <FeaturesSection />
        <IndexComparisonSection />
        <BenchmarksSection />
        <ArchitectureSection />
        <QuickstartSection />
        <CTASection />
      </main>
      <Footer />
    </>
  );
}
