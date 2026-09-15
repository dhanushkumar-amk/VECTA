"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import MinimalCodeWindow from "./components/CodeHighlight";

/* ═══════════════════════════════════════════
   ICONS
   ═══════════════════════════════════════════ */

function GithubIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
    </svg>
  );
}

function ArrowRight({ size = 16 }) {
  return (
    <svg width={size} height={size} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14M12 5l7 7-7 7" />
    </svg>
  );
}

function CopyIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="5" y="5" width="8" height="8" rx="1.5" />
      <path d="M3 10V3.5A.5.5 0 013.5 3H10" />
    </svg>
  );
}

function CheckIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} fill="none" stroke="currentColor" strokeWidth="2" className="text-emerald-400">
      <path d="M2 7l4 4 6-6" />
    </svg>
  );
}

/* ═══════════════════════════════════════════
   NAVBAR
   ═══════════════════════════════════════════ */

function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  const links = [
    { label: "Features", href: "#features" },
    { label: "Performance", href: "#performance" },
    { label: "Quickstart", href: "#quickstart" },
    { label: "Docs", href: "/docs" },
  ];

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? "glass-nav" : ""}`}>
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <a href="#" className="flex items-center gap-2">
          <span className="text-lg">⚡</span>
          <span className="text-base font-semibold tracking-tight text-[var(--color-text-primary)]">Vecta</span>
        </a>

        <div className="hidden md:flex items-center gap-8">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-[13px] text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors duration-200"
            >
              {l.label}
            </a>
          ))}
        </div>

        <div className="hidden md:flex items-center gap-3">
          <a
            href="https://github.com/dhanushkumar-amk/VECTA"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-[13px] text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
          >
            <GithubIcon size={15} />
          </a>
          <a
            href="#quickstart"
            className="text-[13px] font-medium px-4 py-1.5 rounded-lg bg-[var(--color-text-primary)] text-[var(--color-surface)] hover:opacity-90 transition-opacity"
          >
            Get Started
          </a>
        </div>

        <button
          className="md:hidden text-[var(--color-text-muted)]"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Menu"
        >
          <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5">
            {menuOpen ? <path d="M5 5l10 10M5 15L15 5" /> : <path d="M3 6h14M3 10h14M3 14h14" />}
          </svg>
        </button>
      </div>

      {menuOpen && (
        <div className="md:hidden glass-nav border-t border-[var(--color-border)] px-6 py-4 space-y-3">
          {links.map((l) => (
            <a key={l.href} href={l.href} onClick={() => setMenuOpen(false)}
              className="block text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
            >{l.label}</a>
          ))}
        </div>
      )}
    </nav>
  );
}

/* ═══════════════════════════════════════════
   HERO
   ═══════════════════════════════════════════ */

function Hero() {
  const [copied, setCopied] = useState(false);
  const installCmd = "pip install maturin && maturin develop --release";

  const handleCopy = async () => {
    try { await navigator.clipboard.writeText(installCmd); setCopied(true); setTimeout(() => setCopied(false), 2000); } catch { }
  };

  return (
    <section className="relative min-h-[100vh] flex flex-col items-center justify-center px-6 overflow-hidden">
      {/* Ambient glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] rounded-full bg-[var(--color-primary)]/[0.04] blur-[150px] pointer-events-none animate-fade-in" />
      <div className="absolute top-2/3 left-1/3 w-[400px] h-[300px] rounded-full bg-[var(--color-accent)]/[0.03] blur-[120px] pointer-events-none animate-fade-in delay-3" />

      {/* Subtle grid */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.02]" style={{
        backgroundImage: "linear-gradient(var(--color-text-muted) 1px, transparent 1px), linear-gradient(90deg, var(--color-text-muted) 1px, transparent 1px)",
        backgroundSize: "80px 80px",
      }} />

      <div className="relative z-10 max-w-3xl mx-auto text-center">
        {/* Badge */}
        <div className="animate-fade-up inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[var(--color-border)] bg-[var(--color-surface-card)] mb-8 text-xs text-[var(--color-text-muted)]">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          Open Source · MIT Licensed
        </div>

        {/* Headline */}
        <h1 className="animate-fade-up delay-1 text-[clamp(2.5rem,6vw,4.5rem)] font-bold leading-[1.08] tracking-[-0.035em] mb-6">
          Vector search{" "}
          <br className="hidden sm:block" />
          at <span className="gradient-text">warp speed</span>
        </h1>

        {/* Subtitle */}
        <p className="animate-fade-up delay-2 text-base sm:text-lg text-[var(--color-text-secondary)] leading-relaxed max-w-xl mx-auto mb-10">
          A production-grade vector database built from scratch in{" "}
          <span className="text-[var(--color-text-primary)] font-medium">pure Rust</span>.
          Four index architectures, Python bindings, REST API, and LangChain integration.
        </p>

        {/* CTA */}
        <div className="animate-fade-up delay-3 flex flex-col sm:flex-row items-center justify-center gap-3 mb-8">
          <a
            href="#quickstart"
            className="group inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-[var(--color-text-primary)] text-[var(--color-surface)] text-sm font-semibold hover:opacity-90 transition-all duration-200"
          >
            Get Started
            <span className="transition-transform duration-200 group-hover:translate-x-0.5"><ArrowRight size={14} /></span>
          </a>
          <a
            href="https://github.com/dhanushkumar-amk/VECTA"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg border border-[var(--color-border)] text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-border-light)] transition-all duration-200"
          >
            <GithubIcon size={15} />
            GitHub
          </a>
        </div>

        {/* Install */}
        <div className="animate-fade-up delay-4 inline-flex items-center gap-3 px-4 py-2.5 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/60">
          <span className="text-[var(--color-text-muted)] text-xs select-none">$</span>
          <code className="text-xs sm:text-sm text-[var(--color-text-secondary)] font-mono">{installCmd}</code>
          <button onClick={handleCopy} className="text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors cursor-pointer ml-1" title="Copy">
            {copied ? <CheckIcon /> : <CopyIcon />}
          </button>
        </div>
      </div>

      {/* Stats bar */}
      <div className="relative z-10 mt-20 animate-fade-up delay-5">
        <div className="flex items-center gap-8 sm:gap-14 text-center">
          {[
            { value: "45.7k", label: "QPS Peak" },
            { value: "19.5×", label: "Compression" },
            { value: "4", label: "Index Types" },
            { value: "100%", label: "Rust" },
          ].map((s, i) => (
            <div key={s.label} className={i > 0 ? "border-l border-[var(--color-border)] pl-8 sm:pl-14" : ""}>
              <div className="text-xl sm:text-2xl font-semibold text-[var(--color-text-primary)] tracking-tight">{s.value}</div>
              <div className="text-[11px] text-[var(--color-text-muted)] mt-0.5 uppercase tracking-wider">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   FEATURES
   ═══════════════════════════════════════════ */

function Features() {
  const features = [
    {
      icon: "🧠",
      title: "Four Index Architectures",
      desc: "Flat (exact), IVF (k-means), HNSW (graph), and IVFPQ (compressed). Pick the right tradeoff for your data.",
    },
    {
      icon: "🦀",
      title: "Pure Rust, Zero Dependencies",
      desc: "The entire engine is hand-written Rust—k-means, graph traversal, product quantization. No C/C++ linkage.",
    },
    {
      icon: "🐍",
      title: "Native Python Bindings",
      desc: "PyO3 extension with GIL-released concurrency. Sub-microsecond call latency via direct FFI.",
    },
    {
      icon: "🌐",
      title: "REST API Server",
      desc: "Axum + Tokio HTTP server with Bearer auth, WAL crash recovery, and interactive Swagger UI.",
    },
    {
      icon: "🦜",
      title: "LangChain Integration",
      desc: "Drop-in VectaVectorStore implementing LangChain's VectorStore interface for RAG pipelines.",
    },
    {
      icon: "🐳",
      title: "Docker Ready",
      desc: "One-command deployment. Persistent volumes, environment config, and cloud-ready.",
    },
  ];

  return (
    <section id="features" className="relative py-32 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-primary-light)] mb-3">Features</p>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Built for real workloads</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-[var(--color-border)] rounded-2xl overflow-hidden border border-[var(--color-border)]">
          {features.map((f) => (
            <div key={f.title} className="bg-[var(--color-surface)] p-8 hover:bg-[var(--color-surface-card)] transition-colors duration-300">
              <span className="text-2xl block mb-4">{f.icon}</span>
              <h3 className="text-[15px] font-semibold text-[var(--color-text-primary)] mb-2">{f.title}</h3>
              <p className="text-sm text-[var(--color-text-muted)] leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   INDEX TYPES
   ═══════════════════════════════════════════ */

function IndexTypes() {
  const indexes = [
    { name: "Flat", tag: "Exact", recall: "100%", qps: "1,413", mem: "1.0×", desc: "Brute-force exhaustive search. Zero setup.", color: "#60a5fa" },
    { name: "IVF", tag: "Fast", recall: "80–98%", qps: "19,408", mem: "1.0×", desc: "K-means partitioned inverted file.", color: "#34d399" },
    { name: "HNSW", tag: "Recommended", recall: "90–99%", qps: "25,967", mem: "1.1×", desc: "Graph-based skip-list. Best all-rounder.", color: "#a78bfa" },
    { name: "IVFPQ", tag: "Compressed", recall: "50–70%", qps: "45,780", mem: "0.05×", desc: "Product quantized. 19× memory savings.", color: "#fbbf24" },
  ];

  return (
    <section className="relative py-32 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-primary-light)] mb-3">Index Types</p>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Choose your architecture</h2>
          <p className="text-sm text-[var(--color-text-muted)] mt-3 max-w-md mx-auto">Each trades off recall, speed, and memory differently.</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {indexes.map((idx) => (
            <div key={idx.name} className="card-hover rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface-card)] p-6 flex flex-col">
              <div className="flex items-center justify-between mb-5">
                <h3 className="text-lg font-semibold tracking-tight">{idx.name}</h3>
                <span className="text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full border" style={{ borderColor: idx.color + "40", color: idx.color, background: idx.color + "10" }}>
                  {idx.tag}
                </span>
              </div>

              <p className="text-xs text-[var(--color-text-muted)] mb-6 leading-relaxed">{idx.desc}</p>

              <div className="mt-auto space-y-3">
                {[
                  { label: "Recall", value: idx.recall },
                  { label: "QPS", value: idx.qps },
                  { label: "Memory", value: idx.mem },
                ].map((row) => (
                  <div key={row.label} className="flex items-center justify-between text-sm">
                    <span className="text-[var(--color-text-muted)]">{row.label}</span>
                    <span className="font-mono text-xs font-medium text-[var(--color-text-primary)]">{row.value}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   PERFORMANCE
   ═══════════════════════════════════════════ */

function Performance() {
  const rows = [
    { metric: "IVFPQ Throughput", vecta: "16,782 QPS", faiss: "16,152 QPS", winner: "vecta" },
    { metric: "IVFPQ Compression", vecta: "19.52×", faiss: "14.92×", winner: "vecta" },
    { metric: "IVFPQ Memory", vecta: "262 KB", faiss: "343 KB", winner: "vecta" },
    { metric: "IVF ~90% Recall", vecta: "19,408 QPS", faiss: "68,569 QPS", winner: "faiss" },
    { metric: "HNSW ~90% Recall", vecta: "7,252 QPS", faiss: "22,611 QPS", winner: "faiss" },
  ];

  return (
    <section id="performance" className="relative py-32 px-6 bg-[var(--color-surface-alt)]">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-primary-light)] mb-3">Performance</p>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Benchmarked against FAISS</h2>
          <p className="text-sm text-[var(--color-text-muted)] mt-3 max-w-lg mx-auto">
            SIFT10k dataset, single-threaded CPU parity. Vecta outperforms on compression and matches on throughput.
          </p>
        </div>

        {/* Table */}
        <div className="rounded-2xl border border-[var(--color-border)] overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)]">
                <th className="text-left px-6 py-3.5 text-xs text-[var(--color-text-muted)] font-medium uppercase tracking-wider">Metric</th>
                <th className="text-right px-6 py-3.5 text-xs text-[var(--color-primary-light)] font-medium uppercase tracking-wider">Vecta</th>
                <th className="text-right px-6 py-3.5 text-xs text-[var(--color-text-muted)] font-medium uppercase tracking-wider">FAISS</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.metric} className="border-b border-[var(--color-border)]/50 last:border-0 hover:bg-[var(--color-surface-hover)] transition-colors">
                  <td className="px-6 py-3.5 text-[var(--color-text-secondary)]">{r.metric}</td>
                  <td className={`px-6 py-3.5 text-right font-mono text-xs font-medium ${r.winner === "vecta" ? "text-emerald-400" : "text-[var(--color-text-secondary)]"}`}>
                    {r.vecta}
                    {r.winner === "vecta" && <span className="ml-1.5 text-[10px]">✓</span>}
                  </td>
                  <td className={`px-6 py-3.5 text-right font-mono text-xs font-medium ${r.winner === "faiss" ? "text-blue-400" : "text-[var(--color-text-secondary)]"}`}>
                    {r.faiss}
                    {r.winner === "faiss" && <span className="ml-1.5 text-[10px]">✓</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Highlight cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-8">
          {[
            { value: "19.52×", label: "Memory compression", sub: "vs FAISS 14.92×" },
            { value: "45,780", label: "Peak QPS (IVFPQ)", sub: "Single-threaded" },
            { value: "262 KB", label: "10k vectors stored", sub: "Down from 5,120 KB" },
          ].map((c) => (
            <div key={c.label} className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)] p-5 text-center">
              <div className="text-2xl font-semibold gradient-text-static">{c.value}</div>
              <div className="text-xs text-[var(--color-text-secondary)] mt-1">{c.label}</div>
              <div className="text-[10px] text-[var(--color-text-muted)] mt-0.5">{c.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   ARCHITECTURE
   ═══════════════════════════════════════════ */

function Architecture() {
  const [activeMode, setActiveMode] = useState("all");

  return (
    <section id="architecture" className="relative py-32 px-6">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-16">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-primary-light)] mb-3">Architecture</p>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Two modes, one core</h2>
          <p className="text-sm text-[var(--color-text-muted)] mt-3">Both execution modes share the same Rust engine.</p>
        </div>

        {/* Interactive Mode Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
          <div
            onClick={() => setActiveMode(activeMode === "embedded" ? "all" : "embedded")}
            className={`card-hover rounded-2xl border p-7 cursor-pointer transition-all duration-300 ${
              activeMode === "embedded"
                ? "border-emerald-500/50 bg-emerald-950/10 shadow-lg shadow-emerald-500/5"
                : "border-[var(--color-border)] bg-[var(--color-surface-card)] hover:border-zinc-700"
            }`}
          >
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-base">
                  🐍
                </div>
                <div>
                  <h3 className="text-sm font-semibold flex items-center gap-2">
                    Embedded Mode
                    {activeMode === "embedded" && (
                      <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                        Active
                      </span>
                    )}
                  </h3>
                  <p className="text-xs text-[var(--color-text-muted)]">import vecta</p>
                </div>
              </div>
              <span className="text-[11px] font-mono text-emerald-400">&lt; 0.8 µs</span>
            </div>
            <ul className="space-y-2.5 text-sm text-[var(--color-text-secondary)]">
              <li className="flex items-start gap-2"><span className="text-emerald-400 mt-0.5">·</span>In-process native CPython extension (PyO3)</li>
              <li className="flex items-start gap-2"><span className="text-emerald-400 mt-0.5">·</span>Sub-microsecond latency (direct C-ABI)</li>
              <li className="flex items-start gap-2"><span className="text-emerald-400 mt-0.5">·</span>GIL-released for true parallelism</li>
              <li className="flex items-start gap-2"><span className="text-emerald-400 mt-0.5">·</span>Best for ML pipelines &amp; notebooks</li>
            </ul>
          </div>

          <div
            onClick={() => setActiveMode(activeMode === "server" ? "all" : "server")}
            className={`card-hover rounded-2xl border p-7 cursor-pointer transition-all duration-300 ${
              activeMode === "server"
                ? "border-cyan-500/50 bg-cyan-950/10 shadow-lg shadow-cyan-500/5"
                : "border-[var(--color-border)] bg-[var(--color-surface-card)] hover:border-zinc-700"
            }`}
          >
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-base">
                  🌐
                </div>
                <div>
                  <h3 className="text-sm font-semibold flex items-center gap-2">
                    Server Mode
                    {activeMode === "server" && (
                      <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20">
                        Active
                      </span>
                    )}
                  </h3>
                  <p className="text-xs text-[var(--color-text-muted)]">localhost:6333</p>
                </div>
              </div>
              <span className="text-[11px] font-mono text-cyan-400">REST / JSON</span>
            </div>
            <ul className="space-y-2.5 text-sm text-[var(--color-text-secondary)]">
              <li className="flex items-start gap-2"><span className="text-cyan-400 mt-0.5">·</span>Async Axum + Tokio HTTP daemon</li>
              <li className="flex items-start gap-2"><span className="text-cyan-400 mt-0.5">·</span>Any language via REST / JSON</li>
              <li className="flex items-start gap-2"><span className="text-cyan-400 mt-0.5">·</span>WAL crash durability + auto recovery</li>
              <li className="flex items-start gap-2"><span className="text-cyan-400 mt-0.5">·</span>Best for microservices &amp; production</li>
            </ul>
          </div>
        </div>

        {/* Minimal Architecture Diagram */}
        <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface-card)] p-6 sm:p-8 shadow-xl">
          {/* Header Controls */}
          <div className="flex flex-wrap items-center justify-between gap-3 mb-8 pb-4 border-b border-[var(--color-border)]">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span className="text-xs font-mono text-[var(--color-text-secondary)]">Engine Architecture</span>
            </div>

            <div className="flex items-center gap-1 p-1 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-xs">
              <button
                onClick={() => setActiveMode("all")}
                className={`px-2.5 py-1 rounded font-mono text-[11px] transition-all cursor-pointer ${
                  activeMode === "all"
                    ? "bg-[var(--color-text-primary)] text-[var(--color-surface)] font-medium shadow-sm"
                    : "text-[var(--color-text-muted)] hover:text-white"
                }`}
              >
                All
              </button>
              <button
                onClick={() => setActiveMode("embedded")}
                className={`px-2.5 py-1 rounded font-mono text-[11px] transition-all cursor-pointer ${
                  activeMode === "embedded"
                    ? "bg-emerald-500 text-zinc-950 font-medium shadow-sm"
                    : "text-[var(--color-text-muted)] hover:text-white"
                }`}
              >
                Embedded
              </button>
              <button
                onClick={() => setActiveMode("server")}
                className={`px-2.5 py-1 rounded font-mono text-[11px] transition-all cursor-pointer ${
                  activeMode === "server"
                    ? "bg-cyan-500 text-zinc-950 font-medium shadow-sm"
                    : "text-[var(--color-text-muted)] hover:text-white"
                }`}
              >
                Server
              </button>
            </div>
          </div>

          {/* Minimal Tree Structure */}
          <div className="space-y-6">
            {/* 1. Client Layer */}
            <div className="text-center">
              <div className="inline-flex items-center gap-2 sm:gap-3 px-4 py-2 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] text-xs font-mono text-zinc-300">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                <span className="text-[var(--color-text-muted)]">Clients:</span>
                <span>Python SDK</span>
                <span className="text-zinc-700">·</span>
                <span>LangChain</span>
                <span className="text-zinc-700">·</span>
                <span>REST / cURL</span>
              </div>
            </div>

            {/* Straight Clean Connectors */}
            <div className="flex justify-center items-center">
              <div className="grid grid-cols-2 gap-8 sm:gap-24 text-[10px] font-mono text-center">
                <div className="flex flex-col items-center">
                  <span className={`px-2 py-0.5 rounded border transition-colors ${
                    activeMode === "all" || activeMode === "embedded"
                      ? "text-emerald-400 border-emerald-500/30 bg-emerald-950/40"
                      : "text-zinc-600 border-zinc-800 bg-zinc-900/30"
                  }`}>
                    FFI (PyO3) · &lt;0.8µs
                  </span>
                  <div className={`w-px h-5 my-1 transition-colors ${
                    activeMode === "all" || activeMode === "embedded" ? "bg-emerald-500/60" : "bg-zinc-800"
                  }`} />
                  <span className={activeMode === "all" || activeMode === "embedded" ? "text-emerald-400" : "text-zinc-700"}>↓</span>
                </div>

                <div className="flex flex-col items-center">
                  <span className={`px-2 py-0.5 rounded border transition-colors ${
                    activeMode === "all" || activeMode === "server"
                      ? "text-cyan-400 border-cyan-500/30 bg-cyan-950/40"
                      : "text-zinc-600 border-zinc-800 bg-zinc-900/30"
                  }`}>
                    REST · Port 6333
                  </span>
                  <div className={`w-px h-5 my-1 transition-colors ${
                    activeMode === "all" || activeMode === "server" ? "bg-cyan-500/60" : "bg-zinc-800"
                  }`} />
                  <span className={activeMode === "all" || activeMode === "server" ? "text-cyan-400" : "text-zinc-700"}>↓</span>
                </div>
              </div>
            </div>

            {/* 2. Middle Ingress Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* python.rs */}
              <div className={`p-4 rounded-xl border transition-all duration-200 ${
                activeMode === "all" || activeMode === "embedded"
                  ? "border-emerald-500/30 bg-[var(--color-surface)] shadow-sm"
                  : "border-[var(--color-border)] bg-[var(--color-surface)]/40 opacity-40"
              }`}>
                <div className="flex items-center justify-between mb-1">
                  <h4 className="text-xs font-mono font-semibold text-zinc-200">python.rs</h4>
                  <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.2 rounded border border-emerald-500/20">
                    PyO3 FFI
                  </span>
                </div>
                <p className="text-xs text-[var(--color-text-muted)] leading-relaxed">
                  In-process CPython extension with direct pointer memory access and GIL release.
                </p>
              </div>

              {/* vecta-server */}
              <div className={`p-4 rounded-xl border transition-all duration-200 ${
                activeMode === "all" || activeMode === "server"
                  ? "border-cyan-500/30 bg-[var(--color-surface)] shadow-sm"
                  : "border-[var(--color-border)] bg-[var(--color-surface)]/40 opacity-40"
              }`}>
                <div className="flex items-center justify-between mb-1">
                  <h4 className="text-xs font-mono font-semibold text-zinc-200">vecta-server</h4>
                  <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-1.5 py-0.2 rounded border border-cyan-500/20">
                    Axum · WAL
                  </span>
                </div>
                <p className="text-xs text-[var(--color-text-muted)] leading-relaxed">
                  Tokio HTTP daemon with Bearer auth, WAL crash persistence, and Swagger docs.
                </p>
              </div>
            </div>

            {/* Clean Down Connector into Core */}
            <div className="flex flex-col items-center justify-center my-1">
              <div className="w-px h-5 bg-zinc-800" />
              <span className="text-zinc-600 text-[10px]">↓</span>
            </div>

            {/* 3. Core Engine Card */}
            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5 text-center">
              <div className="flex items-center justify-center gap-2 mb-3">
                <span className="text-xs font-mono font-semibold text-zinc-200 uppercase tracking-wider">
                  Vecta Rust Core Engine
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
                {[
                  { name: "Flat", desc: "SIMD Exact" },
                  { name: "IVF", desc: "K-Means Centroids" },
                  { name: "HNSW", desc: "Skip-List Graph" },
                  { name: "IVFPQ", desc: "19.5× Quantized" },
                ].map((item) => (
                  <div key={item.name} className="p-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-card)]">
                    <span className="text-xs font-mono font-medium text-zinc-200 block">{item.name}</span>
                    <span className="text-[10px] text-[var(--color-text-muted)]">{item.desc}</span>
                  </div>
                ))}
              </div>

              <div className="pt-2.5 border-t border-[var(--color-border)] flex flex-wrap items-center justify-center gap-4 text-[11px] font-mono text-[var(--color-text-muted)]">
                <span>Write-Ahead Log (WAL)</span>
                <span>·</span>
                <span>Atomic Snapshots</span>
                <span>·</span>
                <span>Zero-Copy mmap</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   QUICKSTART (TABBED)
   ═══════════════════════════════════════════ */

function Quickstart() {
  const [tab, setTab] = useState("python");

  const tabs = [
    { id: "python", label: "Python", filename: "example.py", lang: "python" },
    { id: "api", label: "REST API", filename: "terminal.sh", lang: "bash" },
    { id: "langchain", label: "LangChain", filename: "rag.py", lang: "python" },
  ];

  const code = {
    python: `import vecta

# Create a 128-d index with cosine similarity
index = vecta.HnswIndex(dim=128, metric="cosine")

# Add vectors
index.add(0, [0.1] * 128)
index.add(1, [0.9] * 128)
index.add(2, [0.5] * 128)

# Search
results = index.search(query=[0.12] * 128, k=2)
for id, dist in results:
    print(f"  ID={id}, Distance={dist:.4f}")

# Persist to disk
index.save("my_index.vecta")`,

    api: `# Start the server
cargo run --release --bin vecta-server

# Create a collection
curl -X POST http://localhost:6333/collections \\
  -H "Authorization: Bearer my_secret_key" \\
  -H "Content-Type: application/json" \\
  -d '{"name":"docs","dim":128,"index_type":"hnsw","metric":"cosine"}'

# Insert a vector
curl -X POST http://localhost:6333/collections/docs/points \\
  -H "Authorization: Bearer my_secret_key" \\
  -H "Content-Type: application/json" \\
  -d '{"id": 1, "vector": [0.1, 0.2, 0.8, 0.5]}'

# Search
curl -X POST http://localhost:6333/collections/docs/search \\
  -H "Authorization: Bearer my_secret_key" \\
  -H "Content-Type: application/json" \\
  -d '{"vector": [0.12, 0.19, 0.78, 0.52], "k": 5}'`,

    langchain: `from vecta_client import VectaClient
from vecta_client.langchain import VectaVectorStore
from langchain_community.embeddings import OpenAIEmbeddings

client = VectaClient("http://localhost:6333", api_key="secret")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client.create_collection("kb", dim=1536, index_type="hnsw", metric="cosine")

store = VectaVectorStore(client=client, collection="kb", embedding=embeddings)

store.add_texts(
    texts=["Vecta is built in Rust.", "HNSW enables fast search."],
    metadatas=[{"topic": "overview"}, {"topic": "hnsw"}]
)

retriever = store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("How does vector search work?")`,
  };

  const currentTab = tabs.find((t) => t.id === tab) || tabs[0];

  return (
    <section id="quickstart" className="relative py-32 px-6 bg-[var(--color-surface-alt)]">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-primary-light)] mb-3">Quickstart</p>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Up and running in minutes</h2>
        </div>

        {/* Tab bar */}
        <div className="flex items-center gap-1 p-1 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border)] w-fit mx-auto mb-8">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 cursor-pointer ${tab === t.id
                  ? "bg-[var(--color-text-primary)] text-[var(--color-surface)] shadow-sm"
                  : "text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]"
                }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Code window with minimal theme syntax highlighting */}
        <MinimalCodeWindow
          code={code[tab]}
          filename={currentTab.filename}
          language={currentTab.lang}
        />
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   API ENDPOINTS
   ═══════════════════════════════════════════ */

function APIEndpoints() {
  const endpoints = [
    { method: "GET", path: "/health", desc: "Server health check" },
    { method: "POST", path: "/collections", desc: "Create a collection" },
    { method: "GET", path: "/collections", desc: "List all collections" },
    { method: "GET", path: "/collections/:name", desc: "Get collection info" },
    { method: "DELETE", path: "/collections/:name", desc: "Delete a collection" },
    { method: "POST", path: "/collections/:name/points", desc: "Insert a vector" },
    { method: "POST", path: "/collections/:name/search", desc: "k-NN search" },
    { method: "POST", path: "/collections/:name/checkpoint", desc: "Snapshot to disk" },
  ];

  const methodColor = {
    GET: "text-emerald-400 bg-emerald-400/8 border-emerald-400/20",
    POST: "text-blue-400 bg-blue-400/8 border-blue-400/20",
    DELETE: "text-red-400 bg-red-400/8 border-red-400/20",
  };

  return (
    <section className="relative py-32 px-6">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-primary-light)] mb-3">REST API</p>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Clean, simple endpoints</h2>
          <p className="text-sm text-[var(--color-text-muted)] mt-3">Interactive docs at <code className="text-[var(--color-accent-light)] text-xs font-mono">localhost:6333/docs</code></p>
        </div>

        <div className="rounded-2xl border border-[var(--color-border)] overflow-hidden divide-y divide-[var(--color-border)]">
          {endpoints.map((e) => (
            <div key={e.path + e.method} className="flex items-center gap-4 px-5 py-3 hover:bg-[var(--color-surface-hover)] transition-colors">
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${methodColor[e.method]} uppercase tracking-wider shrink-0 w-16 text-center`}>
                {e.method}
              </span>
              <code className="text-sm font-mono text-[var(--color-text-primary)] flex-1">{e.path}</code>
              <span className="text-xs text-[var(--color-text-muted)] hidden sm:block">{e.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   CTA
   ═══════════════════════════════════════════ */

function CTA() {
  return (
    <section className="relative py-32 px-6">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[300px] rounded-full bg-[var(--color-primary)]/[0.03] blur-[120px] pointer-events-none" />

      <div className="relative z-10 max-w-2xl mx-auto text-center">
        <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-4">
          Start building with Vecta
        </h2>
        <p className="text-sm text-[var(--color-text-muted)] mb-8 max-w-md mx-auto">
          Open source, MIT licensed, production ready. Build your vector-powered application today.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
          <a
            href="https://github.com/dhanushkumar-amk/VECTA"
            target="_blank"
            rel="noopener noreferrer"
            className="group inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-[var(--color-text-primary)] text-[var(--color-surface)] text-sm font-semibold hover:opacity-90 transition-all"
          >
            <GithubIcon size={15} />
            Star on GitHub
            <span className="transition-transform duration-200 group-hover:translate-x-0.5"><ArrowRight size={14} /></span>
          </a>
          <Link
            href="/docs"
            className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg border border-[var(--color-border)] text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-border-light)] transition-all"
          >
            Read the Docs
          </Link>
        </div>
      </div>
    </section>
  );
}

/* ═══════════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════════ */

function Footer() {
  return (
    <footer className="border-t border-[var(--color-border)] py-8 px-6">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-muted)]">
          <span>⚡</span>
          <span className="font-medium text-[var(--color-text-secondary)]">Vecta</span>
          <span>·</span>
          <span>Built with Rust</span>
        </div>
        <div className="flex items-center gap-5 text-xs text-[var(--color-text-muted)]">
          <a href="https://github.com/dhanushkumar-amk/VECTA" target="_blank" rel="noopener noreferrer" className="hover:text-[var(--color-text-secondary)] transition-colors">
            GitHub
          </a>
          <Link href="/docs" className="hover:text-[var(--color-text-secondary)] transition-colors">
            Documentation
          </Link>
          <span>MIT License © {new Date().getFullYear()}</span>
        </div>
      </div>
    </footer>
  );
}

/* ═══════════════════════════════════════════
   PAGE
   ═══════════════════════════════════════════ */

export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <div className="divider-glow" />
        <Features />
        <IndexTypes />
        <div className="divider-glow" />
        <Performance />
        <Architecture />
        <div className="divider-glow" />
        <Quickstart />
        <APIEndpoints />
        <div className="divider-glow" />
        <CTA />
      </main>
      <Footer />
    </>
  );
}
