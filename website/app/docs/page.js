"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { HighlightedCode } from "../components/CodeHighlight";

/* ═══════════════════════════════════════════════════════════════
   DOCUMENTATION DATA — All docs content lives here
   ═══════════════════════════════════════════════════════════════ */

const SIDEBAR_SECTIONS = [
  {
    title: "Getting Started",
    items: [
      { id: "introduction", label: "Introduction" },
      { id: "installation", label: "Installation" },
      { id: "quickstart", label: "Quickstart" },
      { id: "choosing-mode", label: "Embedded vs. Server" },
    ],
  },
  {
    title: "Index Types",
    items: [
      { id: "flat-index", label: "Flat Index" },
      { id: "ivf-index", label: "IVF Index" },
      { id: "hnsw-index", label: "HNSW Index" },
      { id: "ivfpq-index", label: "IVFPQ Index" },
      { id: "choosing-index", label: "Choosing an Index" },
    ],
  },
  {
    title: "Python SDK (Embedded)",
    items: [
      { id: "python-install", label: "Installation" },
      { id: "python-flat", label: "FlatIndex API" },
      { id: "python-ivf", label: "IVFIndex API" },
      { id: "python-hnsw", label: "HnswIndex API" },
      { id: "python-ivfpq", label: "IVFPQIndex API" },
      { id: "python-concurrent", label: "ConcurrentFlatIndex" },
      { id: "python-sharded", label: "ShardedFlatIndex" },
      { id: "python-metadata", label: "MetadataStore" },
      { id: "python-persistence", label: "Save & Load" },
    ],
  },
  {
    title: "REST API Server",
    items: [
      { id: "server-start", label: "Starting the Server" },
      { id: "server-auth", label: "Authentication" },
      { id: "api-health", label: "GET /health" },
      { id: "api-create-collection", label: "POST /collections" },
      { id: "api-list-collections", label: "GET /collections" },
      { id: "api-get-collection", label: "GET /collections/:name" },
      { id: "api-delete-collection", label: "DELETE /collections/:name" },
      { id: "api-insert", label: "POST .../points" },
      { id: "api-search", label: "POST .../search" },
      { id: "api-checkpoint", label: "POST .../checkpoint" },
    ],
  },
  {
    title: "Python Client SDK",
    items: [
      { id: "client-install", label: "Installation" },
      { id: "client-usage", label: "Usage Guide" },
      { id: "client-api", label: "API Reference" },
    ],
  },
  {
    title: "LangChain Integration",
    items: [
      { id: "langchain-setup", label: "Setup" },
      { id: "langchain-usage", label: "Usage" },
      { id: "langchain-rag", label: "RAG Pipeline" },
    ],
  },
  {
    title: "Deployment",
    items: [
      { id: "docker", label: "Docker" },
      { id: "docker-compose", label: "Docker Compose" },
      { id: "env-vars", label: "Environment Variables" },
    ],
  },
];

/* ═══════════════════════════════════════════════════════════════
   UTILITY COMPONENTS
   ═══════════════════════════════════════════════════════════════ */

function CodeBlock({ children, language = "", filename = "", showCopy = true }) {
  const [copied, setCopied] = useState(false);
  const textContent = typeof children === "string" ? children : String(children || "");

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(textContent.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { }
  };

  return (
    <div className="group relative rounded-xl border border-[var(--color-border)] bg-[#09090e] overflow-hidden my-5 shadow-lg">
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--color-border)] bg-[#0e0e14]">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]/50" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#febc2e]/50" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#28c840]/50" />
          </div>
          {filename && (
            <span className="text-xs text-zinc-400 ml-2 font-mono flex items-center gap-1.5">
              {filename}
            </span>
          )}
        </div>

        {showCopy && (
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2 py-0.5 rounded text-xs text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/60 transition-all cursor-pointer"
            title="Copy code"
          >
            {copied ? (
              <>
                <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" className="text-emerald-400">
                  <path d="M2 6.5l3 3 5.5-5.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span className="text-[11px] font-mono text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="4" y="4" width="6.5" height="6.5" rx="1.2" />
                  <path d="M2.5 7.5V2.5A1 1 0 013.5 1.5H8.5" />
                </svg>
                <span className="text-[11px] font-mono text-zinc-400">Copy</span>
              </>
            )}
          </button>
        )}
      </div>

      <pre className="p-4 sm:p-5 overflow-x-auto text-xs sm:text-sm leading-relaxed font-mono selection:bg-indigo-500/20">
        <code>
          <HighlightedCode code={textContent} />
        </code>
      </pre>
    </div>
  );
}

function Callout({ type = "note", children }) {
  const configs = {
    note: {
      border: "border-l-2 border-indigo-500/80 bg-indigo-500/[0.03]",
      titleColor: "text-indigo-400",
      title: "Note",
      icon: (
        <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="16" x2="12" y2="12" />
          <line x1="12" y1="8" x2="12.01" y2="8" />
        </svg>
      ),
    },
    tip: {
      border: "border-l-2 border-emerald-500/80 bg-emerald-500/[0.03]",
      titleColor: "text-emerald-400",
      title: "Tip",
      icon: (
        <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path d="M9 18h6M10 22h4M12 2a7 7 0 00-7 7c0 2.5 1.5 4.5 3 5.5v1.5a1 1 0 001 1h6a1 1 0 001-1V14.5c1.5-1 3-3 3-5.5a7 7 0 00-7-7z" />
        </svg>
      ),
    },
    warning: {
      border: "border-l-2 border-amber-500/80 bg-amber-500/[0.03]",
      titleColor: "text-amber-400",
      title: "Warning",
      icon: (
        <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
      ),
    },
    important: {
      border: "border-l-2 border-purple-500/80 bg-purple-500/[0.03]",
      titleColor: "text-purple-400",
      title: "Important",
      icon: (
        <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 8v4M12 16h.01" />
        </svg>
      ),
    },
  };

  const c = configs[type] || configs.note;

  return (
    <div className={`rounded-r-xl rounded-l-sm border border-[var(--color-border)] ${c.border} p-4 my-5`}>
      <div className={`flex items-center gap-2 font-mono text-xs font-semibold ${c.titleColor} mb-1.5`}>
        {c.icon}
        <span>{c.title}</span>
      </div>
      <div className="text-xs sm:text-sm text-zinc-300 leading-relaxed">{children}</div>
    </div>
  );
}

function ParamTable({ params }) {
  return (
    <div className="rounded-xl border border-[var(--color-border)] overflow-hidden my-5 bg-[#0a0a0f]">
      <table className="w-full text-xs sm:text-sm">
        <thead>
          <tr className="border-b border-[var(--color-border)] bg-[#0e0e14]">
            <th className="text-left px-4 py-3 text-[11px] font-mono uppercase tracking-wider text-zinc-500 font-medium">Parameter</th>
            <th className="text-left px-4 py-3 text-[11px] font-mono uppercase tracking-wider text-zinc-500 font-medium">Type</th>
            <th className="text-left px-4 py-3 text-[11px] font-mono uppercase tracking-wider text-zinc-500 font-medium">Description</th>
          </tr>
        </thead>
        <tbody>
          {params.map((p, i) => (
            <tr key={i} className="border-b border-[var(--color-border)]/50 last:border-0 hover:bg-zinc-900/30 transition-colors">
              <td className="px-4 py-3 font-mono text-indigo-300 text-xs">
                {p.name}
                {p.required && <span className="text-rose-400 ml-1 font-bold">*</span>}
              </td>
              <td className="px-4 py-3 font-mono text-zinc-500 text-xs">{p.type}</td>
              <td className="px-4 py-3 text-zinc-300 leading-relaxed text-xs sm:text-sm">{p.desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MethodBadge({ method }) {
  const colors = {
    GET: "bg-emerald-500/10 text-emerald-400 border-emerald-500/25",
    POST: "bg-indigo-500/10 text-indigo-400 border-indigo-500/25",
    DELETE: "bg-rose-500/10 text-rose-400 border-rose-500/25",
    PUT: "bg-amber-500/10 text-amber-400 border-amber-500/25",
  };
  return (
    <span className={`inline-flex px-2 py-0.5 rounded text-[11px] font-mono font-bold border ${colors[method] || "text-zinc-300"}`}>
      {method}
    </span>
  );
}

function EndpointHeader({ method, path, description }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 mb-4 p-3.5 rounded-xl bg-[#0b0b10] border border-[var(--color-border)]">
      <MethodBadge method={method} />
      <code className="text-xs sm:text-sm font-mono font-medium text-zinc-200">{path}</code>
      {description && (
        <span className="text-xs text-zinc-400 sm:ml-auto">{description}</span>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════ */

function Sidebar({ activeId, onNavigate, isOpen, onClose }) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden" onClick={onClose} />
      )}

      <aside
        className={`fixed top-0 left-0 z-50 h-screen w-72 bg-[var(--color-surface)] border-r border-[var(--color-border)] overflow-y-auto transition-transform duration-300 lg:sticky lg:top-0 lg:translate-x-0 lg:z-10 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Logo */}
        <div className="sticky top-0 z-10 bg-[var(--color-surface)]/95 backdrop-blur-md border-b border-[var(--color-border)] px-5 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 group">
            <span className="text-base">⚡</span>
            <span className="text-sm font-bold tracking-tight text-white">Vecta</span>
            <span className="text-[10px] font-mono text-zinc-500 border border-zinc-800 px-1.5 py-0.5 rounded">
              Docs
            </span>
          </Link>
          <Link href="/" className="text-xs text-zinc-500 hover:text-white transition-colors" title="Back to home">
            ← Home
          </Link>
        </div>

        <nav className="p-4 pb-24">
          {SIDEBAR_SECTIONS.map((section) => (
            <div key={section.title} className="mb-6">
              <h3 className="text-[10px] font-mono font-semibold uppercase tracking-[0.18em] text-zinc-500 mb-2 px-2.5">
                {section.title}
              </h3>
              <ul className="space-y-0.5">
                {section.items.map((item) => {
                  const isActive = activeId === item.id;
                  return (
                    <li key={item.id}>
                      <a
                        href={`#${item.id}`}
                        onClick={() => {
                          onNavigate(item.id);
                          onClose();
                        }}
                        className={`block px-2.5 py-1.5 rounded-lg text-xs transition-all duration-150 ${
                          isActive
                            ? "bg-zinc-800/80 text-white font-medium border border-zinc-700/60 shadow-sm"
                            : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900/40"
                        }`}
                      >
                        {item.label}
                      </a>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>
      </aside>
    </>
  );
}


/* ═══════════════════════════════════════════════════════════════
   SEARCH BAR
   ═══════════════════════════════════════════════════════════════ */

function SearchBar({ onSearch }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const inputRef = useRef(null);

  const allItems = SIDEBAR_SECTIONS.flatMap((s) =>
    s.items.map((item) => ({ ...item, section: s.title }))
  );

  useEffect(() => {
    if (query.length > 0) {
      const filtered = allItems.filter(
        (item) =>
          item.label.toLowerCase().includes(query.toLowerCase()) ||
          item.section.toLowerCase().includes(query.toLowerCase()) ||
          item.id.toLowerCase().includes(query.toLowerCase())
      );
      setResults(filtered);
      setIsOpen(true);
    } else {
      setResults([]);
      setIsOpen(false);
    }
  }, [query]);

  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "Escape") {
        setIsOpen(false);
        inputRef.current?.blur();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div className="relative w-full max-w-md">
      <div className="relative">
        <svg
          className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]"
          width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2"
        >
          <circle cx="7" cy="7" r="5" />
          <path d="M11 11l3 3" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search docs..."
          className="w-full pl-9 pr-16 py-2 rounded-xl bg-[var(--color-surface-card)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-muted)] focus:outline-none focus:border-[var(--color-primary)]/50 focus:ring-1 focus:ring-[var(--color-primary)]/20 transition-all"
        />
        <kbd className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-[var(--color-text-muted)] bg-[var(--color-surface)] border border-[var(--color-border)] px-1.5 py-0.5 rounded">
          Ctrl+K
        </kbd>
      </div>

      {isOpen && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)] shadow-2xl shadow-black/40 z-50 max-h-72 overflow-y-auto">
          {results.map((r) => (
            <a
              key={r.id}
              href={`#${r.id}`}
              onClick={() => {
                setIsOpen(false);
                setQuery("");
              }}
              className="flex items-center justify-between px-4 py-2.5 hover:bg-[var(--color-surface-hover)] transition-colors border-b border-[var(--color-border)]/30 last:border-0"
            >
              <span className="text-sm text-[var(--color-text-primary)]">{r.label}</span>
              <span className="text-xs text-[var(--color-text-muted)]">{r.section}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   DOCUMENTATION CONTENT
   ═══════════════════════════════════════════════════════════════ */

function DocsContent() {
  return (
    <div className="docs-prose max-w-none">
      {/* ─── INTRODUCTION ─── */}
      <section id="introduction">
        <h1 className="text-3xl sm:text-4xl font-bold mb-4">
          Vecta Documentation
        </h1>
        <p className="text-lg text-[var(--color-text-secondary)] leading-relaxed mb-6">
          Vecta is a production-grade vector search engine built from scratch in pure Rust. It implements four core
          indexing algorithms—Flat, IVF, HNSW, and IVFPQ—with zero external C/C++ dependencies. This documentation
          covers everything you need to integrate Vecta into your application.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 my-8">
          {[
            { emoji: "🐍", title: "Python Embedded", desc: "Import as a native module", href: "#python-install" },
            { emoji: "🌐", title: "REST API Server", desc: "HTTP server for any language", href: "#server-start" },
            { emoji: "🦜", title: "LangChain RAG", desc: "Drop into AI pipelines", href: "#langchain-setup" },
          ].map((card) => (
            <a
              key={card.title}
              href={card.href}
              className="group p-5 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/50 hover:border-[var(--color-primary)]/40 hover:bg-[var(--color-surface-hover)] transition-all duration-300"
            >
              <div className="text-2xl mb-2">{card.emoji}</div>
              <h3 className="font-semibold text-[var(--color-text-primary)] group-hover:text-[var(--color-primary-light)] transition-colors">
                {card.title}
              </h3>
              <p className="text-xs text-[var(--color-text-muted)] mt-1">{card.desc}</p>
            </a>
          ))}
        </div>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ─── INSTALLATION ─── */}
      <section id="installation">
        <h2 className="text-2xl font-bold mb-4">Installation</h2>

        <h3 className="text-lg font-semibold mb-3 text-[var(--color-text-primary)]">Prerequisites</h3>
        <ul className="list-disc list-inside text-sm text-[var(--color-text-secondary)] space-y-1.5 mb-6">
          <li><strong className="text-[var(--color-text-primary)]">Rust 1.75+</strong> — for building from source</li>
          <li><strong className="text-[var(--color-text-primary)]">Python 3.10+</strong> — for embedded mode and client SDK</li>
          <li><strong className="text-[var(--color-text-primary)]">Docker</strong> — optional, for containerized server deployment</li>
        </ul>

        <h3 className="text-lg font-semibold mb-3 text-[var(--color-text-primary)]">Python Embedded Module</h3>
        <CodeBlock filename="terminal">{`# Clone the repository
git clone https://github.com/dhanushkumar-amk/VECTA.git
cd VECTA

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\\Scripts\\activate       # Windows

# Build and install the native extension
pip install maturin
maturin develop --release`}</CodeBlock>

        <h3 className="text-lg font-semibold mb-3 text-[var(--color-text-primary)]">Server Binary</h3>
        <CodeBlock filename="terminal">{`# Build and run the server
cargo run --release --bin vecta-server

# Or build only
cargo build --release --bin vecta-server`}</CodeBlock>

        <h3 className="text-lg font-semibold mb-3 text-[var(--color-text-primary)]">Python Client SDK</h3>
        <CodeBlock filename="terminal">{`cd clients/python
pip install -e .`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ─── QUICKSTART ─── */}
      <section id="quickstart">
        <h2 className="text-2xl font-bold mb-4">Quickstart</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Get a working vector search in 5 lines of Python:
        </p>

        <CodeBlock filename="quickstart.py" language="python">{`import vecta

# 1. Create a 128-dimensional index with cosine similarity
index = vecta.FlatIndex(dim=128, metric="euclidean")

# 2. Add vectors with unique integer IDs
index.add(0, [0.1] * 128)
index.add(1, [0.9] * 128)

# 3. Search for the nearest neighbor
results = index.search(query=[0.12] * 128, k=1)
print(f"Nearest: ID={results[0][0]}, Distance={results[0][1]:.4f}")
# Output: Nearest: ID=0, Distance=0.0200`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ─── CHOOSING MODE ─── */}
      <section id="choosing-mode">
        <h2 className="text-2xl font-bold mb-4">Embedded vs. Server Mode</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-6 leading-relaxed">
          Both modes consume the exact same Rust core. Choose based on your deployment needs:
        </p>

        <div className="rounded-xl border border-[var(--color-border)] overflow-hidden my-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-surface-card)]/40">
                <th className="text-left px-4 py-3 text-[var(--color-text-muted)] font-medium">Dimension</th>
                <th className="text-left px-4 py-3 text-[var(--color-primary-light)] font-medium">Embedded</th>
                <th className="text-left px-4 py-3 text-[var(--color-accent-light)] font-medium">Server</th>
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-secondary)]">
              {[
                ["Interface", "Python classes (import vecta)", "HTTP REST API"],
                ["Latency", "Sub-microsecond (FFI)", "~0.3 – 1.0ms (HTTP)"],
                ["Language", "Python only", "Any (HTTP/JSON)"],
                ["Durability", "Manual .save()", "WAL + auto recovery"],
                ["Concurrency", "RwLock + GIL release", "Multi-threaded Tokio"],
                ["Best For", "ML pipelines, notebooks", "Microservices, production"],
              ].map(([dim, embedded, server]) => (
                <tr key={dim} className="border-b border-[var(--color-border)]/40 last:border-0">
                  <td className="px-4 py-2.5 font-medium text-[var(--color-text-primary)]">{dim}</td>
                  <td className="px-4 py-2.5">{embedded}</td>
                  <td className="px-4 py-2.5">{server}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ═══════════════════ INDEX TYPES ═══════════════════ */}

      <section id="flat-index">
        <h2 className="text-2xl font-bold mb-4">Flat Index</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Exact brute-force search over all stored vectors. Guarantees <strong className="text-emerald-400">100% recall</strong> with
          zero indexing overhead. Best for datasets under 50k vectors or for ground-truth generation.
        </p>
        <div className="grid grid-cols-3 gap-4 my-5">
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold text-emerald-400">100%</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">Recall</div>
          </div>
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold gradient-text-static">1,413</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">QPS</div>
          </div>
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold text-[var(--color-primary-light)]">1.0×</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">Memory</div>
          </div>
        </div>
        <CodeBlock filename="flat_example.py">{`import vecta

index = vecta.FlatIndex(dim=128, metric="euclidean")
index.add(0, [0.1] * 128)
index.add(1, [0.9] * 128)

results = index.search(query=[0.12] * 128, k=5)
print(f"Total vectors: {len(index)}")`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="ivf-index">
        <h2 className="text-2xl font-bold mb-4">IVF Index</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Inverted File Index using Lloyd&apos;s k-means clustering. Partitions vectors into Voronoi cells,
          then searches only nearby clusters. Requires a <strong className="text-[var(--color-primary-light)]">training step</strong> before
          adding vectors.
        </p>
        <div className="grid grid-cols-3 gap-4 my-5">
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold text-emerald-400">80–98%</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">Recall</div>
          </div>
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold gradient-text-static">19,408</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">QPS</div>
          </div>
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold text-[var(--color-primary-light)]">1.0×</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">Memory</div>
          </div>
        </div>
        <CodeBlock filename="ivf_example.py">{`import vecta
import numpy as np

# Create IVF index with 100 clusters
index = vecta.IVFIndex(dim=128, num_clusters=100, metric="euclidean")

# Train on representative data (REQUIRED before adding)
training_data = np.random.rand(10000, 128).tolist()
training_ids = list(range(10000))
index.train(training_ids, training_data)

# Add more vectors
index.add(10001, [0.5] * 128)

# Search with nprobe=10 (probes 10 of 100 clusters)
results = index.search(query=[0.5] * 128, k=5, nprobe=10)

# Inspect cluster distribution
print(f"Trained: {index.is_trained()}")
print(f"Cluster sizes: {index.cluster_sizes()}")`}</CodeBlock>

        <ParamTable params={[
          { name: "dim", type: "int", required: true, desc: "Vector dimensionality" },
          { name: "num_clusters", type: "int", required: true, desc: "Number of Voronoi partitions (centroids)" },
          { name: "metric", type: "str", required: true, desc: '"euclidean", "cosine", or "dot_product"' },
        ]} />

        <Callout type="tip">
          A good rule of thumb: set <code className="text-[var(--color-accent-light)]">num_clusters ≈ √N</code> where N is the total number of vectors.
          For 100k vectors, use ~316 clusters. For <code className="text-[var(--color-accent-light)]">nprobe</code>, start
          with 5–10 and increase for higher recall.
        </Callout>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="hnsw-index">
        <h2 className="text-2xl font-bold mb-4">HNSW Index</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Hierarchical Navigable Small World graph. Builds a multi-layer skip-list graph for extremely fast
          approximate nearest neighbor search. No training required—vectors are indexed on insertion.
        </p>
        <div className="grid grid-cols-3 gap-4 my-5">
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold text-emerald-400">90–99%</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">Recall</div>
          </div>
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold gradient-text-static">25,967</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">QPS</div>
          </div>
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold text-[var(--color-primary-light)]">1.1–1.3×</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">Memory</div>
          </div>
        </div>
        <CodeBlock filename="hnsw_example.py">{`import vecta

# Create HNSW index (no training needed!)
index = vecta.HnswIndex(
    dim=128,
    metric="cosine",
    m=16,              # Max edges per node (default: 16)
    ef_construction=200 # Build-time beam width (default: 200)
)

# Add vectors — graph is built incrementally
for i in range(10000):
    index.add(i, [float(i % 100) / 100.0] * 128)

# Search with adjustable quality
results = index.search(query=[0.5] * 128, k=10, ef_search=64)

# Inspect graph structure
print(f"Layer distribution: {index.max_layer_distribution()}")`}</CodeBlock>

        <ParamTable params={[
          { name: "dim", type: "int", required: true, desc: "Vector dimensionality" },
          { name: "metric", type: "str", required: true, desc: '"euclidean", "cosine", or "dot_product"' },
          { name: "m", type: "int", required: false, desc: "Max edges per node. Higher = better recall, more memory. Default: 16" },
          { name: "ef_construction", type: "int", required: false, desc: "Build-time beam width. Higher = better graph, slower build. Default: 200" },
          { name: "ef_search", type: "int", required: false, desc: "Search-time beam width. Higher = better recall, slower search. Default: 10" },
        ]} />

        <Callout type="important">
          <code className="text-[var(--color-accent-light)]">ef_search</code> is the main knob for tuning the recall/speed tradeoff at query time.
          Start with <code>ef_search=64</code> and adjust up for higher recall or down for faster queries.
        </Callout>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="ivfpq-index">
        <h2 className="text-2xl font-bold mb-4">IVFPQ Index</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Inverted File with Product Quantization. Combines IVF clustering with PQ compression for
          extreme memory savings. Achieves <strong className="text-emerald-400">19.5× compression</strong> while
          maintaining reasonable recall.
        </p>
        <div className="grid grid-cols-3 gap-4 my-5">
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold text-emerald-400">50–70%</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">Recall</div>
          </div>
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold gradient-text-static">45,780</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">QPS</div>
          </div>
          <div className="p-4 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-card)]/30 text-center">
            <div className="text-xl font-bold text-emerald-400">0.05×</div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">Memory (19.5× compression)</div>
          </div>
        </div>
        <CodeBlock filename="ivfpq_example.py">{`import vecta
import numpy as np

# Create IVFPQ index
index = vecta.IVFPQIndex(
    dim=128,
    num_clusters=100,  # IVF partitions
    num_sub=8,         # PQ sub-vector count (dim must be divisible)
    num_bits=8,        # Bits per sub-quantizer (256 centroids)
    metric="euclidean"
)

# Train on representative data (REQUIRED)
training_data = np.random.rand(10000, 128).tolist()
training_ids = list(range(10000))
index.train(training_ids, training_data)

# Search
results = index.search(query=[0.5] * 128, k=10, nprobe=10)

# Check compression
print(f"Memory footprint: {index.memory_footprint_bytes()} bytes")`}</CodeBlock>

        <Callout type="warning">
          IVFPQ currently only supports the <code className="text-[var(--color-accent-light)]">euclidean</code> metric.
          Cosine and dot product support is planned for v0.2.0.
        </Callout>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="choosing-index">
        <h2 className="text-2xl font-bold mb-4">Choosing an Index</h2>
        <div className="rounded-xl border border-[var(--color-border)] overflow-hidden my-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-surface-card)]/40">
                <th className="text-left px-4 py-3 text-[var(--color-text-muted)] font-medium">Scenario</th>
                <th className="text-left px-4 py-3 text-[var(--color-text-muted)] font-medium">Recommended</th>
                <th className="text-left px-4 py-3 text-[var(--color-text-muted)] font-medium">Why</th>
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-secondary)]">
              {[
                ["< 50k vectors, exact results needed", "FlatIndex", "100% recall, zero setup"],
                ["General production search", "HnswIndex", "Best recall/speed tradeoff, no training"],
                ["Medium dataset, fast approximate", "IVFIndex", "Good recall, fast with tunable nprobe"],
                ["Millions of vectors, tight RAM", "IVFPQIndex", "19.5× compression, still fast"],
                ["Concurrent reads/writes", "ConcurrentFlatIndex", "Thread-safe RwLock design"],
                ["Parallel search across shards", "ShardedFlatIndex", "Hash-based partitioning"],
              ].map(([scenario, rec, why]) => (
                <tr key={scenario} className="border-b border-[var(--color-border)]/40 last:border-0">
                  <td className="px-4 py-2.5">{scenario}</td>
                  <td className="px-4 py-2.5 font-mono text-[var(--color-primary-light)] text-xs">{rec}</td>
                  <td className="px-4 py-2.5">{why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ═══════════════════ PYTHON SDK ═══════════════════ */}

      <section id="python-install">
        <h2 className="text-2xl font-bold mb-4">Python Embedded — Installation</h2>
        <CodeBlock filename="terminal">{`pip install maturin
maturin develop --release

# Verify installation
python -c "import vecta; print(vecta.hello_vecta())"
# Output: vecta engine initialized`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="python-flat">
        <h2 className="text-2xl font-bold mb-4">FlatIndex API</h2>
        <h3 className="text-lg font-semibold mb-3 text-[var(--color-text-primary)]">Constructor</h3>
        <CodeBlock>{`vecta.FlatIndex(dim: int, metric: str) -> FlatIndex`}</CodeBlock>
        <ParamTable params={[
          { name: "dim", type: "int", required: true, desc: "Vector dimensionality (must be > 0)" },
          { name: "metric", type: "str", required: true, desc: '"euclidean" | "l2" | "cosine" | "cos" | "dot_product" | "dot" | "ip"' },
        ]} />

        <h3 className="text-lg font-semibold mb-3 mt-8 text-[var(--color-text-primary)]">Methods</h3>
        <div className="rounded-xl border border-[var(--color-border)] overflow-hidden my-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-surface-card)]/40">
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Method</th>
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Returns</th>
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Description</th>
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-secondary)]">
              {[
                [".add(id, vector)", "None", "Insert a single vector with unique ID"],
                [".add_batch(ids, vectors)", "None", "Bulk insert multiple vectors"],
                [".search(query, k)", "list[(id, dist)]", "Find k nearest neighbors"],
                [".save(path)", "None", "Serialize index to disk"],
                [".len()", "int", "Number of indexed vectors"],
                [".dim()", "int", "Vector dimensionality"],
                [".is_empty()", "bool", "Whether the index has no vectors"],
              ].map(([method, returns, desc]) => (
                <tr key={method} className="border-b border-[var(--color-border)]/40 last:border-0">
                  <td className="px-4 py-2.5 font-mono text-[var(--color-accent-light)] text-xs">{method}</td>
                  <td className="px-4 py-2.5 font-mono text-[var(--color-text-muted)] text-xs">{returns}</td>
                  <td className="px-4 py-2.5">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="python-ivf">
        <h2 className="text-2xl font-bold mb-4">IVFIndex API</h2>
        <CodeBlock>{`vecta.IVFIndex(dim: int, num_clusters: int, metric: str) -> IVFIndex`}</CodeBlock>
        <div className="rounded-xl border border-[var(--color-border)] overflow-hidden my-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-surface-card)]/40">
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Method</th>
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Description</th>
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-secondary)]">
              {[
                [".train(ids, vectors)", "Train k-means centroids (required before add)"],
                [".add(id, vector)", "Insert vector into nearest cluster"],
                [".add_batch(ids, vectors)", "Bulk insert vectors"],
                [".search(query, k, nprobe)", "Search k-NN across nprobe clusters"],
                [".is_trained()", "Check if index has been trained"],
                [".cluster_sizes()", "Get vector counts per cluster"],
                [".nprobe_coverage(query, nprobe)", "Count vectors in top nprobe clusters for a query"],
                [".save(path)", "Serialize to disk"],
              ].map(([method, desc]) => (
                <tr key={method} className="border-b border-[var(--color-border)]/40 last:border-0">
                  <td className="px-4 py-2.5 font-mono text-[var(--color-accent-light)] text-xs whitespace-nowrap">{method}</td>
                  <td className="px-4 py-2.5">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="python-hnsw">
        <h2 className="text-2xl font-bold mb-4">HnswIndex API</h2>
        <CodeBlock>{`vecta.HnswIndex(dim: int, metric: str, m: int = 16, ef_construction: int = 200) -> HnswIndex`}</CodeBlock>
        <div className="rounded-xl border border-[var(--color-border)] overflow-hidden my-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-surface-card)]/40">
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Method</th>
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Description</th>
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-secondary)]">
              {[
                [".add(id, vector)", "Insert and link into HNSW graph"],
                [".add_batch(ids, vectors)", "Bulk insert into graph"],
                [".search(query, k, ef_search)", "Beam search with dynamic candidate list"],
                [".max_layer_distribution()", "Layer-level node count distribution"],
                [".save(path)", "Serialize graph to disk"],
              ].map(([method, desc]) => (
                <tr key={method} className="border-b border-[var(--color-border)]/40 last:border-0">
                  <td className="px-4 py-2.5 font-mono text-[var(--color-accent-light)] text-xs whitespace-nowrap">{method}</td>
                  <td className="px-4 py-2.5">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="python-ivfpq">
        <h2 className="text-2xl font-bold mb-4">IVFPQIndex API</h2>
        <CodeBlock>{`vecta.IVFPQIndex(dim: int, num_clusters: int, num_sub: int, num_bits: int, metric: str) -> IVFPQIndex`}</CodeBlock>
        <ParamTable params={[
          { name: "dim", type: "int", required: true, desc: "Vector dimensionality (must be divisible by num_sub)" },
          { name: "num_clusters", type: "int", required: true, desc: "Number of IVF partitions" },
          { name: "num_sub", type: "int", required: true, desc: "Number of PQ sub-vectors (e.g. 8, 16)" },
          { name: "num_bits", type: "int", required: true, desc: "Bits per sub-quantizer (8 = 256 centroids)" },
          { name: "metric", type: "str", required: true, desc: '"euclidean" only (cosine/dot planned for v0.2)' },
        ]} />
        <div className="rounded-xl border border-[var(--color-border)] overflow-hidden my-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-surface-card)]/40">
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Method</th>
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Description</th>
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-secondary)]">
              {[
                [".train(ids, vectors)", "Train IVF centroids + PQ codebooks"],
                [".add(id, vector)", "Quantize and insert into cluster"],
                [".search(query, k, nprobe)", "ADC table lookup search"],
                [".memory_footprint_bytes()", "Total compressed memory usage"],
                [".save(path)", "Serialize to disk"],
              ].map(([method, desc]) => (
                <tr key={method} className="border-b border-[var(--color-border)]/40 last:border-0">
                  <td className="px-4 py-2.5 font-mono text-[var(--color-accent-light)] text-xs whitespace-nowrap">{method}</td>
                  <td className="px-4 py-2.5">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="python-concurrent">
        <h2 className="text-2xl font-bold mb-4">ConcurrentFlatIndex</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Thread-safe wrapper around FlatIndex using <code className="text-[var(--color-accent-light)]">parking_lot::RwLock</code>.
          Allows concurrent reads with exclusive writes. GIL is released during all operations.
        </p>
        <CodeBlock filename="concurrent_example.py">{`import vecta
from concurrent.futures import ThreadPoolExecutor

index = vecta.ConcurrentFlatIndex(dim=128, metric="cosine")

# Safe to call from multiple threads
def insert(i):
    index.add(i, [float(i)] * 128)

with ThreadPoolExecutor(max_workers=8) as pool:
    pool.map(insert, range(10000))

results = index.search(query=[5.0] * 128, k=5)`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="python-sharded">
        <h2 className="text-2xl font-bold mb-4">ShardedFlatIndex</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Hash-partitioned index that distributes vectors across N shards and executes parallel fan-out search.
        </p>
        <CodeBlock filename="sharded_example.py">{`import vecta

index = vecta.ShardedFlatIndex(dim=128, num_shards=8, metric="euclidean")

for i in range(10000):
    index.add(i, [float(i % 50)] * 128)

results = index.search(query=[25.0] * 128, k=10)
print(f"Shard sizes: {index.shard_sizes()}")
print(f"Num shards: {index.num_shards()}")`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="python-metadata">
        <h2 className="text-2xl font-bold mb-4">MetadataStore</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Attach arbitrary key-value metadata to vector IDs and perform filtered search using filter expressions.
        </p>
        <CodeBlock filename="metadata_example.py">{`import vecta

# Create index and metadata store
index = vecta.FlatIndex(dim=4, metric="euclidean")
meta = vecta.MetadataStore()

# Add vectors with metadata
index.add(1, [1.0, 0.0, 0.0, 0.0])
meta.set(1, "category", "science")
meta.set(1, "year", 2024)

index.add(2, [0.0, 1.0, 0.0, 0.0])
meta.set(2, "category", "art")
meta.set(2, "year", 2023)

# Filtered search — only return vectors matching the filter
results = vecta.filtered_search(
    index,
    meta,
    query=[0.9, 0.1, 0.0, 0.0],
    k=5,
    filter={"field": "category", "op": "eq", "value": "science"}
)`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="python-persistence">
        <h2 className="text-2xl font-bold mb-4">Save & Load</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          All index types support serialization. Use <code className="text-[var(--color-accent-light)]">.save(path)</code> to
          persist and <code className="text-[var(--color-accent-light)]">vecta.load(path)</code> to restore. The load function
          auto-detects the index type.
        </p>
        <CodeBlock filename="persistence.py">{`import vecta

# Save any index type
index = vecta.HnswIndex(dim=128, metric="cosine")
index.add(0, [0.1] * 128)
index.save("my_index.vecta")

# Load auto-detects the index type
loaded = vecta.load("my_index.vecta")
results = loaded.search(query=[0.1] * 128, k=1)`}</CodeBlock>

        <Callout type="note">
          The <code className="text-[var(--color-accent-light)]">vecta.load(path)</code> function returns the appropriate
          index type (FlatIndex, HnswIndex, etc.) automatically based on the serialized header.
        </Callout>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ═══════════════════ REST API SERVER ═══════════════════ */}

      <section id="server-start">
        <h2 className="text-2xl font-bold mb-4">Starting the Server</h2>

        <h3 className="text-lg font-semibold mb-3 text-[var(--color-text-primary)]">With Cargo</h3>
        <CodeBlock filename="terminal">{`cargo run --release --bin vecta-server
# Server starts at http://localhost:6333
# Swagger UI at http://localhost:6333/docs`}</CodeBlock>

        <h3 className="text-lg font-semibold mb-3 mt-6 text-[var(--color-text-primary)]">With Docker</h3>
        <CodeBlock filename="terminal">{`docker build -t vecta .
docker run -d \\
  -p 6333:6333 \\
  -v $(pwd)/data:/data \\
  -e VECTA_API_KEY=my_secret_key \\
  -e VECTA_PORT=6333 \\
  -e VECTA_DATA_DIR=/data \\
  vecta`}</CodeBlock>

        <Callout type="tip">
          Open <strong>http://localhost:6333/docs</strong> in your browser for an interactive Swagger UI where you can
          test every endpoint with live requests.
        </Callout>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="server-auth">
        <h2 className="text-2xl font-bold mb-4">Authentication</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Set the <code className="text-[var(--color-accent-light)]">VECTA_API_KEY</code> environment variable to enable Bearer token
          authentication. All endpoints except <code>/health</code> and <code>/docs</code> require authentication when enabled.
        </p>
        <CodeBlock filename="terminal">{`# Set the API key
export VECTA_API_KEY=my_secret_key

# Include in requests
curl -H "Authorization: Bearer my_secret_key" http://localhost:6333/collections`}</CodeBlock>

        <Callout type="note">
          If <code className="text-[var(--color-accent-light)]">VECTA_API_KEY</code> is not set, all endpoints are publicly
          accessible (useful for local development).
        </Callout>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ─── API ENDPOINTS ─── */}

      <section id="api-health">
        <h2 className="text-2xl font-bold mb-4">Health Check</h2>
        <EndpointHeader method="GET" path="/health" description="Check server liveness" />
        <p className="text-sm text-[var(--color-text-secondary)] mb-4">No authentication required.</p>
        <CodeBlock filename="request">{`curl http://localhost:6333/health`}</CodeBlock>
        <CodeBlock filename="response.json">{`{
  "status": "ok"
}`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="api-create-collection">
        <h2 className="text-2xl font-bold mb-4">Create Collection</h2>
        <EndpointHeader method="POST" path="/collections" description="Create a new vector collection" />

        <h3 className="text-lg font-semibold mb-3 text-[var(--color-text-primary)]">Request Body</h3>
        <ParamTable params={[
          { name: "name", type: "string", required: true, desc: "Unique collection name" },
          { name: "dim", type: "integer", required: true, desc: "Vector dimensionality" },
          { name: "index_type", type: "string", required: true, desc: '"flat" | "ivf" | "hnsw" | "ivfpq"' },
          { name: "metric", type: "string", required: true, desc: '"euclidean" | "l2" | "cosine" | "cos" | "dot_product" | "dot"' },
        ]} />

        <CodeBlock filename="request">{`curl -X POST http://localhost:6333/collections \\
  -H "Authorization: Bearer my_secret_key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "documents",
    "dim": 128,
    "index_type": "hnsw",
    "metric": "cosine"
  }'`}</CodeBlock>

        <CodeBlock filename="response.json">{`{
  "name": "documents",
  "index_type": "hnsw",
  "dim": 128,
  "metric": "cosine",
  "vector_count": 0
}`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="api-list-collections">
        <h2 className="text-2xl font-bold mb-4">List Collections</h2>
        <EndpointHeader method="GET" path="/collections" description="List all collections" />
        <CodeBlock filename="request">{`curl http://localhost:6333/collections \\
  -H "Authorization: Bearer my_secret_key"`}</CodeBlock>
        <CodeBlock filename="response.json">{`[
  {
    "name": "documents",
    "index_type": "hnsw",
    "dim": 128,
    "metric": "cosine",
    "vector_count": 1500
  }
]`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="api-get-collection">
        <h2 className="text-2xl font-bold mb-4">Get Collection</h2>
        <EndpointHeader method="GET" path="/collections/:name" description="Get collection metadata" />
        <CodeBlock filename="request">{`curl http://localhost:6333/collections/documents \\
  -H "Authorization: Bearer my_secret_key"`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="api-delete-collection">
        <h2 className="text-2xl font-bold mb-4">Delete Collection</h2>
        <EndpointHeader method="DELETE" path="/collections/:name" description="Delete a collection and its data" />
        <CodeBlock filename="request">{`curl -X DELETE http://localhost:6333/collections/documents \\
  -H "Authorization: Bearer my_secret_key"`}</CodeBlock>
        <Callout type="warning">
          This permanently deletes the collection and all associated vectors. The operation cannot be undone.
        </Callout>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="api-insert">
        <h2 className="text-2xl font-bold mb-4">Insert Point</h2>
        <EndpointHeader method="POST" path="/collections/:name/points" description="Insert a vector into a collection" />

        <ParamTable params={[
          { name: "id", type: "integer", required: true, desc: "Unique external vector identifier (u64)" },
          { name: "vector", type: "float[]", required: true, desc: "Coordinate array matching collection dimensionality" },
        ]} />

        <CodeBlock filename="request">{`curl -X POST http://localhost:6333/collections/documents/points \\
  -H "Authorization: Bearer my_secret_key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "id": 101,
    "vector": [0.1, 0.2, 0.8, 0.5, ...]
  }'`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="api-search">
        <h2 className="text-2xl font-bold mb-4">Search</h2>
        <EndpointHeader method="POST" path="/collections/:name/search" description="k-NN search query" />

        <ParamTable params={[
          { name: "vector", type: "float[]", required: true, desc: "Query vector coordinates" },
          { name: "k", type: "integer", required: true, desc: "Number of nearest neighbors to return" },
          { name: "nprobe", type: "integer", required: false, desc: "Clusters to probe (IVF/IVFPQ only)" },
          { name: "ef_search", type: "integer", required: false, desc: "Beam width (HNSW only)" },
        ]} />

        <CodeBlock filename="request">{`curl -X POST http://localhost:6333/collections/documents/search \\
  -H "Authorization: Bearer my_secret_key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "vector": [0.12, 0.19, 0.78, 0.52, ...],
    "k": 5,
    "ef_search": 64
  }'`}</CodeBlock>

        <CodeBlock filename="response.json">{`{
  "results": [
    { "id": 101, "score": 0.0142 },
    { "id": 203, "score": 0.0389 },
    { "id": 57,  "score": 0.0521 }
  ]
}`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="api-checkpoint">
        <h2 className="text-2xl font-bold mb-4">Checkpoint</h2>
        <EndpointHeader method="POST" path="/collections/:name/checkpoint" description="Force snapshot save & WAL truncation" />
        <CodeBlock filename="request">{`curl -X POST http://localhost:6333/collections/documents/checkpoint \\
  -H "Authorization: Bearer my_secret_key"`}</CodeBlock>
        <Callout type="note">
          FlatIndex collections use WAL with automatic crash recovery. HNSW, IVF, and IVFPQ collections
          persist via explicit checkpoint calls and graceful shutdown signal handlers.
        </Callout>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ═══════════════════ PYTHON CLIENT SDK ═══════════════════ */}

      <section id="client-install">
        <h2 className="text-2xl font-bold mb-4">Python Client SDK — Installation</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          A pure-Python HTTP client for interacting with the Vecta server. Zero dependencies beyond <code className="text-[var(--color-accent-light)]">requests</code>.
        </p>
        <CodeBlock filename="terminal">{`cd clients/python
pip install -e .`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="client-usage">
        <h2 className="text-2xl font-bold mb-4">Python Client — Usage Guide</h2>
        <CodeBlock filename="client_example.py">{`from vecta_client import VectaClient

# Connect to running server
client = VectaClient(
    base_url="http://localhost:6333",
    api_key="my_secret_key",
    timeout=10.0  # seconds
)

# Health check
print(client.health())  # {'status': 'ok'}

# Create a collection
client.create_collection(
    name="products",
    dim=128,
    index_type="hnsw",
    metric="cosine"
)

# Insert vectors
client.insert(collection="products", id=1, vector=[0.1] * 128)
client.insert(collection="products", id=2, vector=[0.9] * 128)

# Search (returns list of {id, score} dicts)
results = client.search(
    collection="products",
    vector=[0.12] * 128,
    k=5,
    ef_search=64  # HNSW parameter
)

for match in results:
    print(f"ID: {match['id']}, Score: {match['score']:.4f}")

# Force save to disk
client.checkpoint(collection="products")

# List and inspect collections
collections = client.list_collections()
info = client.get_collection("products")

# Clean up
client.delete_collection("products")`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="client-api">
        <h2 className="text-2xl font-bold mb-4">Python Client — API Reference</h2>
        <div className="rounded-xl border border-[var(--color-border)] overflow-hidden my-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-surface-card)]/40">
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Method</th>
                <th className="text-left px-4 py-2.5 text-[var(--color-text-muted)] font-medium">Description</th>
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-secondary)]">
              {[
                ["client.health()", "Check server liveness"],
                ["client.create_collection(name, dim, index_type, metric)", "Create new collection"],
                ["client.list_collections()", "List all collections"],
                ["client.get_collection(name)", "Get collection metadata"],
                ["client.delete_collection(name)", "Delete a collection"],
                ["client.insert(collection, id, vector)", "Insert a vector"],
                ["client.search(collection, vector, k, nprobe?, ef_search?)", "k-NN search"],
                ["client.checkpoint(collection)", "Force save to disk"],
              ].map(([method, desc]) => (
                <tr key={method} className="border-b border-[var(--color-border)]/40 last:border-0">
                  <td className="px-4 py-2.5 font-mono text-[var(--color-accent-light)] text-xs">{method}</td>
                  <td className="px-4 py-2.5">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Callout type="tip">
          All methods raise <code className="text-[var(--color-accent-light)]">VectaAPIError</code> on non-2xx responses,
          which includes <code>status_code</code> and <code>message</code> attributes.
        </Callout>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ═══════════════════ LANGCHAIN ═══════════════════ */}

      <section id="langchain-setup">
        <h2 className="text-2xl font-bold mb-4">LangChain — Setup</h2>
        <CodeBlock filename="terminal">{`# Install dependencies
pip install langchain langchain-core langchain-community openai

# Install Vecta client
cd clients/python && pip install -e .

# Start the server
cargo run --release --bin vecta-server`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="langchain-usage">
        <h2 className="text-2xl font-bold mb-4">LangChain — Usage</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Vecta provides a <code className="text-[var(--color-accent-light)]">VectaVectorStore</code> class implementing
          LangChain&apos;s <code>VectorStore</code> interface. It supports <code>add_texts()</code>,{" "}
          <code>similarity_search()</code>, and <code>from_texts()</code>.
        </p>

        <CodeBlock filename="langchain_example.py">{`from vecta_client import VectaClient
from vecta_client.langchain import VectaVectorStore
from langchain_community.embeddings import OpenAIEmbeddings

# Initialize
client = VectaClient(base_url="http://localhost:6333", api_key="my_secret_key")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create collection first
client.create_collection(name="kb", dim=1536, index_type="hnsw", metric="cosine")

# Create vector store
store = VectaVectorStore(
    client=client,
    collection="kb",
    embedding=embeddings
)

# Add documents
ids = store.add_texts(
    texts=[
        "Vecta is a vector database built in Rust.",
        "HNSW provides fast approximate nearest neighbor search.",
        "Product Quantization compresses vectors by 19x.",
    ],
    metadatas=[
        {"source": "docs", "topic": "overview"},
        {"source": "docs", "topic": "hnsw"},
        {"source": "docs", "topic": "compression"},
    ]
)

# Search
docs = store.similarity_search("How does compression work?", k=2)
for doc in docs:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\\n")`}</CodeBlock>

        <h3 className="text-lg font-semibold mb-3 mt-8 text-[var(--color-text-primary)]">Factory Constructor</h3>
        <CodeBlock filename="from_texts.py">{`# One-step creation + ingestion
store = VectaVectorStore.from_texts(
    texts=["doc1", "doc2", "doc3"],
    embedding=embeddings,
    base_url="http://localhost:6333",
    collection="auto_collection",
    index_type="flat",
    metric="euclidean"
)`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="langchain-rag">
        <h2 className="text-2xl font-bold mb-4">LangChain — RAG Pipeline</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4 leading-relaxed">
          Use Vecta as a retriever in a full Retrieval Augmented Generation pipeline:
        </p>
        <CodeBlock filename="rag_pipeline.py">{`from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

# Use the vector store as a retriever
retriever = store.as_retriever(search_kwargs={"k": 3})

# Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# Ask questions over your documents
result = qa_chain.invoke({"query": "What index architectures does Vecta support?"})
print(result["result"])
print(result["source_documents"])`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      {/* ═══════════════════ DEPLOYMENT ═══════════════════ */}

      <section id="docker">
        <h2 className="text-2xl font-bold mb-4">Docker</h2>
        <CodeBlock filename="terminal">{`# Build the image
docker build -t vecta .

# Run with persistent storage
docker run -d \\
  --name vecta-server \\
  -p 6333:6333 \\
  -v $(pwd)/data:/data \\
  -e VECTA_API_KEY=my_secret_key \\
  -e VECTA_PORT=6333 \\
  -e VECTA_DATA_DIR=/data \\
  --restart unless-stopped \\
  vecta`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="docker-compose">
        <h2 className="text-2xl font-bold mb-4">Docker Compose</h2>
        <CodeBlock filename="docker-compose.yml">{`services:
  vecta:
    build:
      context: .
      dockerfile: Dockerfile
    image: vecta:local
    container_name: vecta-server
    ports:
      - "6333:6333"
    volumes:
      - ./data:/data
    environment:
      - VECTA_PORT=6333
      - VECTA_DATA_DIR=/data
      - VECTA_API_KEY=my_secret_key
    restart: unless-stopped`}</CodeBlock>
        <CodeBlock filename="terminal">{`docker-compose up -d`}</CodeBlock>
      </section>

      <hr className="border-[var(--color-border)] my-10" />

      <section id="env-vars">
        <h2 className="text-2xl font-bold mb-4">Environment Variables</h2>
        <div className="rounded-xl border border-[var(--color-border)] overflow-hidden my-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-surface-card)]/40">
                <th className="text-left px-4 py-3 text-[var(--color-text-muted)] font-medium">Variable</th>
                <th className="text-left px-4 py-3 text-[var(--color-text-muted)] font-medium">Default</th>
                <th className="text-left px-4 py-3 text-[var(--color-text-muted)] font-medium">Description</th>
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-secondary)]">
              {[
                ["VECTA_PORT", "6333", "HTTP server listen port"],
                ["VECTA_DATA_DIR", "./data", "Directory for WAL, snapshots, and collections"],
                ["VECTA_API_KEY", "(none)", "Bearer token for authentication. If unset, auth is disabled"],
              ].map(([name, def, desc]) => (
                <tr key={name} className="border-b border-[var(--color-border)]/40 last:border-0">
                  <td className="px-4 py-2.5 font-mono text-[var(--color-accent-light)] text-xs">{name}</td>
                  <td className="px-4 py-2.5 font-mono text-[var(--color-text-muted)] text-xs">{def}</td>
                  <td className="px-4 py-2.5">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Footer spacer */}
      <div className="h-20" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN PAGE
   ═══════════════════════════════════════════════════════════════ */

export default function DocsPage() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeId, setActiveId] = useState("introduction");

  // Intersection Observer for active section tracking
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        }
      },
      { rootMargin: "-80px 0px -70% 0px", threshold: 0 }
    );

    const sections = document.querySelectorAll("section[id]");
    sections.forEach((s) => observer.observe(s));

    return () => observer.disconnect();
  }, []);

  return (
    <div className="flex min-h-screen">
      <Sidebar
        activeId={activeId}
        onNavigate={setActiveId}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex-1 min-w-0">
        {/* Top bar */}
        <header className="sticky top-0 z-30 glass-nav border-b border-[var(--color-border)] px-6 py-2.5 flex items-center gap-4">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="lg:hidden text-[var(--color-text-muted)] hover:text-white cursor-pointer p-1"
            aria-label="Toggle sidebar"
          >
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <SearchBar />
          <div className="hidden sm:flex items-center gap-4 ml-auto">
            <a
              href="https://github.com/dhanushkumar-amk/VECTA"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-white transition-colors"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              GitHub
            </a>
            <Link
              href="/"
              className="text-xs text-zinc-400 hover:text-white transition-colors"
            >
              ← Home
            </Link>
          </div>
        </header>

        {/* Content area */}
        <main className="max-w-4xl mx-auto px-6 sm:px-12 py-10">
          <DocsContent />
        </main>
      </div>
    </div>
  );
}
