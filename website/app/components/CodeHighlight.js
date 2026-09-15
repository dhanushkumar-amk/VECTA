"use client";

import { useState } from "react";

// Minimal, reliable token parser for Python, Bash/cURL, and JSON
export function tokenizeCode(code = "") {
  if (!code) return [];

  const rules = [
    // Comments (# or //)
    { type: "comment", regex: /^(?:#|\/\/)[^\n]*/ },
    // Triple-quoted strings
    { type: "string", regex: /^(?:"""[\s\S]*?"""|'''[\s\S]*?''')/ },
    // Strings (with escaped quotes, f-strings, r-strings)
    { type: "string", regex: /^(?:[frbFRB]?"(?:\\.|[^"\\])*"|[frbFRB]?'(?:\\.|[^'\\])*')/ },
    // Numbers
    { type: "number", regex: /^(?:0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)/ },
    // Keywords
    {
      type: "keyword",
      regex: /^\b(?:import|from|as|def|class|return|for|in|if|else|elif|while|with|try|except|finally|raise|yield|async|await|pass|break|continue|lambda)\b/
    },
    // Booleans & Null
    { type: "boolean", regex: /^\b(?:True|False|None|true|false|null)\b/ },
    // Shell executables
    { type: "command", regex: /^\b(?:cargo|curl|pip|maturin|docker|cd|git|python|python3)\b/ },
    // Flags
    { type: "flag", regex: /^--?[a-zA-Z0-9_\-]+/ },
    // HTTP Verbs
    { type: "method", regex: /^\b(?:GET|POST|PUT|DELETE|PATCH)\b/ },
    // Classes & Types
    {
      type: "type",
      regex: /^\b(?:FlatIndex|ConcurrentFlatIndex|ShardedFlatIndex|IVFIndex|HnswIndex|IVFPQIndex|MetadataStore|VectaClient|VectaVectorStore|OpenAIEmbeddings|int|str|float|bool|list|dict)\b/
    },
    // Function calls
    { type: "function", regex: /^[a-zA-Z_]\w*(?=\s*\()/ },
    // Named parameters in calls
    { type: "param", regex: /^[a-zA-Z_]\w*(?=\s*=)/ },
    // Identifiers
    { type: "identifier", regex: /^[a-zA-Z_]\w*/ },
    // Operators
    { type: "operator", regex: /^(?:==|!=|<=|>=|=>|->|\+=|-=|\*=|\/=|&&|\|\||[=+\-*/<>%!&|^~])/ },
    // Punctuation
    { type: "punctuation", regex: /^[()[\]{},.:;]/ },
    // Whitespace
    { type: "whitespace", regex: /^\s+/ },
    // Fallback
    { type: "other", regex: /^./ }
  ];

  const tokens = [];
  let remaining = code;

  while (remaining.length > 0) {
    let matched = false;
    for (const rule of rules) {
      const match = remaining.match(rule.regex);
      if (match) {
        tokens.push({ type: rule.type, text: match[0] });
        remaining = remaining.slice(match[0].length);
        matched = true;
        break;
      }
    }
    if (!matched) {
      tokens.push({ type: "other", text: remaining[0] });
      remaining = remaining.slice(1);
    }
  }

  return tokens;
}

export function HighlightedCode({ code = "" }) {
  const tokens = tokenizeCode(code);

  const getColor = (type) => {
    switch (type) {
      case "comment":
        return "text-zinc-500 italic";
      case "string":
        return "text-emerald-400";
      case "number":
        return "text-amber-400";
      case "keyword":
        return "text-purple-400 font-medium";
      case "boolean":
        return "text-rose-400";
      case "command":
        return "text-cyan-400 font-medium";
      case "flag":
        return "text-teal-300";
      case "method":
        return "text-emerald-300 font-bold";
      case "type":
        return "text-sky-300 font-medium";
      case "function":
        return "text-indigo-300";
      case "param":
        return "text-violet-300";
      case "operator":
      case "punctuation":
        return "text-zinc-500";
      case "identifier":
        return "text-zinc-200";
      default:
        return "text-zinc-300";
    }
  };

  return (
    <>
      {tokens.map((tok, i) => (
        <span key={i} className={getColor(tok.type)}>
          {tok.text}
        </span>
      ))}
    </>
  );
}

export default function MinimalCodeWindow({ code, filename = "example.py", language = "python" }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  };

  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-[#09090e] overflow-hidden shadow-2xl transition-all duration-300">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[var(--color-border)] bg-[#0e0e14]">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]/60" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#febc2e]/60" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#28c840]/60" />
          </div>
          <span className="text-[11px] text-[var(--color-text-muted)] ml-2 font-mono">
            {filename}
          </span>
        </div>

        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 rounded text-xs text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/60 transition-colors cursor-pointer"
          title="Copy to clipboard"
        >
          {copied ? (
            <>
              <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" className="text-emerald-400">
                <path d="M2 6.5l3 3 5.5-5.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span className="text-[11px] text-emerald-400 font-mono">Copied</span>
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
      </div>

      {/* Code contents */}
      <pre className="p-5 overflow-x-auto text-xs sm:text-sm leading-relaxed font-mono selection:bg-indigo-500/20">
        <code>
          <HighlightedCode code={code} />
        </code>
      </pre>
    </div>
  );
}
