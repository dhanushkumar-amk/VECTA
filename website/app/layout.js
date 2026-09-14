import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

export const metadata = {
  title: "Vecta — Fast Vector Search Engine Built in Rust",
  description:
    "A production-grade vector database engine built from scratch in pure Rust. 4 index architectures, Python bindings, REST API, and LangChain integration. Open source.",
  keywords: [
    "vector database",
    "Rust",
    "HNSW",
    "IVF",
    "product quantization",
    "similarity search",
    "embeddings",
    "RAG",
    "LangChain",
  ],
  openGraph: {
    title: "Vecta — Fast Vector Search Engine Built in Rust",
    description:
      "Production-grade vector search with 4 index architectures. Pure Rust core, Python bindings, REST API, Docker ready.",
    type: "website",
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
