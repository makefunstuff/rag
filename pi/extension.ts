import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const RAG_DIR = process.env.RAG_DIR || join(homedir(), "Work/rag");

async function rag(pi: ExtensionAPI, cmd: string, args: string[], timeout = 120_000) {
  const venv = join(RAG_DIR, ".venv/bin/rag");
  if (!existsSync(venv))
    throw new Error(`rag not found. Run: cd ${RAG_DIR} && python3 -m venv .venv && .venv/bin/pip install -e .`);
  const r = await pi.exec("bash", ["-c",
    `set -a; [ -f ~/.env ] && source ~/.env; set +a; cd ${RAG_DIR} && .venv/bin/rag ${cmd} ${args.join(" ")}`
  ], { timeout, env: { ...process.env } });
  if (r.code !== 0) throw new Error(`rag ${cmd} failed: ${r.stderr?.trim() || r.stdout?.trim()}`);
  return r.stdout.trim();
}

function pretty(s: string) { try { return JSON.stringify(JSON.parse(s), null, 2); } catch { return s; } }

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "rag_query",
    label: "RAG Query",
    description:
      "Query the personal knowledge base using RAG (hybrid BM25 + semantic search + LLM answer). " +
      "Use for factual questions where you want an answer grounded in the user's KB, " +
      "not general knowledge. ",
    promptSnippet: "Search and query a personal knowledge base (RAG with hybrid BM25+semantic search, contextual chunk embeddings)",
    promptGuidelines: [
      "Use rag_query for factual lookups in the personal KB before answering from memory",
      "Use rag_search for retrieving chunks without LLM summarization",
      "Use rag_ingest to add new files to the knowledge base before querying",
    ],
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Question to answer using the knowledge base" },
        top_k: { type: "number", description: "Number of chunks to retrieve (default: 20, higher = more context)", default: 20 },
      },
      required: ["query"],
    },
    async execute(_id, p) {
      const text = await rag(pi, "query", [p.query, "--top-k", String(p.top_k ?? 20), "--json"]);
      const data = JSON.parse(text);
      return { content: [{ type: "text", text: `Q: ${p.query}\n\nA: ${data.answer}` }], details: data };
    },
  });

  pi.registerTool({
    name: "rag_search",
    label: "RAG Search",
    description:
      "Search the personal knowledge base (hybrid BM25 + semantic, RRF fusion) without LLM summarization. " +
      "Returns ranked chunks with fused score, BM25 score, and semantic score. " +
      "Use for exploring what the KB contains or when you want to see raw retrieved context.",
    promptSnippet: "Search a personal knowledge base: hybrid BM25 + semantic search, returns ranked chunks with scores",
    promptGuidelines: [
      "Use rag_search to explore KB contents or see raw retrieved chunks",
      "Use rag_query for full LLM-grounded answers",
    ],
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        k: { type: "number", description: "Number of results to return (default: 10)", default: 10 },
      },
      required: ["query"],
    },
    async execute(_id, p) {
      const text = await rag(pi, "search", [p.query, "-k", String(p.k ?? 10), "--json"]);
      const results = JSON.parse(text);
      const lines = results.map((r: any, i: number) =>
        `${i + 1}. [${r.doc}] fused=${r.fused} bm25=${r.bm25} sem=${r.semantic}\n   ${r.text}`);
      return { content: [{ type: "text", text: lines.join("\n\n") }], details: { results } };
    },
  });

  pi.registerTool({
    name: "rag_ingest",
    label: "RAG Ingest",
    description:
      "Ingest files into the personal knowledge base: chunk → contextualize (LLM) → embed → store. " +
      "Supports text, markdown, code, configs, and more. " +
      "Ask before ingesting — it calls the LLM to generate contextual headers for each chunk (costs API credits).",
    promptSnippet: "Ingest files into the personal knowledge base (chunking, contextual embedding, storage)",
    promptGuidelines: [
      "Ask user before ingesting — each chunk requires an LLM call for contextual embedding",
      "Ingest supports glob patterns, e.g. path='~/docs/', glob='**/*.md'",
      "Run rag_search after ingest to verify files were indexed",
    ],
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File or directory to ingest" },
        glob: { type: "string", description: "Glob pattern when path is a directory (default: **/*)", default: "**/*" },
      },
      required: ["path"],
    },
    async execute(_id, p) {
      const args = [p.path];
      if (p.glob) args.push("--glob", p.glob);
      const text = await rag(pi, "ingest", args, 300_000);
      return { content: [{ type: "text", text }], details: { output: text } };
    },
  });
}
