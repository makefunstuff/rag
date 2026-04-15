# rag — contextual retrieval

Hybrid BM25 + semantic search with LLM-generated chunk descriptions.
Based on [Anthropic's Contextual Retrieval cookbook](https://docs.anthropic.com/en/docs/build-with-claude/retrieval-augmented-generation).

Works with any OpenAI-compatible API: OpenRouter, OpenAI, Ollama, LM Studio, vLLM, Together.
Embeddings are handled locally by ChromaDB (no external embedding API needed).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env   # configure API_BASE, API_KEY, LLM_MODEL
```

## Usage

```bash
export API_BASE=https://openrouter.ai/api/v1
export API_KEY=your-key

rag ingest ~/Documents/my-docs/          # chunk → contextualize → embed → store
rag search "contextual embeddings" -k 5  # hybrid search (no LLM)
rag query "how does prompt caching work" # search + LLM answer
rag eval --pairs eval.json               # recall@K metrics
```

### Provider examples

```bash
# OpenRouter
API_BASE=https://openrouter.ai/api/v1 API_KEY=your-key LLM_MODEL=minimax/minimax-m2.7

# OpenAI
API_BASE=https://api.openai.com/v1 API_KEY=sk-... LLM_MODEL=gpt-4o-mini

# Local llama.cpp / Ollama (no key needed)
API_BASE=http://localhost:1234/v1 API_KEY= LLM_MODEL=your-model-name
```

## Architecture

```
rag/
├── core.py    # config, LLM client, chunking, contextualizer
├── db.py      # ChromaDB storage + BM25 + hybrid search
└── cli.py     # ingest / search / query / eval subcommands
pi/
├── extension.ts  # pi agent tools (rag_query, rag_search, rag_ingest)
└── rag/SKILL.md  # pi skill description
```

### Key technique

Each chunk gets a 1-2 sentence LLM-generated description prepended before embedding.
This "situates" the chunk in its document, improving retrieval ~35% ([source](https://www.anthropic.com/news/contextual-retrieval)).

### Search pipeline

1. **Semantic**: cosine similarity via ChromaDB (built-in embeddings, HNSW index)
2. **BM25**: keyword scoring over stored documents
3. **RRF fusion**: weighted merge (80% semantic, 20% BM25)

## pi integration

Add to `~/.pi/agent/settings.json`:

```json
{
  "extensions": ["/path/to/rag/pi/extension.ts"],
  "skills": ["/path/to/rag/pi"]
}
```

Set `RAG_DIR` env var if the repo isn't at `~/Work/rag`.

## Config

| Env var | Default | Description |
|---------|---------|-------------|
| `API_BASE` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `API_KEY` | — | API key (optional for local models) |
| `LLM_MODEL` | `minimax/minimax-m2.7` | Chat model for contextualization + query |
| `RAG_DATA_DIR` | `./data` | ChromaDB + cache directory |
| `RAG_CHUNK_SIZE` | `512` | Target chunk size in tokens |
