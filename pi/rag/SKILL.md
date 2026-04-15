---
name: rag
description: Search and query a personal knowledge base using contextual RAG (hybrid BM25 + semantic search with LLM-generated chunk descriptions). Use when asked to look up facts, find specific passages, or answer questions grounded in ingested documents.
---

# RAG — Contextual Retrieval

## Tools

This skill provides three pi tools: `rag_query`, `rag_search`, `rag_ingest`.

### Query (rag_query)
Answer a question grounded in the knowledge base:
```
rag_query "how does prompt caching work?"
```

### Search (rag_search)
Find relevant chunks without LLM summarization:
```
rag_search "contextual embeddings" -k 5
```

### Ingest (rag_ingest)
Add files to the corpus. **Ask the user first** — each document costs one LLM call for contextual embedding.
```
rag_ingest ~/Documents/my-docs/ --glob "**/*.md"
```

## When to use

- User asks a factual question that might be in their KB
- User wants to find a specific passage or code snippet
- User says "search my docs for..." or "what do my notes say about..."

## When NOT to use

- General knowledge questions (use your training data)
- Questions about the current codebase (use `read` / `bash`)
