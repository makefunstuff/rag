"""Chroma-backed vector store with hybrid BM25 + semantic search."""
import math, re
import chromadb
from .core import DATA_DIR

_TOK = re.compile(r"(?u)\w+")


def get_db() -> chromadb.ClientAPI:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))


def get_collection(db: chromadb.ClientAPI | None = None):
    """ChromaDB embeds documents automatically with its built-in model."""
    db = db or get_db()
    return db.get_or_create_collection("chunks")


def upsert(chunks: list[dict], ctx_texts: list[str]):
    """Store chunks — ChromaDB generates embeddings automatically."""
    col = get_collection()
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=ctx_texts,
        metadatas=[{"doc": c["doc"], "path": c["path"], "index": c["index"],
                    "text": c["text"]} for c in chunks],
    )
    return len(chunks)


def delete_by_prefix(prefix: str) -> int:
    """Delete all chunks whose ID starts with prefix (one file's chunks)."""
    col = get_collection()
    all_ids = col.get()["ids"]
    to_delete = [cid for cid in all_ids if cid.startswith(prefix)]
    if to_delete:
        col.delete(ids=to_delete)
    return len(to_delete)


def semantic_search(query: str, top_k: int = 100) -> list[tuple[str, float]]:
    """Cosine similarity search via Chroma (embeds query automatically)."""
    col = get_collection()
    if col.count() == 0:
        return []
    results = col.query(query_texts=[query], n_results=min(top_k, col.count()),
                        include=["distances"])
    return list(zip(results["ids"][0], [1 - d for d in results["distances"][0]]))


class BM25:
    """In-memory BM25 over stored chunks. Fine for <50K chunks."""

    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self.tfs: list[dict[str, int]] = []
        self.dls: list[int] = []
        self.dfs: dict[str, int] = {}
        for text in texts:
            toks = _TOK.findall(text.lower())
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.tfs.append(tf)
            self.dls.append(len(toks))
        self.avgdl = sum(self.dls) / max(len(self.dls), 1)
        for tf in self.tfs:
            for t in tf:
                self.dfs[t] = self.dfs.get(t, 0) + 1

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        qt = _TOK.findall(query.lower())
        n = len(self.ids)
        scores = []
        for i in range(n):
            s = 0.0
            for t in qt:
                df = self.dfs.get(t, 0)
                if not df:
                    continue
                tf = self.tfs[i].get(t, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                s += idf * (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * self.dls[i] / max(self.avgdl, 1)))
            scores.append(s)
        return sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)[:top_k]


def hybrid_search(query: str, top_k: int = 10,
                  sem_weight: float = 0.8, bm25_weight: float = 0.2,
                  rrf_k: int = 60) -> list[dict]:
    """Hybrid BM25 + semantic search with RRF fusion."""
    col = get_collection()
    if col.count() == 0:
        return []

    sem_hits = semantic_search(query, top_k=100)

    all_data = col.get(include=["documents", "metadatas"])
    bm25 = BM25(all_data["ids"], all_data["documents"])
    bm25_hits = bm25.search(query, top_k=100)

    # RRF fusion
    fused: dict[str, float] = {}
    bscores: dict[str, float] = {}
    sscores: dict[str, float] = {}
    for rank, (cid, s) in enumerate(bm25_hits):
        fused[cid] = fused.get(cid, 0) + bm25_weight / (rrf_k + rank + 1)
        bscores[cid] = s
    for rank, (cid, s) in enumerate(sem_hits):
        fused[cid] = fused.get(cid, 0) + sem_weight / (rrf_k + rank + 1)
        sscores[cid] = s

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    meta_map = dict(zip(all_data["ids"], all_data["metadatas"]))
    doc_map = dict(zip(all_data["ids"], all_data["documents"]))
    return [{"id": cid, "fused": round(f, 4),
             "bm25": round(bscores.get(cid, 0.0), 4),
             "semantic": round(sscores.get(cid, 0.0), 4),
             "doc": meta_map.get(cid, {}).get("doc", ""),
             "text": doc_map.get(cid, "")[:300]}
            for cid, f in ranked]
