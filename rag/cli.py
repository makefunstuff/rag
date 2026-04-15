"""CLI: rag ingest | sync | search | query | eval."""
import json, sys, time
from argparse import ArgumentParser
from pathlib import Path


def cmd_ingest(args):
    from .core import chunk_file, contextualize, SUPPORTED
    from .db import upsert

    p = Path(args.path)
    if not p.exists():
        print(f"[ingest] not found: {p}"); sys.exit(1)

    files = sorted(f for f in (p.glob(args.glob) if p.is_dir() else [p]) if f.is_file())
    total, ingested = 0, 0
    for f in files:
        if f.suffix.lower() not in SUPPORTED:
            continue
        print(f"[ingest] {f}")
        chunks = chunk_file(f, root=p if p.is_dir() else None)
        if not chunks:
            continue
        ctx = contextualize(str(f), chunks)
        t0 = time.time()
        upsert(chunks, ctx)
        print(f"[ingest]   {len(chunks)} chunks ({time.time()-t0:.1f}s)")
        total += len(chunks)
        ingested += 1
    print(f"\n[ingest] done. {ingested} files, {total} chunks.")


def cmd_sync(args):
    import hashlib
    from .core import chunk_file, contextualize, SUPPORTED, DATA_DIR
    from .db import upsert, delete_by_prefix

    p = Path(args.path)
    if not p.exists():
        print(f"[sync] not found: {p}"); sys.exit(1)

    manifest_path = DATA_DIR / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    new = {}

    files = sorted(f for f in (p.glob(args.glob) if p.is_dir() else [p])
                   if f.is_file() and f.suffix.lower() in SUPPORTED)

    added, updated, unchanged, removed = 0, 0, 0, 0
    for f in files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        key = str(f.relative_to(p) if p.is_dir() else f)
        new[key] = h
        if key in old and old[key] == h:
            unchanged += 1
            continue
        # Changed or new — re-ingest
        chunks = chunk_file(f, root=p if p.is_dir() else None)
        if not chunks:
            continue
        ctx = contextualize(str(f), chunks)
        upsert(chunks, ctx)
        if key in old:
            updated += 1
            print(f"[sync] updated {key} ({len(chunks)} chunks)")
        else:
            added += 1
            print(f"[sync] added {key} ({len(chunks)} chunks)")

    # Delete chunks from removed files
    for key in old:
        if key not in new:
            stem = key.replace("/", "_").replace(" ", "_")
            n = delete_by_prefix(stem)
            removed += 1
            print(f"[sync] removed {key} ({n} chunks)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(new, indent=2))
    print(f"\n[sync] done. +{added} ~{updated} ={unchanged} -{removed}")


def cmd_search(args):
    from .db import hybrid_search

    q = " ".join(args.query)
    results = hybrid_search(q, top_k=args.k)
    if not results:
        print("[search] no chunks — run `rag ingest` first"); sys.exit(1)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n=== top {args.k} for: {q!r} ===\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['doc']}] fused={r['fused']} bm25={r['bm25']} sem={r['semantic']}")
            print(f"   {r['text'][:200]}\n")


def cmd_query(args):
    from .core import llm
    from .db import hybrid_search

    q = " ".join(args.query)
    results = hybrid_search(q, top_k=args.top_k)
    if not results:
        print("No context found. Run `rag ingest` first."); sys.exit(1)

    ctx = "\n\n---\n\n".join(
        f"[{i}] {r['doc']}:\n{r['text']}" for i, r in enumerate(results, 1))
    prompt = f"CONTEXT:\n{ctx}\n\nQUESTION: {q}"
    system = ("Answer using only the provided context. "
              "If the context doesn't contain the answer, say so. Be concise.")

    answer = llm(prompt, system=system)
    if args.json:
        print(json.dumps({"query": q, "answer": answer}))
    else:
        print(f"\n{answer}\n")


def cmd_eval(args):
    from .db import hybrid_search

    pairs = json.loads(Path(args.pairs).read_text()) if args.pairs else [
        {"q": "what is rag?", "a": "retrieval augmented generation"}]
    metrics = {}
    for k in [1, 5, 10, 20]:
        scores = []
        for p in pairs:
            results = hybrid_search(p["q"], top_k=k)
            scores.append(len(results) / max(k, 1))
        metrics[f"recall@{k}"] = round(sum(scores) / len(scores), 3)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(f"\n=== recall ({len(pairs)} pairs) ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f} {'█' * int(v * 20)}")


def main():
    ap = ArgumentParser(prog="rag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="Ingest files into the corpus")
    p.add_argument("path")
    p.add_argument("--glob", default="**/*")

    p = sub.add_parser("sync", help="Sync: only re-ingest changed/new files, remove deleted")
    p.add_argument("path")
    p.add_argument("--glob", default="**/*")

    p = sub.add_parser("search", help="Hybrid search (no LLM)")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("query", help="Search + LLM answer")
    p.add_argument("query", nargs="+")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("eval", help="Evaluate retrieval quality")
    p.add_argument("--pairs", help="JSON file with q/a pairs")
    p.add_argument("--json", action="store_true")

    args = ap.parse_args()
    {"ingest": cmd_ingest, "sync": cmd_sync, "search": cmd_search,
     "query": cmd_query, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
