"""Config, LLM client, chunking, contextual embeddings.

Works with any OpenAI-compatible API: OpenRouter, OpenAI, Ollama, LM Studio,
vLLM, Together, etc. Set API_BASE + API_KEY + LLM_MODEL.

Embeddings are handled by ChromaDB's built-in model (no external API needed).
"""
import hashlib, json, re, os, requests
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────

API_BASE = os.environ.get("API_BASE", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
LLM_MODEL = os.environ.get("LLM_MODEL", "minimax/minimax-m2.7")
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "./data"))

SUPPORTED = {".txt", ".md", ".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java",
             ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".sh", ".yaml", ".yml",
             ".toml", ".json", ".xml", ".html", ".css", ".sql", ".rst", ".adoc",
             ".ipynb"}

_S = requests.Session()
_S.headers["Content-Type"] = "application/json"


# ── LLM ─────────────────────────────────────────────────────────

def llm(prompt: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    h = {}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    r = _S.post(f"{API_BASE}/chat/completions", headers=h, json={
        "model": LLM_MODEL, "messages": msgs, "temperature": 0.3, "max_tokens": 4096,
    }, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── Chunking ────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping paragraph-boundary chunks (~4 chars/token)."""
    if len(text) // 4 <= size:
        return [text.strip()] if text.strip() else []
    paras = re.split(r"\n\n+", text)
    chunks, buf, buf_toks = [], "", 0
    for para in paras:
        para = para.strip()
        if not para:
            continue
        pt = len(para) // 4
        if buf_toks + pt <= size:
            buf += ("\n" if buf else "") + para
            buf_toks += pt
        else:
            if buf.strip():
                chunks.append(buf.strip())
            ov = buf[-overlap * 4:]
            ov = re.sub(r"^[^.!?\n]*[.!?\n]\s*", "", ov)
            buf = (ov + "\n" + para).strip()
            buf_toks = len(buf) // 4
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def read_file(path: Path) -> str:
    """Read file content, extracting cells from .ipynb notebooks."""
    if path.suffix.lower() == ".ipynb":
        nb = json.loads(path.read_text(errors="replace"))
        cells = []
        for cell in nb.get("cells", []):
            src = cell.get("source", [])
            src = "".join(src) if isinstance(src, list) else src
            if src.strip():
                cells.append(src.strip())
        return "\n\n---\n\n".join(cells)
    return path.read_text(errors="replace")


def chunk_file(path: Path, root: Path | None = None) -> list[dict]:
    """Chunk a file into records with unique IDs."""
    chunks = chunk_text(read_file(path))
    rel = path.relative_to(root) if root else path
    stem = str(rel).replace("/", "_").replace(" ", "_")
    return [{"id": f"{stem}_{i}", "doc": path.name, "path": str(path),
             "index": i, "text": c} for i, c in enumerate(chunks)]


# ── Contextual embeddings ───────────────────────────────────────

def contextualize(doc_path: str, chunks: list[dict]) -> list[str]:
    """Prepend LLM-generated context to each chunk (batch, cached per doc)."""
    cache_dir = DATA_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(doc_path.encode()).hexdigest()[:16]
    cache = cache_dir / f"ctx_{key}.json"
    texts = [c["text"] for c in chunks]
    if not texts:
        return []
    if cache.exists():
        descs = json.loads(cache.read_text())["descs"]
    else:
        listing = "\n".join(
            f"[{i}] {t[:400]}{'...' if len(t) > 400 else ''}" for i, t in enumerate(texts))
        resp = llm(
            f"For each chunk below, write a 1-2 sentence description of what it "
            f"contains and where it fits in the document.\n\nChunks:\n{listing}\n\n"
            f"Reply with exactly one JSON array of description strings.",
            system="Reply with only valid JSON: a JSON array of strings.")
        try:
            descs = json.loads(resp)
        except Exception:
            descs = []
        while len(descs) < len(texts):
            descs.append("")
        descs = descs[:len(texts)]
        cache.write_text(json.dumps({"descs": descs}))
    return [f"{d}\n\n{t}" if d else t for d, t in zip(descs, texts)]
