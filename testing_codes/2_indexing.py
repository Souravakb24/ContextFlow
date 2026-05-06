#!/usr/bin/env python3
"""
Index hybrid_chunks.json files into ChromaDB.

Each chunk → one ChromaDB document:
  - id         : "{document_name}_chunk_{chunk_index}"
  - document   : chunk["text"]  (what gets embedded & searched)
  - embedding  : bge-large-en-v1.5 encoding of chunk["text"]
  - metadata   : flat dict including provenance as JSON string for UI bbox highlighting

Usage:
    python3 indexing.py                          # index all docs under Documents-OCR/
    python3 indexing.py --docs-dir /path/to/dir
    python3 indexing.py --db-path /path/to/chroma_db
    python3 indexing.py --device cuda:1          # use a specific GPU
    python3 indexing.py --fp16                   # half-precision (faster on modern GPUs)
    python3 indexing.py --batch-size 512         # override auto batch size
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DOCS_DIR = Path(__file__).parent / "Documents-OCR"
DEFAULT_DB_PATH  = Path(__file__).parent / "vector/chroma_db"
COLLECTION_NAME  = "rag_chunks"
EMBED_MODEL      = "BAAI/bge-large-en-v1.5"

# Batch sizes tuned per backend; --batch-size overrides these
BATCH_SIZE_CUDA = 512
BATCH_SIZE_CPU  = 64

# GPU VRAM thresholds (bytes) → auto batch size scaling
_VRAM_BATCH: list[tuple[int, int]] = [
    (24 * 1024**3, 512),   # ≥ 24 GB  (A100 / 4090)
    (16 * 1024**3, 384),   # ≥ 16 GB  (A100-SXM4-40G, 3090)
    (10 * 1024**3, 256),   # ≥ 10 GB  (3080 / T4)
    (6  * 1024**3, 128),   # ≥  6 GB  (2080 Ti / P40)
    (0,             64),   # <  6 GB  (low VRAM fallback)
]


# ── Device helpers ────────────────────────────────────────────────────────────

def resolve_device(requested: str | None) -> str:
    """
    Return the device string to use. Requires a GPU (CUDA or MPS); exits if none found.

    Priority:
      1. Explicit --device argument
      2. First available CUDA GPU
      3. MPS (Apple Silicon)
    """
    if requested:
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                sys.exit("[error] CUDA requested but no CUDA GPU is available. Aborting.")
            idx = int(requested.split(":")[-1]) if ":" in requested else 0
            if idx >= torch.cuda.device_count():
                sys.exit(
                    f"[error] cuda:{idx} requested but only "
                    f"{torch.cuda.device_count()} GPU(s) found. Aborting."
                )
        elif requested == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                sys.exit("[error] MPS requested but not available. Aborting.")
        elif requested == "cpu":
            sys.exit("[error] CPU execution is disabled. Use a CUDA or MPS device.")
        return requested

    if torch.cuda.is_available():
        return "cuda:0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    sys.exit("[error] No GPU found (CUDA or MPS required). Aborting.")


def auto_batch_size(device: str) -> int:
    """Return a sensible batch size based on available VRAM."""
    if not device.startswith("cuda"):
        return BATCH_SIZE_CUDA

    dev_idx = int(device.split(":")[-1]) if ":" in device else 0
    try:
        total_vram = torch.cuda.get_device_properties(dev_idx).total_memory
        for vram_threshold, bs in _VRAM_BATCH:
            if total_vram >= vram_threshold:
                return bs
    except Exception:
        pass

    return BATCH_SIZE_CUDA


def print_device_info(device: str) -> None:
    """Pretty-print device information at startup."""
    if device.startswith("cuda"):
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        props   = torch.cuda.get_device_properties(dev_idx)
        vram_gb = props.total_memory / 1024**3
        free_gb = (
            torch.cuda.mem_get_info(dev_idx)[0] / 1024**3
            if hasattr(torch.cuda, "mem_get_info")
            else float("nan")
        )
        print(
            f"Device : {device} — {props.name}  "
            f"({vram_gb:.1f} GB total, {free_gb:.1f} GB free)"
        )
        print(f"CUDA   : {torch.version.cuda}   cuDNN: {torch.backends.cudnn.version()}")
    elif device == "mps":
        print("Device : MPS (Apple Silicon GPU)")


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(device: str, fp16: bool) -> SentenceTransformer:
    """Load and configure SentenceTransformer for the target device."""
    print(f"\nLoading embedding model: {EMBED_MODEL}")

    model = SentenceTransformer(EMBED_MODEL, device=device)

    if fp16:
        model = model.half()
        print("Precision : fp16 (half precision)")
    else:
        print("Precision : fp32")

    # Warm up the model so the first real batch isn't slow
    if device.startswith("cuda"):
        _ = model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        torch.cuda.synchronize()
        print("Warm-up   : done")

    return model


# ── Provenance extraction ─────────────────────────────────────────────────────

def _extract_provenance(chunk: dict) -> list[dict]:
    """Return list of {page_no, x0, y0, x1, y1} for every bbox in the chunk."""
    provenance: list[dict] = []

    if chunk.get("chunker") == "figure":
        bbox = chunk.get("bbox")
        page = chunk.get("page")
        if bbox and page is not None:
            provenance.append({
                "page_no": page,
                "x0": bbox.get("x0"),
                "y0": bbox.get("y0"),
                "x1": bbox.get("x1"),
                "y1": bbox.get("y1"),
            })
        return provenance

    for item in chunk.get("meta", {}).get("doc_items", []):
        for prov in item.get("prov", []):
            bbox = prov.get("bbox")
            page = prov.get("page_no")
            if bbox and page is not None:
                provenance.append({
                    "page_no": page,
                    "x0": bbox.get("x0"),
                    "y0": bbox.get("y0"),
                    "x1": bbox.get("x1"),
                    "y1": bbox.get("y1"),
                })

    return provenance


def _unique_pages(provenance: list[dict]) -> str:
    pages = sorted({p["page_no"] for p in provenance if p.get("page_no") is not None})
    return ",".join(str(p) for p in pages)


# ── Metadata builder ──────────────────────────────────────────────────────────

def build_metadata(file_meta: dict, chunk: dict, provenance: list[dict]) -> dict:
    chunk_type = chunk.get("type") or chunk.get("chunker", "text")

    headings_raw = (chunk.get("meta") or {}).get("headings") or []
    headings = " > ".join(headings_raw) if isinstance(headings_raw, list) else str(headings_raw)

    meta = {
        "source":        file_meta.get("source", ""),
        "document_name": file_meta.get("document_name", ""),
        "chunk_index":   chunk.get("chunk_index", -1),
        "chunk_type":    chunk_type,
        "chunker":       chunk.get("chunker", ""),
        "headings":      headings,
        "page_numbers":  _unique_pages(provenance),
        "provenance":    json.dumps(provenance),
    }

    raw = chunk.get("raw_text")
    if raw:
        meta["raw_text"] = raw

    if chunk_type == "figure":
        meta["caption"]        = chunk.get("caption", "")
        meta["parent_heading"] = chunk.get("parent_heading", "")
        meta["image_ref"]      = chunk.get("image_ref", "")

    return meta


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_file(
    chunks_path: Path,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    batch_size: int,
    device: str,
) -> int:
    data     = json.loads(chunks_path.read_text(encoding="utf-8"))
    doc_name = data.get("document_name", chunks_path.parent.name)
    chunks   = data.get("chunks", [])

    if not chunks:
        print(f"  [skip] {chunks_path} — no chunks")
        return 0

    ids:       list[str]  = []
    documents: list[str]  = []
    metadatas: list[dict] = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        chunk_id   = f"{doc_name}_chunk_{chunk['chunk_index']}"
        provenance = _extract_provenance(chunk)
        metadata   = build_metadata(data, chunk, provenance)

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata)

    if not ids:
        print(f"  [skip] {doc_name} — all chunks empty")
        return 0

    print(f"  Embedding {len(ids)} chunks from '{doc_name}' …")
    t0 = time.perf_counter()

    encode_kwargs: dict = dict(
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity ≡ dot product after L2 norm
    )

    # Pin memory for faster CPU→GPU transfers on CUDA
    if device.startswith("cuda"):
        encode_kwargs["device"] = device

    embeddings = model.encode(documents, **encode_kwargs)

    # Sync CUDA so elapsed time is accurate
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    print(f"  Encoded  {len(ids)} chunks in {elapsed:.1f}s  ({len(ids)/elapsed:.0f} chunks/s)")

    # Upsert in batches to avoid ChromaDB per-call limits
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    print(f"  ✓  {len(ids)} chunks indexed for '{doc_name}'")
    return len(ids)


def index_all(docs_dir: Path, db_path: Path, device: str, batch_size: int, fp16: bool) -> None:
    chunk_files = sorted(docs_dir.rglob("hybrid_chunks.json"))
    if not chunk_files:
        print(f"No hybrid_chunks.json files found under {docs_dir}")
        return

    print(f"Found {len(chunk_files)} chunk file(s):")
    for f in chunk_files:
        print(f"  {f}")

    print_device_info(device)
    model = load_model(device, fp16)

    print(f"\nConnecting to ChromaDB at: {db_path}")
    client     = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{COLLECTION_NAME}' — {collection.count()} existing docs\n")
    print(f"Batch size : {batch_size}\n")

    total    = 0
    t_global = time.perf_counter()

    for f in chunk_files:
        total += index_file(f, collection, model, batch_size, device)

        if device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved  = torch.cuda.memory_reserved()  / 1024**2
            print(f"  GPU mem  : {allocated:.0f} MB allocated / {reserved:.0f} MB reserved")

    elapsed_total = time.perf_counter() - t_global
    print(f"\nDone in {elapsed_total:.1f}s.  Total chunks indexed: {total}")
    print(f"Collection now has {collection.count()} documents.")

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        print("GPU cache cleared.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index PDF chunks into ChromaDB with flexible GPU/CPU support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--docs-dir", type=Path, default=DEFAULT_DOCS_DIR,
        help="Root directory containing hybrid_chunks.json files",
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH,
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help=(
            "Compute device: 'cuda', 'cuda:0', 'cuda:1', 'mps'. "
            "Auto-detected when omitted. CPU is not allowed."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Embedding batch size. Auto-tuned from VRAM when omitted.",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use half-precision (fp16) on CUDA/MPS for faster embedding.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = resolve_device(args.device)
    bs     = args.batch_size if args.batch_size else auto_batch_size(device)

    index_all(
        docs_dir   = args.docs_dir,
        db_path    = args.db_path,
        device     = device,
        batch_size = bs,
        fp16       = args.fp16,
    )