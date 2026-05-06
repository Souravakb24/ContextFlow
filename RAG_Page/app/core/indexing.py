"""
indexing.py — Load chunks from disk, embed, and upsert to ChromaDB.
Skips chunks already indexed (by chunk_id presence in collection).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from app.config import cfg
from app.logger import get_logger
from app.utils.file_utils import chunks_path_for
from app.utils.gpu_utils import log_gpu_memory

log = get_logger(__name__)


# ── Provenance extraction ─────────────────────────────────────────────────────

def _extract_provenance(chunk: dict) -> list[dict]:
    if chunk.get("chunker") == "figure":
        bbox = chunk.get("bbox")
        page = chunk.get("page")
        if bbox and page is not None:
            return [{"page_no": page, "x0": bbox.get("x0"), "y0": bbox.get("y0"),
                     "x1": bbox.get("x1"), "y1": bbox.get("y1")}]
        return []
    provenance: list[dict] = []
    for item in chunk.get("meta", {}).get("doc_items", []):
        for prov in item.get("prov", []):
            bbox = prov.get("bbox")
            page = prov.get("page_no")
            if bbox and page is not None:
                provenance.append({
                    "page_no": page,
                    "x0": bbox.get("x0"), "y0": bbox.get("y0"),
                    "x1": bbox.get("x1"), "y1": bbox.get("y1"),
                })
    return provenance


def _unique_pages(provenance: list[dict]) -> str:
    pages = sorted({p["page_no"] for p in provenance if p.get("page_no") is not None})
    return ",".join(str(p) for p in pages)


def _build_metadata(file_meta: dict, chunk: dict, provenance: list[dict]) -> dict:
    chunk_type   = chunk.get("type") or chunk.get("chunker", "text")
    headings_raw = (chunk.get("meta") or {}).get("headings") or []
    headings     = " > ".join(headings_raw) if isinstance(headings_raw, list) else str(headings_raw)
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
        meta["raw_text"] = str(raw)
    if chunk_type == "figure":
        meta["caption"]        = chunk.get("caption", "")
        meta["parent_heading"] = chunk.get("parent_heading", "")
        meta["image_ref"]      = chunk.get("image_ref", "")
    return meta


# ── Index one chunks file ─────────────────────────────────────────────────────

def index_document(
    chunks_path: Path,
    progress_cb=None,
) -> int:
    """
    Embed and upsert all chunks from a chunks JSON file.
    Returns number of chunks indexed.
    Skips chunks whose IDs already exist in ChromaDB.
    """
    from app.services import embedding_service, chromadb_service

    def _p(msg: str):
        log.info("[INDEX] %s", msg)
        if progress_cb:
            progress_cb(msg)

    if not chunks_path.exists():
        _p(f"Chunks file not found: {chunks_path}")
        return 0

    data     = json.loads(chunks_path.read_text(encoding="utf-8"))
    doc_name = data.get("document_name", chunks_path.parent.name)
    chunks   = data.get("chunks", [])

    if not chunks:
        _p(f"No chunks in {chunks_path} — skipping.")
        return 0

    _p(f"Processing {len(chunks)} chunks for '{doc_name}' …")

    # Build lists
    ids:       list[str]  = []
    documents: list[str]  = []
    metadatas: list[dict] = []

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        chunk_id   = f"{doc_name}_chunk_{chunk['chunk_index']}"
        provenance = _extract_provenance(chunk)
        metadata   = _build_metadata(data, chunk, provenance)
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata)

    if not ids:
        _p(f"All chunks empty for '{doc_name}' — skipping.")
        return 0

    # Check which IDs already exist
    col        = chromadb_service.get_collection()
    existing   = set(col.get(ids=ids)["ids"])
    new_ids    = [i for i in ids if i not in existing]
    new_docs   = [documents[ids.index(i)] for i in new_ids]
    new_metas  = [metadatas[ids.index(i)] for i in new_ids]

    if not new_ids:
        _p(f"'{doc_name}' already fully indexed ({len(ids)} chunks). Skipping.")
        return 0

    _p(f"Embedding {len(new_ids)} new chunks for '{doc_name}' (skipping {len(existing)} already indexed) …")
    t0 = time.perf_counter()

    embeddings = embedding_service.embed_texts(new_docs)
    elapsed    = time.perf_counter() - t0
    _p(f"Encoded {len(new_ids)} chunks in {elapsed:.1f}s ({len(new_ids)/elapsed:.0f} chunks/s).")

    chromadb_service.upsert_chunks(
        ids=new_ids,
        documents=new_docs,
        embeddings=embeddings.tolist(),
        metadatas=new_metas,
    )
    log_gpu_memory(cfg.embed_device, "post-index")
    _p(f"Indexed {len(new_ids)} chunks for '{doc_name}'.")
    return len(new_ids)


def index_all_documents(progress_cb=None) -> int:
    """
    Find all hybrid_chunks.json files under docs_dir and index them.
    Returns total chunks indexed.
    """
    chunk_files = sorted(cfg.docs_dir.rglob("hybrid_chunks.json"))
    if not chunk_files:
        log.warning("No chunk files found under %s", cfg.docs_dir)
        return 0

    log.info("Found %d chunk file(s) to index.", len(chunk_files))
    total = 0
    for f in tqdm(chunk_files, desc="Indexing documents", unit="doc"):
        total += index_document(f, progress_cb=progress_cb)
    log.info("Total chunks indexed this run: %d", total)
    return total


def index_single_pdf(pdf_name: str, progress_cb=None) -> int:
    """Index a single document by name."""
    cpath = chunks_path_for(cfg.docs_dir, pdf_name, cfg.chunker_type)
    return index_document(cpath, progress_cb=progress_cb)
