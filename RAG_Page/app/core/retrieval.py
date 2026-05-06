"""
retrieval.py — Hybrid retrieval (dense + BM25 + RRF) with multi-query decomposition and reranking.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from pathlib import Path

from app.config import cfg
from app.logger import get_logger

log = get_logger(__name__)

_BM25_CACHE: dict | None = None


# ── BM25 index ────────────────────────────────────────────────────────────────

def _get_stopwords() -> set:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return set()


_STOPWORDS = _get_stopwords()


def clean_query(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _build_bm25_index(col) -> dict:
    from rank_bm25 import BM25Okapi
    log.info("Building BM25 index …")
    total  = col.count()
    result = col.get(limit=total, include=["documents", "metadatas"])
    ids    = result["ids"]
    docs   = result["documents"]
    metas  = result["metadatas"]
    tokenized = [re.findall(r"[a-zA-Z0-9]+", d.lower()) for d in docs]
    bm25      = BM25Okapi(tokenized)
    log.info("BM25 index built: %d docs.", len(ids))
    return {"bm25": bm25, "ids": ids, "docs": docs, "metas": metas}


def get_bm25_index(col) -> dict:
    global _BM25_CACHE
    if _BM25_CACHE is None:
        _BM25_CACHE = _build_bm25_index(col)
    return _BM25_CACHE


def invalidate_bm25_cache() -> None:
    global _BM25_CACHE
    _BM25_CACHE = None
    log.info("BM25 cache invalidated.")


# ── Hit builder ───────────────────────────────────────────────────────────────

def _parse_hit(doc: str, meta: dict, dense_score: float = 0.0, bm25_score: float = 0.0) -> dict:
    provenance = []
    try:
        provenance = json.loads(meta.get("provenance", "[]"))
    except (json.JSONDecodeError, TypeError):
        pass
    return {
        "score":             0.0,
        "dense_score":       dense_score,
        "bm25_score":        bm25_score,
        "reranker_score":    0.0,
        "document_name":     meta.get("document_name", ""),
        "source":            meta.get("source", ""),
        "chunk_index":       meta.get("chunk_index", -1),
        "chunk_type":        meta.get("chunk_type", ""),
        "chunker":           meta.get("chunker", ""),
        "headings":          meta.get("headings", ""),
        "page_numbers":      meta.get("page_numbers", ""),
        "text":              doc,
        "raw_text":          meta.get("raw_text", ""),
        "image_ref":         meta.get("image_ref", ""),
        "caption":           meta.get("caption", ""),
        "parent_heading":    meta.get("parent_heading", ""),
        "provenance_parsed": provenance,
    }


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(
    dense_ranked: list[str],
    bm25_ranked:  list[str],
    id_to_data:   dict,
    top_k:        int | None = None,
) -> list[dict]:
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (cfg.rrf_k + rank)
    for rank, cid in enumerate(bm25_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (cfg.rrf_k + rank)

    fused_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    if top_k is not None:
        fused_ids = fused_ids[:top_k]

    hits = []
    for cid in fused_ids:
        entry = id_to_data[cid].copy()
        entry["score"]       = round(scores[cid], 6)
        entry["dense_score"] = round(id_to_data[cid].get("dense_score", 0.0), 4)
        entry["bm25_score"]  = round(id_to_data[cid].get("bm25_score",  0.0), 4)
        hits.append(entry)
    return hits


# ── Single-query hybrid retrieval ─────────────────────────────────────────────

def _retrieve_single(
    query:       str,
    col,
    top_k:       int | None,
    filter_docs: list[str] | None,
) -> list[dict]:
    from app.services.embedding_service import embed_query
    from app.services.chromadb_service import query_dense

    count   = col.count()
    if filter_docs and len(filter_docs) == 1:
        where = {"document_name": filter_docs[0]}
    elif filter_docs:
        where = {"document_name": {"$in": filter_docs}}
    else:
        where = None
    n_fetch = min(cfg.hybrid_candidates, count)
    id_to_data: dict[str, dict] = {}

    # Dense
    log.debug("Dense retrieval: n_fetch=%d …", n_fetch)
    vec    = embed_query(query)
    result = query_dense(vec, n_fetch, where)
    dense_ranked: list[str] = []
    for cid, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        dense_score = round(1 - dist, 4)
        dense_ranked.append(cid)
        id_to_data[cid] = _parse_hit(doc, meta, dense_score=dense_score)

    # BM25
    log.debug("BM25 retrieval …")
    bm25_tokens = clean_query(query)
    idx_data    = get_bm25_index(col)
    bm25_obj    = idx_data["bm25"]
    corpus_ids  = idx_data["ids"]
    corpus_docs = idx_data["docs"]
    corpus_metas= idx_data["metas"]

    raw_scores = bm25_obj.get_scores(bm25_tokens)
    if filter_docs:
        filter_set = set(filter_docs)
        raw_scores = [
            s if corpus_metas[i].get("document_name") in filter_set else 0.0
            for i, s in enumerate(raw_scores)
        ]

    top_idx = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:n_fetch]
    bm25_ranked: list[str] = []
    max_bm25 = raw_scores[top_idx[0]] if top_idx else 1.0
    for idx in top_idx:
        cid        = corpus_ids[idx]
        bm25_score = round(raw_scores[idx] / max_bm25, 4) if max_bm25 > 0 else 0.0
        bm25_ranked.append(cid)
        if cid not in id_to_data:
            id_to_data[cid] = _parse_hit(corpus_docs[idx], corpus_metas[idx], bm25_score=bm25_score)
        else:
            id_to_data[cid]["bm25_score"] = bm25_score

    fused = _rrf_fuse(dense_ranked, bm25_ranked, id_to_data, top_k=top_k)
    log.debug("RRF fusion: %d candidates.", len(fused))
    return fused


# ── Multi-query retrieval ─────────────────────────────────────────────────────

def _multi_query_retrieve(
    sub_queries:  list[str],
    col,
    filter_docs:  list[str] | None,
) -> list[dict]:
    seen: dict[str, dict] = {}

    def _fetch(sq: str) -> list[dict]:
        log.info("  Sub-query retrieval: %r", sq)
        return _retrieve_single(sq, col, top_k=cfg.multi_query_chunks, filter_docs=filter_docs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
        futures = {pool.submit(_fetch, sq): sq for sq in sub_queries}
        for future in concurrent.futures.as_completed(futures):
            sq = futures[future]
            try:
                hits = future.result()
            except Exception as e:
                log.warning("Sub-query failed (%r): %s", sq, e)
                continue
            for hit in hits:
                key = f"{hit['document_name']}::{hit['chunk_index']}"
                if key not in seen or hit["score"] > seen[key]["score"]:
                    seen[key] = hit

    log.info("Multi-query: %d unique candidates merged.", len(seen))
    return list(seen.values())


# ── File enrichment ───────────────────────────────────────────────────────────

def enrich_with_files(hit: dict) -> dict:
    doc_name   = hit.get("document_name", "")
    provenance = hit.get("provenance_parsed", [])
    base       = cfg.docs_dir / doc_name

    def exists(p: Path) -> str | None:
        return str(p) if p.exists() else None

    pages = sorted({int(p["page_no"]) for p in provenance if p.get("page_no") is not None})
    if not pages:
        raw = hit.get("page_numbers", "")
        pages = [int(x) for x in raw.split(",") if x.strip().isdigit()]

    page_images, page_mds, layout_jsons = [], [], []
    for pn in pages:
        page_images .append(exists(base / "pages"    / f"page_{pn}.png"))
        page_mds    .append(exists(base / "pages_md" / f"page_{pn}.md"))
        layout_jsons.append(exists(base / "layout"   / f"page_{pn}_layout.json"))

    image_ref      = hit.get("image_ref") or ""
    image_ref_path = exists(base / "images" / image_ref) if image_ref else None

    hit["files"] = {
        "page_images":    [p for p in page_images    if p],
        "page_markdown":  [p for p in page_mds       if p],
        "layout_json":    [p for p in layout_jsons   if p],
        "image_ref_path": image_ref_path,
        "all_pages_json": exists(base / "layout" / "all_pages.json"),
        "pdf_md":         exists(base / f"{doc_name}.md"),
    }
    return hit


# ── Main retrieval entry point ────────────────────────────────────────────────

def retrieve(
    query:       str,
    top_k:       int | None = None,
    filter_docs: list[str] | None = None,
    progress_cb=None,
) -> tuple[list[dict], dict]:
    """
    Full retrieval pipeline:
      1. Query decomposition (Ollama LLM)
      2. Hybrid retrieval (dense + BM25 + RRF)
      3. CrossEncoder reranking
      4. File enrichment

    Returns (hits, query_meta).
    """
    from app.services.llm_service import decompose_query
    from app.services.reranker_service import rerank
    from app.services.chromadb_service import get_collection

    tk = top_k or cfg.default_top_k

    def _p(msg: str):
        log.info("[RETRIEVE] %s", msg)
        if progress_cb:
            progress_cb(msg)

    col = get_collection()
    if col.count() == 0:
        _p("Collection is empty. Please index documents first.")
        return [], {}

    # Step 1: Decompose
    _p(f"Analysing query complexity …")
    is_complex, sub_queries = decompose_query(query)

    if is_complex and sub_queries:
        _p(f"Complex query → {len(sub_queries)} sub-queries: {sub_queries}")
        all_candidates = _multi_query_retrieve(sub_queries, col, filter_docs)
        origin_label   = "multi-query"
    else:
        _p("Simple query — single hybrid retrieval.")
        all_candidates = _retrieve_single(query, col, top_k=None, filter_docs=filter_docs)
        origin_label   = "single"

    _p(f"{len(all_candidates)} candidate(s) → reranking …")

    # Step 2: Rerank
    ranked = rerank(query, all_candidates)
    hits   = ranked[:tk]

    # Step 3: File enrichment
    for hit in hits:
        enrich_with_files(hit)

    query_meta = {
        "original_query":           query,
        "is_complex":               is_complex,
        "retrieval_mode":           origin_label,
        "sub_queries":              sub_queries if is_complex else [],
        "candidates_before_rerank": len(all_candidates),
        "top_k":                    tk,
        "llm_model":                cfg.llm_model,
    }
    _p(f"Retrieval complete: {len(hits)} results returned.")
    return hits, query_meta
