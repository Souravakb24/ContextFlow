"""
retrieval.py — Query ChromaDB and return enriched results
==========================================================

Matches the indexing schema from indexing.py:
  - Single collection  : "rag_chunks"
  - Metadata fields    : document_name, chunk_type, chunker, headings,
                         page_numbers, provenance (JSON string), raw_text,
                         source, image_ref, caption, parent_heading
  - Embedding model    : BAAI/bge-large-en-v1.5  (SentenceTransformer)
  - bbox format        : {x0, y0, x1, y1}  inside provenance list

Multi-query mode (complex queries):
  - Ollama gpt-oss:20b detects if a query is complex
  - Complex queries are split into simple sub-queries
  - Sub-queries are processed in parallel (5 chunks each)
  - All chunks are merged and reranked

Usage:
    python retrieval.py --query "What is KV cache?"
    python retrieval.py --query "SigLIP encoder" --top-k 3
    python retrieval.py --db /path/to/chroma_db --save-dir ./results
    python retrieval.py   # interactive mode
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import math
import torch
import chromadb
import ollama
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from sentence_transformers import CrossEncoder, SentenceTransformer

# ── Device ─────────────────────────────────────────────────────────────────────

def resolve_device(requested: str | None) -> str:
    """Require a GPU (CUDA or MPS); exit if none found."""
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


# Models are initialised in main() once the device is known
_MODEL:    SentenceTransformer | None = None
_RERANKER: CrossEncoder        | None = None

OLLAMA_MODEL = "gpt-oss:20b"


def _init_models(device: str) -> None:
    global _MODEL, _RERANKER

    if device.startswith("cuda"):
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        props   = torch.cuda.get_device_properties(dev_idx)
        print(f"Device : {device} — {props.name}  ({props.total_memory/1024**3:.1f} GB)")

    print("Loading BAAI/bge-large-en-v1.5 …")
    _MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    print("✓ Embedding model ready")

    print("Loading BAAI/bge-reranker-v2-m3 …")
    _RERANKER = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        max_length=512,
        device=device,
    )
    print("✓ Reranker ready\n")

    # Verify Ollama is reachable and model exists
    print(f"Checking Ollama model {OLLAMA_MODEL} …")
    try:
        ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            options={"num_predict": 1},
        )
        print(f"✓ Ollama {OLLAMA_MODEL} ready\n")
    except Exception as e:
        print(f"[warning] Ollama/{OLLAMA_MODEL} unavailable: {e}. Multi-query decomposition disabled.\n")


# ── Stopwords ──────────────────────────────────────────────────────────────────
_STOPWORDS = set(stopwords.words("english"))

HYBRID_CANDIDATES      = 10   # top-N fetched from each retriever before fusion
RRF_K                  = 60   # RRF constant
MULTI_QUERY_CHUNKS     = 5    # chunks retrieved per sub-query in multi-query mode


def embed_query(text: str) -> list[float]:
    return _MODEL.encode([text])[0].tolist()


def clean_query(text: str) -> list[str]:
    """Lowercase, remove stopwords, tokenize — for BM25."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


# ── BM25 index (built once per collection, cached) ────────────────────────────
_BM25_CACHE: dict | None = None   # {bm25, ids, docs, metas}


def _build_bm25_index(col: chromadb.Collection) -> dict:
    """Load every document text from ChromaDB and build a BM25Okapi index."""
    print("  Building BM25 index …", end=" ", flush=True)
    total  = col.count()
    result = col.get(limit=total, include=["documents", "metadatas"])

    ids    = result["ids"]
    docs   = result["documents"]
    metas  = result["metadatas"]

    tokenized = [re.findall(r"[a-zA-Z0-9]+", d.lower()) for d in docs]
    bm25      = BM25Okapi(tokenized)
    print(f"done ({len(ids)} docs)")
    return {"bm25": bm25, "ids": ids, "docs": docs, "metas": metas}


def _get_bm25_index(col: chromadb.Collection) -> dict:
    global _BM25_CACHE
    if _BM25_CACHE is None:
        _BM25_CACHE = _build_bm25_index(col)
    return _BM25_CACHE


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_fuse(
    dense_ranked:  list[str],
    bm25_ranked:   list[str],
    id_to_data:    dict,
    top_k:         int | None = None,
) -> list[dict]:
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, cid in enumerate(bm25_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

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


# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH   = "/storage/sourava/RAG_Pipeline/FInal/vectorDB/all_chroma_db"
DEFAULT_DOCS_ROOT = Path(__file__).parent / "Documents-OCR"
DEFAULT_SAVE_DIR  = Path(__file__).parent / "results"
COLLECTION_NAME   = "rag_chunks"


# ══════════════════════════════════════════════════════════════════════════════
# Ollama — Query complexity detection and decomposition
# ══════════════════════════════════════════════════════════════════════════════

_DECOMPOSE_SYSTEM = """\
You are a query decomposition assistant for a retrieval-augmented generation (RAG) system.

Your job: decide if a query is COMPLEX or SIMPLE, then split complex ones.

COMPLEX means the query contains TWO OR MORE of these signals:
  • Multiple distinct questions joined by "and", "also", "as well as", "additionally"
  • A comparison between two or more things (e.g. "compared to", "vs", "difference between")
  • Questions about causes AND effects, OR mechanisms AND trade-offs
  • Asks about multiple separate topics that would live in different document sections
  • Contains "Also explain ...", "Additionally ...", "What trade-offs ..."
  • Long sentence (>30 words) covering more than one conceptual angle

SIMPLE means it asks one focused thing — even if the sentence is long.

Examples of SIMPLE queries:
  "What is KV cache?"
  "How does attention work in transformers?"

Rules for sub-queries:
  - 2 to 4 sub-queries maximum
  - Each must be a complete, self-contained question
  - Cover every distinct aspect of the original query
  - Do NOT overlap — each sub-query should target a different aspect

Respond ONLY with valid JSON, no markdown, no extra text:
  {"complex": true,  "sub_queries": ["...", "..."]}
  {"complex": false, "sub_queries": []}
"""

# Heuristic signals that strongly suggest a complex query even if the LLM disagrees
_COMPLEXITY_PATTERNS = re.compile(
    r"\b(also explain|additionally|as well as|compared to|vs\b|versus|what trade.offs|"
    r"how does .{5,60} affect|difference between|and what|and how)\b",
    re.IGNORECASE,
)


def _heuristic_is_complex(query: str) -> bool:
    """Light rule-based check as a safety net for obvious multi-part queries."""
    if len(query.split()) > 35 and _COMPLEXITY_PATTERNS.search(query):
        return True
    # Count strong conjunctive signals
    signals = len(_COMPLEXITY_PATTERNS.findall(query))
    return signals >= 2


def decompose_query(query: str) -> tuple[bool, list[str]]:
    """
    Use Ollama gpt-oss:20b to decide if a query is complex and split it.
    A heuristic pre-check overrides the LLM when the query is obviously multi-part.
    Returns (is_complex, sub_queries). Falls back to (False, []) on LLM error.
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user",   "content": f"Query: {query}"},
            ],
            options={"temperature": 0.0, "num_predict": 512},
        )
        raw = response["message"]["content"].strip()
        # Strip markdown code fences if the model adds them
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        data = json.loads(raw)
        is_complex  = bool(data.get("complex", False))
        sub_queries = [s.strip() for s in data.get("sub_queries", []) if s.strip()]

        # Override: if heuristic says complex but LLM said simple, ask again with more force
        if not is_complex and _heuristic_is_complex(query):
            print("  [heuristic] Query looks complex — retrying decomposition with explicit hint …")
            retry_msg = (
                f"This query contains multiple distinct questions and comparisons. "
                f"It MUST be classified as complex. Split it into focused sub-queries.\n\nQuery: {query}"
            )
            retry_resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": _DECOMPOSE_SYSTEM},
                    {"role": "user",   "content": retry_msg},
                ],
                options={"temperature": 0.0, "num_predict": 512},
            )
            raw2 = retry_resp["message"]["content"].strip()
            raw2 = re.sub(r"^```[a-z]*\n?", "", raw2).rstrip("`").strip()
            data2 = json.loads(raw2)
            is_complex  = bool(data2.get("complex", False))
            sub_queries = [s.strip() for s in data2.get("sub_queries", []) if s.strip()]

            # Final fallback: heuristic wins if LLM still refuses
            if not is_complex and _heuristic_is_complex(query):
                print("  [heuristic] LLM still returned simple — forcing complex via heuristic split.")
                is_complex  = True
                sub_queries = _heuristic_split(query)

        return is_complex, sub_queries
    except Exception as e:
        print(f"  [warning] Query decomposition failed ({e}). Checking heuristic …")
        if _heuristic_is_complex(query):
            return True, _heuristic_split(query)
        return False, []


def _heuristic_split(query: str) -> list[str]:
    """
    Naive fallback splitter: break on 'and', 'also', 'additionally', '?'
    to produce rough sub-queries when the LLM is unavailable or uncooperative.
    """
    parts = re.split(r"\?\s+|\band\b|\balso\b|\badditionally\b", query, flags=re.IGNORECASE)
    cleaned = []
    for p in parts:
        p = p.strip().strip(".").strip()
        if len(p.split()) >= 4:          # skip fragments shorter than 4 words
            if not p.endswith("?"):
                p += "?"
            cleaned.append(p)
    return cleaned[:4] or [query]


# ══════════════════════════════════════════════════════════════════════════════
# File enrichment
# ══════════════════════════════════════════════════════════════════════════════

def enrich_with_files(hit: dict, docs_root: Path) -> dict:
    """
    Attach real file paths to a hit so the UI knows where to find
    page images, markdown, layout JSON, and figure images.

    docs_root/
      <document_name>/
        pages/       page_N.png
        pages_md/    page_N.md
        layout/      page_N_layout.json   all_pages.json
        images/      page_N_pic_K.png
    """
    doc_name   = hit.get("document_name", "")
    provenance = hit.get("provenance_parsed", [])
    base       = docs_root / doc_name

    def exists(p: Path) -> str | None:
        return str(p) if p.exists() else None

    pages = sorted({int(p["page_no"]) for p in provenance if p.get("page_no") is not None})
    if not pages:
        raw_pages = hit.get("page_numbers", "")
        pages = [int(x) for x in raw_pages.split(",") if x.strip().isdigit()]

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


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def _parse_hit(doc: str, meta: dict, dense_score: float = 0.0, bm25_score: float = 0.0) -> dict:
    """Build a hit dict from a ChromaDB document + metadata."""
    provenance = []
    try:
        provenance = json.loads(meta.get("provenance", "[]"))
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "score":             0.0,
        "dense_score":       dense_score,
        "bm25_score":        bm25_score,
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


def _retrieve_single(
    query:      str,
    col:        chromadb.Collection,
    top_k:      int,
    filter_doc: str | None,
) -> list[dict]:
    """
    Hybrid retrieval (dense + BM25 + RRF) for one query.
    Returns top_k hits WITHOUT reranking (reranking happens upstream).
    """
    count   = col.count()
    where   = {"document_name": filter_doc} if filter_doc else None
    n_fetch = min(HYBRID_CANDIDATES, count)
    id_to_data: dict[str, dict] = {}

    # Dense retrieval
    vec = embed_query(query)
    dense_results = col.query(
        query_embeddings=[vec],
        n_results=n_fetch,
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    dense_ranked: list[str] = []
    for cid, doc, meta, dist in zip(
        dense_results["ids"][0],
        dense_results["documents"][0],
        dense_results["metadatas"][0],
        dense_results["distances"][0],
    ):
        dense_score = round(1 - dist, 4)
        dense_ranked.append(cid)
        id_to_data[cid] = _parse_hit(doc, meta, dense_score=dense_score)

    # BM25 retrieval
    bm25_tokens = clean_query(query)
    bm25_index  = _get_bm25_index(col)
    bm25_obj    = bm25_index["bm25"]
    corpus_ids  = bm25_index["ids"]
    corpus_docs = bm25_index["docs"]
    corpus_metas= bm25_index["metas"]

    raw_scores = bm25_obj.get_scores(bm25_tokens)
    if filter_doc:
        raw_scores = [
            s if corpus_metas[i].get("document_name") == filter_doc else 0.0
            for i, s in enumerate(raw_scores)
        ]

    top_bm25_idx = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:n_fetch]
    bm25_ranked: list[str] = []
    max_bm25 = raw_scores[top_bm25_idx[0]] if top_bm25_idx else 1.0
    for idx in top_bm25_idx:
        cid        = corpus_ids[idx]
        bm25_score = round(raw_scores[idx] / max_bm25, 4) if max_bm25 > 0 else 0.0
        bm25_ranked.append(cid)
        if cid not in id_to_data:
            id_to_data[cid] = _parse_hit(corpus_docs[idx], corpus_metas[idx], bm25_score=bm25_score)
        else:
            id_to_data[cid]["bm25_score"] = bm25_score

    fused = _rrf_fuse(dense_ranked, bm25_ranked, id_to_data, top_k=top_k)
    return fused


def retrieve(
    query:      str,
    client:     chromadb.ClientAPI,
    docs_root:  Path,
    top_k:      int = 5,
    filter_doc: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Retrieval pipeline with optional multi-query decomposition.

    1. Ollama gpt-oss:20b checks if the query is complex.
    2. If complex → split into sub-queries, run each in parallel (5 chunks each),
       merge all results, deduplicate by chunk id, then rerank the union.
    3. If simple → single hybrid (dense+BM25+RRF) retrieval, then rerank.
    4. Return (hits, query_meta) where query_meta carries decomposition details.
    """
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Run indexing.py first.", file=sys.stderr)
        return [], {}

    if col.count() == 0:
        print("Collection is empty. Run indexing.py first.", file=sys.stderr)
        return [], {}

    # ── Step 1: Query decomposition ───────────────────────────────────────────
    print(f"\n  Analysing query complexity via Ollama {OLLAMA_MODEL} …")
    is_complex, sub_queries = decompose_query(query)

    if is_complex and sub_queries:
        print(f"  Complex query detected → {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"    [{i}] {sq}")
        all_candidates = _multi_query_retrieve(sub_queries, col, filter_doc)
        origin_label   = "multi-query"
    else:
        print("  Simple query — single retrieval path.")
        bm25_tokens = clean_query(query)
        print(f"  BM25 query tokens (stopwords removed): {bm25_tokens}")
        all_candidates = _retrieve_single(query, col, top_k=None, filter_doc=filter_doc)
        origin_label   = "single"

    print(f"  [{origin_label}] {len(all_candidates)} unique candidate(s) → reranking …")

    # ── Step 2: Rerank all candidates against the ORIGINAL query ─────────────
    pairs      = [[query, h["raw_text"] or h["text"]] for h in all_candidates]
    raw_scores = _RERANKER.predict(pairs)
    for hit, rs in zip(all_candidates, raw_scores):
        hit["reranker_score"] = round(1.0 / (1.0 + math.exp(-float(rs))), 6)

    all_candidates.sort(key=lambda h: h["reranker_score"], reverse=True)
    hits = all_candidates[:top_k]

    for hit in hits:
        enrich_with_files(hit, docs_root)

    query_meta = {
        "original_query":    query,
        "is_complex":        is_complex,
        "retrieval_mode":    origin_label,
        "sub_queries":       sub_queries if is_complex else [],
        "candidates_before_rerank": len(all_candidates),
        "ollama_model":      OLLAMA_MODEL,
    }

    return hits, query_meta


def _multi_query_retrieve(
    sub_queries: list[str],
    col:         chromadb.Collection,
    filter_doc:  str | None,
) -> list[dict]:
    """
    Retrieve MULTI_QUERY_CHUNKS chunks for each sub-query in parallel,
    then merge and deduplicate by chunk_id.
    """
    # Build a stable chunk-id map for deduplication
    seen_ids:     dict[str, dict] = {}  # chunk_index key → hit (best score wins)

    def _fetch(sq: str) -> list[dict]:
        print(f"    → retrieving for: {sq!r}", flush=True)
        return _retrieve_single(sq, col, top_k=MULTI_QUERY_CHUNKS, filter_doc=filter_doc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
        futures = {pool.submit(_fetch, sq): sq for sq in sub_queries}
        for future in concurrent.futures.as_completed(futures):
            sq = futures[future]
            try:
                hits = future.result()
            except Exception as e:
                print(f"    [warning] Sub-query failed ({sq!r}): {e}")
                continue
            for hit in hits:
                # Use (document_name, chunk_index) as a stable dedup key
                key = f"{hit['document_name']}::{hit['chunk_index']}"
                if key not in seen_ids or hit["score"] > seen_ids[key]["score"]:
                    seen_ids[key] = hit

    return list(seen_ids.values())


# ══════════════════════════════════════════════════════════════════════════════
# Display + Save
# ══════════════════════════════════════════════════════════════════════════════

def display(query: str, hits: list[dict], query_meta: dict | None = None) -> None:
    sep  = "═" * 70
    thin = "─" * 70
    print(f"\n{sep}")
    print(f"  Query : {query}")
    if query_meta:
        mode = query_meta.get("retrieval_mode", "single")
        print(f"  Mode  : {mode}  (complex={query_meta.get('is_complex', False)}, "
              f"candidates before rerank={query_meta.get('candidates_before_rerank', '?')})")
        if query_meta.get("sub_queries"):
            print(f"  Sub-queries ({len(query_meta['sub_queries'])}):")
            for i, sq in enumerate(query_meta["sub_queries"], 1):
                print(f"    [{i}] {sq}")
    print(f"{sep}")

    for i, h in enumerate(hits, 1):
        pages = h["page_numbers"] or "?"
        print(f"\n  [{i}]  reranker={h.get('reranker_score',0):.6f}  rrf={h['score']:.6f}  "
              f"dense={h.get('dense_score',0):.4f}  bm25={h.get('bm25_score',0):.4f}   "
              f"doc={h['document_name']}   pages={pages}   type={h['chunk_type']}   chunk={h['chunk_index']}")
        if h["headings"]:
            print(f"       heading : {h['headings']}")
        if h["image_ref"]:
            print(f"       image   : {h['image_ref']}")
        files = h.get("files", {})
        if files.get("page_images"):
            print(f"       page img: {files['page_images'][0]}")
        if files.get("page_markdown"):
            print(f"       page md : {files['page_markdown'][0]}")

        provenance = h.get("provenance_parsed", [])
        if provenance:
            print(f"       bboxes  : {len(provenance)} region(s)")
            for prov in provenance[:3]:
                print(f"                 p{prov['page_no']}  "
                      f"({prov['x0']}, {prov['y0']}) → ({prov['x1']}, {prov['y1']})")
            if len(provenance) > 3:
                print(f"                 … +{len(provenance) - 3} more")

        print(thin)
        body  = h["raw_text"] or h["text"]
        words, line, lines = body.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 80:
                lines.append(line); line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        for ln in lines[:12]:
            print(f"  {ln}")
        if len(lines) > 12:
            print(f"  … [{len(lines) - 12} more lines]")

    print(f"\n{sep}")
    print(f"\n  Summary:")
    print(f"  {'#':<3}  {'Document':<22}  {'Pages':<8}  {'Score':<7}  Heading")
    print(f"  {'─'*3}  {'─'*22}  {'─'*8}  {'─'*7}  {'─'*30}")
    for i, h in enumerate(hits, 1):
        print(f"  {i:<3}  {h['document_name']:<22}  {h['page_numbers']:<8}  "
              f"{h['score']:<7}  {(h['headings'] or '')[:30]}")
    print()


def save_results(query: str, hits: list[dict], save_dir: Path, query_meta: dict | None = None) -> str:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug      = re.sub(r"[^a-z0-9]+", "_", query.lower())[:40].strip("_")
    out_file  = save_dir / f"retrieved_{slug}_{timestamp}.json"

    payload = {
        "query":      query,
        "timestamp":  timestamp,
        "total":      len(hits),
        "query_meta": query_meta or {"original_query": query},
        "results":    hits,
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved → {out_file}")
    return str(out_file)


# ══════════════════════════════════════════════════════════════════════════════
# Interactive loop
# ══════════════════════════════════════════════════════════════════════════════

def interactive(client: chromadb.ClientAPI, docs_root: Path, top_k: int, save_dir: Path) -> None:
    col = client.get_collection(COLLECTION_NAME)
    print(f"Collection     : {COLLECTION_NAME}  ({col.count()} chunks)")
    print(f"Top-k          : {top_k}")
    print(f"Docs root      : {docs_root}")
    print(f"Saving to      : {save_dir.resolve()}")
    print("\nType your query and press Enter.  Ctrl-C or 'exit' to quit.\n")

    while True:
        try:
            query = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not query or query.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break
        hits, query_meta = retrieve(query, client, docs_root, top_k)
        if not hits:
            print("  No results found.\n")
            continue
        display(query, hits, query_meta)
        save_results(query, hits, save_dir, query_meta)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Query ChromaDB rag_chunks collection"
    )
    p.add_argument("--db",       default=str(DEFAULT_DB_PATH),
                   help=f"ChromaDB directory (default: {DEFAULT_DB_PATH})")
    p.add_argument("--docs-root", default=str(DEFAULT_DOCS_ROOT),
                   help=f"Documents-OCR root directory (default: {DEFAULT_DOCS_ROOT})")
    p.add_argument("--top-k",    type=int, default=5,
                   help="Number of results to return (default: 5)")
    p.add_argument("--query",    nargs="+", default=[],
                   help="Query text")
    p.add_argument("--filter-doc", default=None,
                   help="Restrict search to a specific document_name")
    p.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR),
                   help=f"Directory to save result JSON files (default: {DEFAULT_SAVE_DIR})")
    p.add_argument("--device", default=None,
                   help="Compute device: 'cuda', 'cuda:0', 'cuda:1', 'mps'. Auto-detected when omitted.")
    args = p.parse_args()

    device = resolve_device(args.device)
    _init_models(device)

    client    = chromadb.PersistentClient(path=args.db)
    docs_root = Path(args.docs_root)
    save_dir  = Path(args.save_dir)
    query_str = " ".join(args.query).strip()

    if query_str:
        hits, query_meta = retrieve(query_str, client, docs_root, args.top_k, args.filter_doc)
        display(query_str, hits, query_meta)
        save_results(query_str, hits, save_dir, query_meta)
    else:
        interactive(client, docs_root, args.top_k, save_dir)
