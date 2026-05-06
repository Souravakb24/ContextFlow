"""
handlers.py — All Gradio event callbacks. Calls core/ only, yields progress updates.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Generator

from app.config import cfg
from app.logger import get_logger
from app.core import session_cache
from app.ui.components import (
    chunk_card_html, pdf_library_html, db_stats_html,
    status_html, proof_page_header, query_meta_html,
)

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion handlers
# ══════════════════════════════════════════════════════════════════════════════

def _run_ingest_pipeline(files, force_reingest: bool, progress_yields: bool):
    """
    Shared ingestion logic used by both landing-page and library-drawer handlers.
    If progress_yields=True, yields (status, lib_html, stats).
    Otherwise returns the final tuple once done.
    """
    from app.core.ingestion import ingest_pdf
    from app.core.indexing import index_single_pdf
    from app.services.chromadb_service import list_indexed_documents, collection_count
    from app.core.retrieval import invalidate_bm25_cache

    log_lines: list[str] = []

    def _p(msg: str) -> str:
        log_lines.append(msg)
        return "\n".join(log_lines[-20:])

    if not files:
        yield "⚠️ No files selected.", "", ""
        return

    pdf_paths: list[Path] = []
    for f in files:
        src = Path(f.name) if hasattr(f, "name") else Path(str(f))
        dst = cfg.uploads_dir / src.name
        shutil.copy2(str(src), str(dst))
        pdf_paths.append(dst)
        log.info("Uploaded: %s → %s", src.name, dst)

    yield _p(f"Received {len(pdf_paths)} file(s). Starting pipeline …"), "", ""

    for pdf_path in pdf_paths:
        yield _p(f"─── Ingesting: {pdf_path.name} ───"), "", ""

        status_log: list[str] = []

        def cb(msg: str):
            status_log.append(msg)

        try:
            report = ingest_pdf(pdf_path, progress_cb=cb, force=force_reingest)
            for m in status_log:
                log_lines.append(m)
            yield _p(f"✅ Ingestion done: {pdf_path.stem} ({report['chunk_count']} chunks)"), "", ""
        except Exception as e:
            log.error("Ingestion failed for %s: %s", pdf_path.name, e, exc_info=True)
            yield _p(f"❌ Ingestion FAILED: {pdf_path.name} — {e}"), "", ""
            continue

        yield _p(f"Indexing '{pdf_path.stem}' into ChromaDB …"), "", ""
        try:
            idx_status: list[str] = []
            n = index_single_pdf(pdf_path.stem, progress_cb=lambda m: idx_status.append(m))
            for m in idx_status:
                log_lines.append(m)
            invalidate_bm25_cache()
            yield _p(f"✅ Indexed {n} chunks for '{pdf_path.stem}'."), "", ""
        except Exception as e:
            log.error("Indexing failed for %s: %s", pdf_path.stem, e, exc_info=True)
            yield _p(f"❌ Indexing FAILED: {pdf_path.stem} — {e}"), "", ""

    docs     = list_indexed_documents()
    n_chks   = collection_count()
    lib_html = pdf_library_html(docs)
    stats    = db_stats_html(len(docs), n_chks)
    yield _p("All done!"), lib_html, stats


def handle_upload_and_ingest(files, force_reingest: bool = False):
    """
    Generator handler for landing-page PDF upload.
    Yields (status_md, library_html, db_stats, doc_selector, filter_doc_dd).
    Page switch is handled by a .then() in app.py after this generator finishes.
    """
    import gradio as gr
    from app.services.chromadb_service import list_indexed_documents
    for status, lib, stats in _run_ingest_pipeline(files, force_reingest, True):
        lib_out   = lib or ""
        stats_out = stats or ""
        if lib:
            docs     = list_indexed_documents()
            doc_dd   = gr.update(choices=[" "] + docs, value=" ")
            filt_dd  = gr.update(choices=docs, value=[])
            yield status, lib_out, stats_out, doc_dd, filt_dd
        else:
            yield status, lib_out, stats_out, gr.update(), gr.update()


def handle_switch_to_main():
    """Called via .then() after ingest completes to switch the visible page."""
    import gradio as gr
    return gr.update(visible=True), gr.update(visible=False)


def handle_add_pdfs(files, force_reingest: bool = False):
    """
    Generator handler for adding PDFs from the library drawer (stays on main page).
    Yields (status_md, library_html, db_stats, doc_selector, filter_doc_dd).
    """
    import gradio as gr
    from app.services.chromadb_service import list_indexed_documents
    for status, lib, stats in _run_ingest_pipeline(files, force_reingest, True):
        if lib:
            docs    = list_indexed_documents()
            doc_dd  = gr.update(choices=[" "] + docs, value=" ")
            filt_dd = gr.update(choices=docs, value=[])
            yield status, lib, stats, doc_dd, filt_dd
        else:
            yield status, lib, stats, gr.update(), gr.update()


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval + chat handlers
# ══════════════════════════════════════════════════════════════════════════════

_N_CHUNK_SLOTS = 5  # must match N_CHUNK_SLOTS in app.py


def _chunk_slot_updates(hits: list) -> tuple[list[str], list]:
    """Return (per-slot HTML strings, per-slot row gr.update visibility list)."""
    import gradio as gr
    htmls = [chunk_card_html(hits[i], i + 1) if i < len(hits) else "" for i in range(_N_CHUNK_SLOTS)]
    rows  = [gr.update(visible=i < len(hits)) for i in range(_N_CHUNK_SLOTS)]
    return htmls, rows


def handle_query(
    query: str,
    top_k: int,
    filter_docs: list,          # multiselect → list of doc names (empty = all)
    session_id: str,
    chat_history: list,
) -> Generator:
    """
    Generator handler for query → retrieval → LLM synthesis → stream answer.
    Yields (chat_history, *chunk_htmls x5, query_meta_html, proof_imgs,
            status_md, session_id, *chunk_row_updates x5).
    """
    from app.core.retrieval import retrieve
    from app.services.llm_service import synthesize_answer_stream
    import gradio as gr

    _empty_htmls, _hidden_rows = _chunk_slot_updates([])

    if not query.strip():
        yield chat_history, *_empty_htmls, "", [], "⚠️ Empty query.", session_id, *_hidden_rows
        return

    if not session_id:
        session_id = session_cache.new_session()
    else:
        session_cache.delete_metadata(session_id)

    # Normalise filter list — remove blanks
    docs_filter = [d for d in (filter_docs or []) if d and d.strip()] or None
    log.info("Query: %r (top_k=%d, filter=%s)", query, top_k, docs_filter)

    # Gradio messages format
    chat_history = list(chat_history or [])
    chat_history.append({"role": "user",      "content": query})
    chat_history.append({"role": "assistant", "content": ""})

    retrieval_log: list[str] = []

    def _p(msg: str):
        retrieval_log.append(msg)

    yield chat_history, *_empty_htmls, "", [], "🔍 Retrieving relevant chunks …", session_id, *_hidden_rows

    try:
        hits, query_meta = retrieve(
            query=query,
            top_k=top_k,
            filter_docs=docs_filter,
            progress_cb=_p,
        )
    except Exception as e:
        log.error("Retrieval failed: %s", e, exc_info=True)
        chat_history[-1]["content"] = f"Retrieval error: {e}"
        yield chat_history, *_empty_htmls, "", [], f"❌ Retrieval error: {e}", session_id, *_hidden_rows
        return

    if not hits:
        chat_history[-1]["content"] = "No relevant documents found."
        yield chat_history, *_empty_htmls, "", [], "No results found.", session_id, *_hidden_rows
        return

    session_cache.save_retrieval_results(session_id, hits, query_meta)

    from app.core.proof import annotate_hits
    proof_data   = annotate_hits(hits)
    proof_images = [v["image"] for v in proof_data.values()]
    session_cache.save_metadata(session_id, {
        "proof_pages": [
            {"doc": v["document_name"], "page": v["page_no"], "chunks": v["chunks"]}
            for v in proof_data.values()
        ]
    })

    chunk_htmls, crow_updates = _chunk_slot_updates(hits)
    qmeta_html = query_meta_html(query_meta)

    yield chat_history, *chunk_htmls, qmeta_html, proof_images, "💬 Generating answer …", session_id, *crow_updates

    answer_tokens: list[str] = []

    try:
        for token in synthesize_answer_stream(query, hits):
            answer_tokens.append(token)
            chat_history[-1]["content"] = "".join(answer_tokens)
            yield chat_history, *chunk_htmls, qmeta_html, proof_images, "💬 Generating …", session_id, *crow_updates
    except Exception as e:
        log.error("LLM synthesis failed: %s", e, exc_info=True)
        chat_history[-1]["content"] = f"Error generating answer: {e}"

    log.info("Query complete. %d chunks retrieved.", len(hits))
    yield chat_history, *chunk_htmls, qmeta_html, proof_images, f"✅ Done — {len(hits)} chunks retrieved.", session_id, *crow_updates


# ══════════════════════════════════════════════════════════════════════════════
# Proof window handlers
# ══════════════════════════════════════════════════════════════════════════════

def handle_jump_to_chunk(hit_idx: int, session_id: str):
    """Select the gallery image that corresponds to this chunk's page."""
    import gradio as gr
    hits, _ = session_cache.load_retrieval_results(session_id)
    if not hits or hit_idx >= len(hits):
        return gr.update()

    meta = session_cache.load_metadata(session_id) or {}
    proof_pages = meta.get("proof_pages", [])

    # chunk_idx stored in proof_pages is 1-based
    target_chunk_idx = hit_idx + 1
    for gallery_idx, pp in enumerate(proof_pages):
        for chunk in pp.get("chunks", []):
            if chunk.get("chunk_idx") == target_chunk_idx:
                return gr.update(selected_index=gallery_idx)

    return gr.update(selected_index=0)


def handle_clear_chat(session_id: str):
    """Clear chat history and session metadata."""
    if session_id:
        session_cache.delete_metadata(session_id)
    empty_htmls, hidden_rows = _chunk_slot_updates([])
    return [], *empty_htmls, "", [], "Chat cleared.", *hidden_rows


# ══════════════════════════════════════════════════════════════════════════════
# Library / document management handlers
# ══════════════════════════════════════════════════════════════════════════════

def handle_toggle_library(is_open: bool):
    import gradio as gr
    new_state = not is_open
    return gr.update(visible=new_state), new_state


def handle_refresh_library():
    import gradio as gr
    from app.services.chromadb_service import list_indexed_documents, collection_count
    docs   = list_indexed_documents()
    n_chks = collection_count()
    doc_dd  = gr.update(choices=[" "] + docs, value=" ")      # single-select for delete
    filt_dd = gr.update(choices=docs, value=[])               # multiselect for filter
    return pdf_library_html(docs), db_stats_html(len(docs), n_chks), doc_dd, filt_dd


def handle_delete_document(doc_name: str):
    if not doc_name or doc_name.strip() == " ":
        return "⚠️ No document selected.", *handle_refresh_library()
    from app.services.chromadb_service import delete_document
    from app.core.retrieval import invalidate_bm25_cache
    try:
        delete_document(doc_name)
        invalidate_bm25_cache()
        log.info("Deleted document: %s", doc_name)
        return f"✅ Deleted '{doc_name}'.", *handle_refresh_library()
    except Exception as e:
        log.error("Delete failed for '%s': %s", doc_name, e)
        return f"❌ Delete failed: {e}", *handle_refresh_library()
