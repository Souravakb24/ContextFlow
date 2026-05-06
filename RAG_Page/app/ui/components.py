"""
components.py — Reusable HTML component builders for ContextFlow UI.
"""

from __future__ import annotations

import html
from app.ui.theme import (
    PURPLE_DARK, PURPLE_MID, PURPLE_LIGHT, PURPLE_PALE,
    SUCCESS, WARNING, ERROR, TEXT_MID, TEXT_LIGHT, BORDER,
)


def _score_class(score: float) -> str:
    if score >= 0.75:
        return "score-high"
    if score >= 0.5:
        return "score-medium"
    return "score-low"


def _score_label(score: float) -> str:
    if score >= 0.75:
        return "●  High"
    if score >= 0.5:
        return "◑  Mid"
    return "○  Low"


def chunk_card_html(hit: dict, idx: int) -> str:
    """Render a single retrieved chunk as an HTML card."""
    doc   = html.escape(hit.get("document_name", "unknown"))
    pages = html.escape(hit.get("page_numbers", "?"))
    head  = html.escape(hit.get("headings", ""))
    ctype = html.escape(hit.get("chunk_type", "text"))
    score = hit.get("reranker_score", hit.get("score", 0.0))
    text  = html.escape((hit.get("raw_text") or hit.get("text", ""))[:300])
    sc    = _score_class(score)
    sl    = _score_label(score)

    heading_html = f'<div class="chunk-meta">📑 {head}</div>' if head else ""
    return f"""
<div class="chunk-card">
  <div class="chunk-card-header">
    <span class="chunk-doc-name">#{idx} · {doc}</span>
    <span class="chunk-score-badge {sc}">{sl} &nbsp;{score:.3f}</span>
  </div>
  <div class="chunk-meta">📄 Pages: {pages} &nbsp;|&nbsp; 🏷 {ctype}</div>
  {heading_html}
  <div class="chunk-text">{text}{"…" if len(hit.get("raw_text") or hit.get("text","")) > 300 else ""}</div>
</div>
""".strip()


def all_chunks_html(hits: list[dict]) -> str:
    if not hits:
        return (
            '<div style="color:#7A7A9A;text-align:center;padding:20px;'
            'border:1.5px dashed #D5CBF0;border-radius:10px;">'
            'Ask a question to see retrieved source chunks here.</div>'
        )
    return "\n".join(chunk_card_html(h, i + 1) for i, h in enumerate(hits))


def pdf_library_html(doc_names: list[str], active: str = "") -> str:
    if not doc_names:
        return '<div style="color:#7A7A9A;font-size:0.85rem;">No documents indexed yet.</div>'

    cards = []
    for name in doc_names:
        active_cls = " active" if name == active else ""
        cards.append(
            f'<div class="pdf-lib-card{active_cls}">'
            f'📄 {html.escape(name)}'
            f'</div>'
        )
    return "\n".join(cards)


def db_stats_html(n_docs: int, n_chunks: int) -> str:
    return f"""
<div style="background:#EDE7F6;border-radius:8px;padding:10px 14px;font-size:0.83rem;">
  <div style="color:{PURPLE_DARK};font-weight:700;margin-bottom:4px;">Database Stats</div>
  <div style="color:{TEXT_MID};">📚 Documents: <b>{n_docs}</b></div>
  <div style="color:{TEXT_MID};">🔢 Chunks: <b>{n_chunks}</b></div>
</div>
""".strip()


def status_html(message: str, kind: str = "info") -> str:
    """kind: 'info' | 'success' | 'error' | 'warning'"""
    colors = {
        "info":    (PURPLE_PALE, PURPLE_MID),
        "success": ("#E8F5E9", SUCCESS),
        "error":   ("#FFEBEE", ERROR),
        "warning": ("#FFF3E0", WARNING),
    }
    icons = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}
    bg, fg = colors.get(kind, colors["info"])
    icon   = icons.get(kind, "ℹ️")
    return (
        f'<div style="background:{bg};border-left:4px solid {fg};'
        f'border-radius:6px;padding:8px 12px;font-size:0.85rem;color:{fg};margin:4px 0;">'
        f'{icon} {html.escape(message)}</div>'
    )


def proof_page_header(doc_name: str, page_no: int, n_chunks: int) -> str:
    return (
        f'<div style="background:{PURPLE_PALE};border-radius:8px;padding:8px 14px;'
        f'font-size:0.85rem;color:{PURPLE_DARK};margin-bottom:8px;">'
        f'<b>📄 {html.escape(doc_name)}</b> &nbsp;·&nbsp; Page {page_no} '
        f'&nbsp;·&nbsp; <span style="color:{PURPLE_MID};">{n_chunks} chunk(s) highlighted</span>'
        f'</div>'
    )


def query_meta_html(meta: dict) -> str:
    if not meta:
        return ""
    mode      = meta.get("retrieval_mode", "single")
    is_cx     = meta.get("is_complex", False)
    cands     = meta.get("candidates_before_rerank", "?")
    sub_qs    = meta.get("sub_queries", [])
    mode_icon = "🔀" if is_cx else "🔍"

    sub_html = ""
    if sub_qs:
        items = "".join(f"<li>{html.escape(q)}</li>" for q in sub_qs)
        sub_html = f'<ul style="margin:4px 0 0 16px;font-size:0.8rem;">{items}</ul>'

    return (
        f'<div style="background:{PURPLE_PALE};border-radius:8px;padding:8px 14px;'
        f'font-size:0.83rem;color:{PURPLE_DARK};margin-bottom:8px;">'
        f'{mode_icon} Mode: <b>{mode}</b> &nbsp;·&nbsp; '
        f'Candidates before rerank: <b>{cands}</b>'
        f'{sub_html}</div>'
    )
