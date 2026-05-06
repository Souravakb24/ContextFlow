"""
ingestion.py — Full PDF ingestion pipeline:
  PDF → render pages → parallel OCR → VLM enrich → chunk → write outputs.
Supports PDF cache: if fingerprint matches, skips re-processing.
"""

from __future__ import annotations

import io
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from tqdm import tqdm

from app.config import cfg
from app.logger import get_logger
from app.utils.file_utils import (
    doc_output_dir, is_already_processed, pdf_fingerprint,
    save_json, chunks_path_for,
)
from app.utils.coord_utils import normalize_bbox

log = get_logger(__name__)

ProgressCallback = Callable[[str], None]


# ── Serialization helper ───────────────────────────────────────────────────────

def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


# ── Chunk bbox normalization ───────────────────────────────────────────────────

def _normalize_chunk_bboxes(chunk_payload: dict) -> None:
    first_bbox: Optional[dict] = None
    first_page: Optional[int]  = None
    for item in chunk_payload.get("meta", {}).get("doc_items", []):
        for prov in item.get("prov", []):
            raw = prov.get("bbox")
            if raw:
                normed = normalize_bbox(raw)
                if normed:
                    prov["bbox"] = normed
                    if first_bbox is None:
                        first_bbox = normed
            if first_page is None:
                page_no = prov.get("page_no")
                if page_no is not None:
                    first_page = page_no
    if first_bbox is not None:
        chunk_payload.setdefault("bbox", first_bbox)
    if first_page is not None:
        chunk_payload.setdefault("page", first_page)


# ── Markdown builders ─────────────────────────────────────────────────────────

def _extract_fig_id(caption: str) -> str:
    m = re.match(r"(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?)", caption, re.IGNORECASE)
    return m.group(1) if m else ""


def _picture_md_block(elem: dict) -> list[str]:
    ref  = elem.get("image_ref", "")
    cap  = (elem.get("caption") or "").strip()
    desc = (elem.get("description") or "").strip()
    img_type = (desc.split("\n")[0].strip().rstrip(" —:-") if desc else None) or cap or "figure"
    fig_id   = _extract_fig_id(cap)
    lines: list[str] = []
    if ref:
        lines.append(f"\n![{img_type}]({ref})\n")
    else:
        lines.append("\n<!-- figure: image not extracted -->\n")
    if fig_id:
        lines.append(f"\n**Figure:** {fig_id}")
    if cap:
        lines.append(f"**Caption:** {cap}")
    if desc:
        lines.append(f"**Description:**\n{desc}")
    lines.append("")
    return lines


def build_page_markdown(page_no: int, elements: list[dict]) -> str:
    lines: list[str] = []
    claimed = {
        (e.get("caption") or "").strip()
        for e in elements
        if e.get("type") == "picture" and e.get("caption")
    }
    for elem in elements:
        t       = elem.get("type", "")
        content = (elem.get("content") or "").strip()
        if t in ("title", "section_header"):
            depth = 1 if t == "title" else 2
            lines.append(f"\n{'#' * depth} {content}\n")
        elif t == "table":
            lines.append(f"\n{content}\n")
        elif t == "equation":
            latex = (elem.get("latex_vlm") or elem.get("latex") or content or "").strip()
            if latex and latex.upper() != "UNREADABLE":
                lines.append(f"\n$$\n{latex}\n$$\n")
            else:
                lines.append("\n<!-- equation: LaTeX could not be extracted -->\n")
        elif t == "code":
            lines.append(f"\n```\n{content}\n```\n")
        elif t == "list_item":
            lines.append(f"- {content}")
        elif t == "caption":
            if content not in claimed:
                lines.append(f"\n*{content}*\n")
        elif t == "picture":
            lines.extend(_picture_md_block(elem))
        elif content:
            lines.append(f"\n{content}\n")
    return "\n".join(lines)


# ── VLM enrichment ────────────────────────────────────────────────────────────

def enrich_with_vlm(
    page_elements: dict[int, list[dict]],
    pages_dir: Path,
    page_sizes: dict[int, dict],
    progress_cb: ProgressCallback | None = None,
) -> None:
    from app.services.vlm_service import describe_image, _path_to_b64
    from app.utils.coord_utils import pdf_to_img_coords
    from PIL import Image
    import base64

    work_items: list[tuple[int, int, str]] = []
    for page_no, elements in page_elements.items():
        for idx, elem in enumerate(elements):
            if elem.get("type") == "picture":
                work_items.append((page_no, idx, "image"))

    if not work_items:
        log.info("No pictures found — VLM enrichment skipped.")
        if progress_cb:
            progress_cb("No pictures to enrich — VLM step skipped.")
        return

    log.info("VLM enrichment: %d images to describe …", len(work_items))
    if progress_cb:
        progress_cb(f"VLM enrichment: {len(work_items)} images to describe …")

    for done, (page_no, idx, _) in enumerate(
        tqdm(work_items, desc="VLM enrichment", unit="img"), 1
    ):
        elem         = page_elements[page_no][idx]
        b64: str | None = None

        abs_path = elem.get("image_abs")
        if abs_path and Path(abs_path).exists():
            try:
                b64 = _path_to_b64(Path(abs_path))
            except Exception:
                pass

        if b64 is None:
            page_img = pages_dir / f"page_{page_no}.png"
            if page_img.exists():
                try:
                    psize = page_sizes.get(page_no, {})
                    img   = Image.open(page_img)
                    w, h  = img.size
                    pw    = psize.get("width", 595)
                    ph    = psize.get("height", 842)
                    bbox  = elem.get("bbox", {})
                    l, t, r, b = pdf_to_img_coords(
                        bbox.get("x0", 0), bbox.get("y0", 0),
                        bbox.get("x1", 0), bbox.get("y1", 0),
                        pw, ph, w, h,
                    )
                    if r > l and b > t:
                        cropped = img.crop((l, t, r, b))
                        buf = io.BytesIO()
                        cropped.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception as e:
                    log.warning("Could not crop page image for p%d elem %d: %s", page_no, idx, e)

        if not b64:
            log.warning("No image data for p%d elem %d — skipping VLM.", page_no, idx)
            continue

        value = ""
        for attempt in range(3):
            try:
                value = describe_image(b64)
                if not value.startswith("ERROR:"):
                    break
                if attempt < 2:
                    log.warning("VLM attempt %d failed (p%d): %s — retrying…", attempt + 1, page_no, value[:60])
                    time.sleep(2 ** attempt)
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    value = f"ERROR: {exc}"

        page_elements[page_no][idx]["description"] = value
        short = value[:60].replace("\n", " ") + ("…" if len(value) > 60 else "")
        status = "FAIL" if value.startswith("ERROR:") else f"OK [{done}/{len(work_items)}]"
        log.info("VLM p%d: %s — %s", page_no, status, short)
        if progress_cb:
            progress_cb(f"VLM [{done}/{len(work_items)}] page {page_no}: {short}")


# ── Figure chunks ─────────────────────────────────────────────────────────────

def build_figure_chunks(page_elements: dict[int, list[dict]]) -> list[dict]:
    chunks: list[dict] = []
    for page_no in sorted(page_elements.keys()):
        current_heading = ""
        for elem in page_elements[page_no]:
            t = elem.get("type", "")
            if t in ("title", "section_header"):
                current_heading = (elem.get("content") or "").strip()
            elif t == "picture":
                desc = (elem.get("description") or "").strip()
                if not desc:
                    continue
                cap   = (elem.get("caption") or "").strip()
                parts = []
                if current_heading:
                    parts.append(current_heading)
                if cap:
                    parts.append(cap)
                parts.append(desc)
                chunks.append({
                    "chunker":        "figure",
                    "type":           "figure",
                    "page":           page_no,
                    "parent_heading": current_heading,
                    "caption":        cap,
                    "image_ref":      elem.get("image_ref", ""),
                    "bbox":           elem.get("bbox"),
                    "text":           "\n".join(parts),
                })
    log.info("Built %d figure chunks.", len(chunks))
    return chunks


# ── Docling chunking ──────────────────────────────────────────────────────────

def build_chunks(
    dl_doc,
    chunker_type: str = "hybrid",
    tokenizer: str | None = None,
    max_tokens: int | None = None,
    merge_peers: bool = True,
    include_raw_text: bool = True,
) -> list[dict]:
    from docling.chunking import HybridChunker, HierarchicalChunker

    tok = tokenizer or cfg.tokenizer
    if chunker_type == "hybrid":
        kwargs: dict = {"tokenizer": tok, "merge_peers": merge_peers}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        chunker = HybridChunker(**kwargs)
    else:
        chunker = HierarchicalChunker()

    chunks: list[dict] = []
    for index, chunk in enumerate(tqdm(chunker.chunk(dl_doc=dl_doc), desc="Chunking", unit="chunk")):
        ctx_text  = chunker.contextualize(chunk=chunk)
        raw_text  = getattr(chunk, "text", None)
        payload   = _to_jsonable(chunk)
        payload.update({
            "chunk_index": index,
            "chunker":     chunker_type,
            "text":        ctx_text,
        })
        if include_raw_text:
            payload["raw_text"] = raw_text
        _normalize_chunk_bboxes(payload)
        chunks.append(payload)

    log.info("Built %d text chunks.", len(chunks))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdf(
    pdf_path: Path,
    progress_cb: ProgressCallback | None = None,
    force: bool = False,
) -> dict:
    """
    Full ingestion pipeline. Returns the ingestion report dict.
    If the PDF fingerprint matches a previous run, skips processing (cache hit).
    Set force=True to re-process regardless.
    """
    from app.services.ocr_service import (
        run_parallel_ocr, run_docling_fast, sort_elements_reading_order,
    )
    from app.utils.pdf_utils import render_pages

    t_start = time.perf_counter()

    def _p(msg: str):
        log.info("[INGEST] %s", msg)
        if progress_cb:
            progress_cb(msg)

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_name   = pdf_path.stem
    docs_dir   = cfg.docs_dir
    out        = doc_output_dir(docs_dir, pdf_name)
    pages_dir  = out / "pages"
    layout_dir = out / "layout"
    ocr_dir    = out / "pages_md"
    images_dir = out / "images"
    fp         = pdf_fingerprint(pdf_path)

    # ── Cache check ────────────────────────────────────────────────────────────
    if not force and is_already_processed(pdf_path, docs_dir):
        _p(f"'{pdf_name}' already processed — loading from cache.")
        report_path = out / "ingestion_report.json"
        if report_path.exists():
            import json as _json
            return _json.loads(report_path.read_text(encoding="utf-8"))
        # Report missing but flag present — re-process
        _p("Ingestion report missing — re-processing.")

    _p(f"Starting ingestion: {pdf_path.name}")

    # ── Step 1: Render page images ────────────────────────────────────────────
    _p("Step 1/6 — Rendering page images …")
    page_paths = render_pages(pdf_path, pages_dir, dpi=cfg.page_render_dpi)
    _p(f"  Rendered {len(page_paths)} pages.")

    # ── Step 2: Parallel OCR ──────────────────────────────────────────────────
    _p(f"Step 2/6 — Running parallel Docling OCR ({cfg.docling_workers} workers) …")
    page_elements, page_sizes = run_parallel_ocr(
        pdf_path, images_dir,
        n_workers=cfg.docling_workers,
        progress_cb=_p,
    )
    _p(f"  OCR complete: {len(page_elements)} pages.")

    # ── Step 3: Reading order sort ─────────────────────────────────────────────
    _p("Step 3/6 — Sorting reading order …")
    sort_elements_reading_order(page_elements, page_sizes)
    total_elems = sum(len(v) for v in page_elements.values())
    _p(f"  {total_elems} elements sorted across {len(page_elements)} pages.")

    # ── Step 4: VLM enrichment ────────────────────────────────────────────────
    if cfg.skip_vlm:
        _p("Step 4/6 — VLM enrichment SKIPPED (SKIP_VLM=true).")
    else:
        _p(f"Step 4/6 — VLM enrichment via Ollama ({cfg.vlm_model}) …")
        enrich_with_vlm(page_elements, pages_dir, page_sizes, progress_cb=_p)
        _p("  VLM enrichment complete.")

    # ── Step 5: Layout JSON + Markdown ────────────────────────────────────────
    _p("Step 5/6 — Saving layout JSON + markdown …")
    layout_dir.mkdir(parents=True, exist_ok=True)
    all_pages: list[dict] = []
    for page_no, elements in tqdm(sorted(page_elements.items()), desc="Layout JSON", unit="page"):
        clean = [{k: v for k, v in e.items() if k not in ("image_abs", "_needs_vlm")} for e in elements]
        layout = {
            "pdf_name":      pdf_name,
            "page_no":       page_no,
            "page_image":    f"../pages/page_{page_no}.png",
            **page_sizes.get(page_no, {}),
            "element_count": len(clean),
            "elements":      clean,
        }
        (layout_dir / f"page_{page_no}_layout.json").write_text(
            json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        all_pages.append(layout)
        md_txt = build_page_markdown(page_no, elements)
        ocr_dir.mkdir(parents=True, exist_ok=True)
        (ocr_dir / f"page_{page_no}.md").write_text(md_txt, encoding="utf-8")

    all_pages_path = layout_dir / "all_pages.json"
    all_pages_path.write_text(json.dumps(all_pages, indent=2, ensure_ascii=False), encoding="utf-8")

    # Full markdown
    parts = []
    for page_no in sorted(page_elements.keys()):
        p = ocr_dir / f"page_{page_no}.md"
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    (out / f"{pdf_name}.md").write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    _p(f"  Layout + markdown saved.")

    # ── Step 6: Chunking ──────────────────────────────────────────────────────
    _p("Step 6/6 — Chunking (fast Docling pass + chunker) …")
    fast_result = run_docling_fast(pdf_path)
    fast_doc    = fast_result.document

    chunks = build_chunks(
        dl_doc=fast_doc,
        chunker_type=cfg.chunker_type,
        max_tokens=cfg.max_tokens,
        merge_peers=cfg.merge_peers,
        include_raw_text=cfg.include_raw_text,
    )
    figure_chunks = build_figure_chunks(page_elements)

    # Merge figure chunks in page order
    if figure_chunks:
        def _chunk_page(c: dict) -> int:
            try:
                return c["meta"]["doc_items"][0]["prov"][0]["page_no"]
            except (KeyError, IndexError, TypeError):
                return 0

        all_chunks: list[dict] = []
        reg_by_page: dict[int, list] = defaultdict(list)
        fig_by_page: dict[int, list] = defaultdict(list)
        for c in chunks:
            reg_by_page[_chunk_page(c)].append(c)
        for c in figure_chunks:
            fig_by_page[c["page"]].append(c)
        for pno in sorted(set(list(reg_by_page) + list(fig_by_page))):
            all_chunks.extend(reg_by_page.get(pno, []))
            all_chunks.extend(fig_by_page.get(pno, []))
        for i, c in enumerate(all_chunks):
            c["chunk_index"] = i
        chunks = all_chunks

    chunks_path = chunks_path_for(docs_dir, pdf_name, cfg.chunker_type)
    chunks_payload = {
        "source":             str(pdf_path),
        "document_name":      pdf_name,
        "chunker":            cfg.chunker_type,
        "chunk_count":        len(chunks),
        "figure_chunk_count": len(figure_chunks),
        "vlm_enriched":       not cfg.skip_vlm,
        "chunks":             chunks,
    }
    chunks_path.write_text(
        json.dumps(_to_jsonable(chunks_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _p(f"  {len(chunks)} chunks saved ({len(figure_chunks)} figure chunks).")

    # ── Ingestion report ──────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    from collections import Counter
    type_counts = Counter(
        e.get("type", "other")
        for elems in page_elements.values()
        for e in elems
    )
    report = {
        "pdf":             str(pdf_path),
        "pdf_name":        pdf_name,
        "fingerprint":     fp,
        "parsed_at":       time.strftime("%Y%m%d_%H%M%S"),
        "total_pages":     max(page_elements.keys(), default=0),
        "total_elements":  sum(type_counts.values()),
        "element_counts":  dict(type_counts),
        "chunker":         cfg.chunker_type,
        "chunk_count":     len(chunks),
        "figure_chunk_count": len(figure_chunks),
        "vlm_enriched":    not cfg.skip_vlm,
        "elapsed_seconds": round(elapsed, 2),
        "outputs": {
            "layout_json":  str(all_pages_path),
            "chunks_json":  str(chunks_path),
            "pages_dir":    str(pages_dir),
            "pages_md_dir": str(ocr_dir),
            "images_dir":   str(images_dir),
            "full_md":      str(out / f"{pdf_name}.md"),
        },
    }
    report_path = out / "ingestion_report.json"
    save_json(report_path, report)
    _p(f"Ingestion complete in {elapsed:.1f}s. Report saved.")
    return report
