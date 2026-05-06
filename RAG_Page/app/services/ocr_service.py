"""
ocr_service.py — Docling parallel OCR worker pool.
Splits PDF into chunks, runs full OCR in parallel processes, merges results.
"""

from __future__ import annotations

import math
import multiprocessing
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import fitz
from tqdm import tqdm

from app.config import cfg
from app.logger import get_logger

log = get_logger(__name__)


# ── Docling pipeline options ───────────────────────────────────────────────────

def _make_full_pipeline_options():
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    opts = PdfPipelineOptions()
    opts.do_ocr                                   = True
    opts.do_table_structure                       = True
    opts.table_structure_options.do_cell_matching = True
    opts.table_structure_options.mode             = TableFormerMode.FAST
    opts.generate_page_images                     = False
    opts.generate_picture_images                  = True
    opts.images_scale                             = 3.0
    opts.do_formula_enrichment                    = True
    return opts


def _make_fast_pipeline_options():
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    opts = PdfPipelineOptions()
    opts.do_ocr                                   = False
    opts.do_table_structure                       = True
    opts.table_structure_options.do_cell_matching = False
    opts.table_structure_options.mode             = TableFormerMode.FAST
    opts.generate_page_images                     = False
    opts.generate_picture_images                  = False
    opts.do_formula_enrichment                    = False
    return opts


# ── Top-level worker (must be module-level for pickling) ─────────────────────

def _docling_worker(args: tuple) -> tuple[dict, dict]:
    """
    Runs full Docling OCR on a sub-PDF.
    Returns (page_elements, page_sizes) with global page numbers.
    """
    sub_pdf_str, page_offset, images_dir_str = args
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    sub_pdf  = Path(sub_pdf_str)
    img_dir  = Path(images_dir_str)

    opts      = _make_full_pipeline_options()
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(str(sub_pdf))
    doc    = result.document

    page_elements = _build_page_element_map(doc, img_dir, page_offset)
    page_sizes    = _get_page_sizes(doc, page_offset)
    return page_elements, page_sizes


# ── Element map builder ───────────────────────────────────────────────────────

def _bbox_dict(bbox) -> dict:
    return {
        "x0": round(float(bbox.l), 2),
        "y0": round(float(bbox.t), 2),
        "x1": round(float(bbox.r), 2),
        "y1": round(float(bbox.b), 2),
    }


def _build_page_element_map(doc, images_dir: Path, page_offset: int = 0) -> dict[int, list[dict]]:
    from docling_core.types.doc import DocItemLabel, PictureItem, TableItem

    page_elements: dict[int, list[dict]] = defaultdict(list)
    pic_counters:  dict[int, int]        = defaultdict(int)

    for item, level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        for prov in item.prov:
            page_no = prov.page_no + page_offset
            label   = item.label
            elem: dict[str, Any] = {
                "type":  label.value if hasattr(label, "value") else str(label),
                "bbox":  _bbox_dict(prov.bbox),
                "level": level,
            }
            if hasattr(item, "text") and item.text:
                elem["content"] = item.text

            if isinstance(item, TableItem):
                elem["type"] = "table"
                try:
                    elem["content"] = item.export_to_markdown()
                except Exception:
                    elem["content"] = ""
                try:
                    df = item.export_to_dataframe()
                    elem["table_data"] = df.to_dict(orient="records")
                    elem["columns"]    = df.columns.tolist()
                    elem["rows"]       = len(df)
                    elem["cols"]       = len(df.columns)
                except Exception:
                    pass

            elif isinstance(item, PictureItem):
                elem["type"] = "picture"
                pic_counters[page_no] += 1
                idx          = pic_counters[page_no]
                pic_filename = f"page_{page_no}_pic_{idx}.png"
                abs_path     = images_dir / pic_filename
                try:
                    img = item.get_image(doc)
                    if img is not None:
                        img.save(str(abs_path))
                        elem["image_ref"] = f"images/{pic_filename}"
                        elem["image_abs"] = str(abs_path)
                except Exception:
                    pass
                try:
                    elem["caption"] = item.caption_text(doc)
                except Exception:
                    pass
                elem["description"] = ""

            elif label.value == "formula" if hasattr(label, "value") else str(label) == "formula":
                elem["type"] = "equation"
                text_val = (getattr(item, "text", "") or "").strip()
                if text_val:
                    elem["latex"] = text_val
                else:
                    elem["_needs_vlm"] = True
                elem["latex_vlm"] = ""

            page_elements[page_no].append(elem)

    return dict(page_elements)


def _get_page_sizes(doc, page_offset: int = 0) -> dict[int, dict]:
    sizes: dict[int, dict] = {}
    if not hasattr(doc, "pages"):
        return sizes
    for key, page in doc.pages.items():
        try:
            local_no = int(key)
        except (TypeError, ValueError):
            local_no = getattr(page, "page_no", None)
        if local_no and hasattr(page, "size") and page.size:
            sizes[local_no + page_offset] = {
                "width":  round(float(page.size.width),  2),
                "height": round(float(page.size.height), 2),
            }
    return sizes


# ── PDF splitter ──────────────────────────────────────────────────────────────

def split_pdf(pdf_path: Path, n_workers: int, tmp_dir: Path) -> list[tuple[str, int]]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[str, int]] = []
    with fitz.open(str(pdf_path)) as src:
        total    = src.page_count
        chunk_sz = max(1, math.ceil(total / n_workers))
        log.info("Splitting %d pages into chunks of ~%d for %d workers.", total, chunk_sz, n_workers)
        for i in range(n_workers):
            start = i * chunk_sz
            end   = min(start + chunk_sz, total)
            if start >= total:
                break
            sub      = fitz.open()
            sub.insert_pdf(src, from_page=start, to_page=end - 1)
            sub_path = tmp_dir / f"chunk_{i:02d}.pdf"
            sub.save(str(sub_path))
            sub.close()
            chunks.append((str(sub_path), start))
            log.debug("  Sub-PDF %d: pages %d-%d (offset %d)", i, start + 1, end, start)
    return chunks


# ── Reading order sort ────────────────────────────────────────────────────────

def sort_elements_reading_order(
    page_elements: dict[int, list[dict]],
    page_sizes:    dict[int, dict],
) -> None:
    for page_no, elements in page_elements.items():
        pw = (page_sizes.get(page_no) or {}).get("width", 600)
        def _key(e, _pw=pw):
            b = e.get("bbox") or {}
            l = b.get("x0", 0)
            t = b.get("y0", 0)
            return (0 if l < _pw / 2 else 1, -t)
        elements.sort(key=_key)


# ── Fast full-doc pass for chunking ──────────────────────────────────────────

def run_docling_fast(pdf_path: Path):
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    log.info("Running fast Docling pass (no OCR) on full document …")
    opts      = _make_fast_pipeline_options()
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(str(pdf_path))
    log.info("Fast Docling pass complete.")
    return result


# ── Main parallel OCR runner ──────────────────────────────────────────────────

def run_parallel_ocr(
    pdf_path:   Path,
    images_dir: Path,
    n_workers:  int | None = None,
    progress_cb=None,
) -> tuple[dict[int, list[dict]], dict[int, dict]]:
    """
    Run Docling OCR in parallel across sub-PDFs.
    progress_cb(message: str) is called with status updates for the UI.
    Returns (page_elements, page_sizes).
    """
    n_w     = n_workers or cfg.docling_workers
    tmp_dir = Path(tempfile.mkdtemp(prefix="contextflow_ocr_"))

    def _progress(msg: str):
        log.info(msg)
        if progress_cb:
            progress_cb(msg)

    try:
        chunks_info   = split_pdf(pdf_path, n_w, tmp_dir)
        actual_workers = len(chunks_info)
        _progress(f"Starting {actual_workers} parallel OCR workers …")

        worker_args = [
            (sub_path, offset, str(images_dir))
            for sub_path, offset in chunks_info
        ]

        page_elements: dict[int, list[dict]] = {}
        page_sizes:    dict[int, dict]       = {}
        completed = 0

        spawn_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=actual_workers, mp_context=spawn_ctx) as pool:
            futures = {pool.submit(_docling_worker, a): i for i, a in enumerate(worker_args)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="OCR workers", unit="chunk"):
                worker_idx = futures[fut]
                try:
                    pe, ps = fut.result()
                    page_elements.update(pe)
                    page_sizes.update(ps)
                    completed += 1
                    _progress(f"OCR worker {worker_idx} complete ({completed}/{actual_workers}). Pages so far: {len(page_elements)}")
                except Exception as e:
                    log.error("OCR worker %d failed: %s", worker_idx, e, exc_info=True)
                    _progress(f"OCR worker {worker_idx} FAILED: {e}")

        log.info("Parallel OCR complete: %d pages merged.", len(page_elements))
        return page_elements, page_sizes

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.debug("Temp OCR dir cleaned up.")
