#!/usr/bin/env python3
"""
Combined PDF Ingestion + Chunking Pipeline  ─  Multiprocessing Edition
═══════════════════════════════════════════════════════════════════════════════
Splits the PDF into page-range chunks and runs Docling OCR + layout +
formula enrichment in parallel worker processes, then merges results.

For chunking a fast second Docling pass (OCR disabled) on the full PDF
gives the complete DoclingDocument so chunks span page boundaries normally.

Outputs (all under <output_root>/<pdf_name>/):
  pages/page_<N>.png
  layout/page_<N>_layout.json
  layout/all_pages.json
  pages_md/page_<N>.md
  <pdf_name>.md
  images/
  ingestion_report.json
  <chunker>_chunks.json

Examples:
    python3 injestion+chunking_multi.py ./paper.pdf ./output
    python3 injestion+chunking_multi.py ./paper.pdf ./output --docling-workers 4
    python3 injestion+chunking_multi.py ./paper.pdf ./output --no-vlm --chunker hierarchical

Install:
    pip install pymupdf docling aiohttp pillow transformers
    # And have Ollama running with: ollama pull qwen3-vl:30b
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
from PIL import Image

from docling.chunking import HierarchicalChunker, HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, DoclingDocument, PictureItem, TableItem


class ChunkerType(str, Enum):
    HYBRID       = "hybrid"
    HIERARCHICAL = "hierarchical"


# ── Pin process to available CPUs ─────────────────────────────────────────────
_TOTAL_CPUS  = os.cpu_count() or 1
_PINNED_CPUS = min(200, _TOTAL_CPUS)
os.sched_setaffinity(0, set(range(_PINNED_CPUS)))
print(f"CPU affinity: using {_PINNED_CPUS} of {_TOTAL_CPUS} logical CPUs")

_PIPELINE_START = time.perf_counter()

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_MODEL = "qwen3-vl:30b"
OLLAMA_URL   = "http://localhost:11434/v1/chat/completions"

FORMULA_RENDER_DPI = 300

# Each Docling worker loads heavy models; keep N modest relative to core count.
_DEFAULT_DOCLING_WORKERS = min(8, max(2, _TOTAL_CPUS // 4))


# ── Prompts ───────────────────────────────────────────────────────────────────
IMAGE_DESCRIPTION_PROMPT = """\
You will receive an image. Classify it into one of the categories below and respond ONLY with the description — no preamble, no labels, no meta-commentary.

---

CATEGORY 1 — LOGO
Condition: Image is a logo or brand mark.
Output: "Logo: <brand name or brief description>"

CATEGORY 2 — FLOWCHART / FLOW DIAGRAM
Condition: Image contains boxes, arrows, or flow-based connections.
Output:
  Blocks:
    - <block 1>
    - <block 2>
    ...
  Connections:
    - BlockA -> BlockB (<label if any>)
    ...
  Summary: <one paragraph describing the overall process>

CATEGORY 3 — PLOT / CHART / GRAPH
Condition: Image is a data visualization (bar, line, pie, scatter, etc.).
Output:
  Chart type: <type>
  Title: <title if visible>
  Axes: X — <label>, Y — <label>
  Trend: <one sentence>
  Key data points:
  | <column> | <column> |
  |----------|----------|
  | ...      | ...      |

CATEGORY 4 — SIMPLE BLOCK WITH TEXT
Condition: Image is a plain box or shape containing text.
Output: <transcribe only the text inside the block, nothing else>

CATEGORY 5 — SIMPLE BLOCK (NO TEXT)
Condition: Image is a plain geometric shape with no meaningful content.
Output: (empty — output nothing)

CATEGORY 6 — OTHER IMAGE
Condition: Anything not covered above.
Output: <one thorough paragraph covering subject, key visual elements, colours, spatial layout, and any visible text or numbers>

---

If the image is unreadable or too low quality to interpret, output exactly: UNREADABLE
"""


# FORMULA_EXTRACTION_PROMPT = """\
# This image contains a mathematical formula or equation. Your task is to \
# transcribe it into valid LaTeX.

# Rules:
# - Output ONLY the LaTeX code for the formula, nothing else.
# - Do NOT wrap the output in $ or $$ or code fences.
# - Do NOT add any explanation, preamble, or commentary.
# - Use standard LaTeX math commands (\\frac, \\sum, \\int, \\sqrt, etc.).
# - If the image contains multiple numbered equations, separate them with \\\\.
# - If the formula is unreadable, respond with exactly: UNREADABLE
# """

FORMULA_EXTRACTION_PROMPT = """<formula>"""

# ══════════════════════════════════════════════════════════════════════════════
# Ollama VLM helpers  (sequential — one call at a time, zero CPU spin)
# ══════════════════════════════════════════════════════════════════════════════

def _image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _path_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def call_ollama_vision(
    b64_image: str,
    prompt: str,
    timeout: int = 120,
    retries: int = 2,
) -> str:
    import urllib.request, urllib.error
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as exc:
            if "Connection refused" in str(exc):
                return "ERROR: Ollama not reachable — is it running on localhost:11434?"
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return f"ERROR: {exc}"
        except Exception as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return f"ERROR: {exc}"
    return "ERROR: max retries exceeded"


def describe_image(b64: str) -> str:
    return call_ollama_vision(b64, IMAGE_DESCRIPTION_PROMPT)


def extract_formula(b64: str) -> str:
    return call_ollama_vision(b64, FORMULA_EXTRACTION_PROMPT)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Render pages → PNG
# ══════════════════════════════════════════════════════════════════════════════

def render_pages(pdf_path: Path, out_dir: Path, dpi: int = 300) -> dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    mat   = fitz.Matrix(scale, scale)
    paths: dict[int, Path] = {}

    with fitz.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            p   = out_dir / f"page_{i}.png"
            pix.save(str(p))
            paths[i] = p

    return paths

# ══════════════════════════════════════════════════════════════════════════════
# 2. Run Docling  (two variants: full OCR pass and fast text-only pass)
# ══════════════════════════════════════════════════════════════════════════════

def run_docling(pdf_path: Path):
    """Full OCR + layout + formula pass used by parallel workers."""
    opts = PdfPipelineOptions()
    opts.do_ocr                                   = True
    opts.do_table_structure                       = True
    opts.table_structure_options.do_cell_matching = True
    opts.table_structure_options.mode             = TableFormerMode.FAST
    opts.generate_page_images                     = False
    opts.generate_picture_images                  = True
    opts.images_scale                             = 3.0
    opts.do_formula_enrichment                    = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return converter.convert(str(pdf_path))


def run_docling_fast(pdf_path: Path):
    """Fast pass on the full PDF for chunking only — no OCR, no formula model.
    Uses the PDF's embedded text layer so it completes in seconds.
    Produces a complete DoclingDocument so the chunker sees the whole document
    and can create cross-page chunks normally."""
    opts = PdfPipelineOptions()
    opts.do_ocr                                   = False
    opts.do_table_structure                       = True
    opts.table_structure_options.do_cell_matching = False
    opts.table_structure_options.mode             = TableFormerMode.FAST
    opts.generate_page_images                     = False
    opts.generate_picture_images                  = False
    opts.do_formula_enrichment                    = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return converter.convert(str(pdf_path))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Build per-page element map
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_dict(bbox) -> dict:
    return {
        "x0": round(float(bbox.l), 2),
        "y0": round(float(bbox.t), 2),
        "x1": round(float(bbox.r), 2),
        "y1": round(float(bbox.b), 2),
    }


def _normalize_bbox(b: Any) -> Optional[dict]:
    """Convert any bbox representation to {x0, y0, x1, y1}."""
    if not b or not isinstance(b, dict):
        return None
    if "x0" in b:
        return {
            "x0": round(float(b["x0"]), 2),
            "y0": round(float(b["y0"]), 2),
            "x1": round(float(b["x1"]), 2),
            "y1": round(float(b["y1"]), 2),
        }
    if "l" in b:
        return {
            "x0": round(float(b["l"]), 2),
            "y0": round(float(b["t"]), 2),
            "x1": round(float(b["r"]), 2),
            "y1": round(float(b["b"]), 2),
        }
    return None


def _normalize_chunk_bboxes(chunk_payload: dict) -> None:
    """Normalize all embedded prov bboxes to {x0,y0,x1,y1} and add top-level bbox + page."""
    first_bbox: Optional[dict] = None
    first_page: Optional[int] = None

    for item in chunk_payload.get("meta", {}).get("doc_items", []):
        for prov in item.get("prov", []):
            raw = prov.get("bbox")
            if raw:
                normed = _normalize_bbox(raw)
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


def build_page_element_map(
    doc,
    images_dir: Path,
    page_offset: int = 0,
) -> dict[int, list[dict]]:
    """page_offset: add to Docling's local page numbers to get global page numbers.
    When processing a sub-PDF that starts at global page 14, pass page_offset=13."""
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
                abs_pic_path = images_dir / pic_filename

                try:
                    img = item.get_image(doc)
                    if img is not None:
                        img.save(str(abs_pic_path))
                        elem["image_ref"] = f"images/{pic_filename}"
                        elem["image_abs"] = str(abs_pic_path)
                except Exception:
                    pass

                try:
                    elem["caption"] = item.caption_text(doc)
                except Exception:
                    pass

                elem["description"] = ""

            elif label == DocItemLabel.FORMULA:
                elem["type"] = "equation"
                text_val = (getattr(item, "text", "") or "").strip()
                if text_val:
                    elem["latex"] = text_val
                else:
                    elem["_needs_vlm"] = True
                elem["latex_vlm"] = ""

            page_elements[page_no].append(elem)

    return dict(page_elements)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Reading order sort + page sizes
# ══════════════════════════════════════════════════════════════════════════════

def sort_elements_reading_order(
    page_elements: dict[int, list[dict]],
    page_sizes: dict[int, dict],
) -> None:
    for page_no, elements in page_elements.items():
        page_w = (page_sizes.get(page_no) or {}).get("width", 600)

        def _reading_order(e, _pw=page_w):
            bbox = e.get("bbox") or {}
            l = bbox.get("x0", 0)
            t = bbox.get("y0", 0)
            col = 0 if l < _pw / 2 else 1
            return (col, -t)

        elements.sort(key=_reading_order)


def get_page_sizes(doc, page_offset: int = 0) -> dict[int, dict]:
    sizes: dict[int, dict] = {}
    if not hasattr(doc, "pages"):
        return sizes
    for key, page in doc.pages.items():
        try:
            local_page_no = int(key)
        except (TypeError, ValueError):
            local_page_no = getattr(page, "page_no", None)
        if local_page_no and hasattr(page, "size") and page.size:
            global_page_no = local_page_no + page_offset
            sizes[global_page_no] = {
                "width":  round(float(page.size.width),  2),
                "height": round(float(page.size.height), 2),
            }
    return sizes


# ══════════════════════════════════════════════════════════════════════════════
# 5. PDF splitting + parallel Docling workers
# ══════════════════════════════════════════════════════════════════════════════

def _split_pdf(
    pdf_path: Path,
    n_workers: int,
    tmp_dir: Path,
) -> list[tuple[str, int]]:
    """Split PDF into up to n_workers sub-PDFs.
    Returns list of (sub_pdf_path_str, page_offset) where page_offset is the
    0-based index of the first page of the chunk in the original PDF."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[str, int]] = []

    with fitz.open(str(pdf_path)) as src:
        total     = src.page_count
        chunk_sz  = max(1, math.ceil(total / n_workers))

        for i in range(n_workers):
            start = i * chunk_sz          # 0-based
            end   = min(start + chunk_sz, total)
            if start >= total:
                break

            sub = fitz.open()
            sub.insert_pdf(src, from_page=start, to_page=end - 1)
            sub_path = tmp_dir / f"chunk_{i:02d}.pdf"
            sub.save(str(sub_path))
            sub.close()

            # page_offset = start so that local page 1 maps to global page start+1
            chunks.append((str(sub_path), start))

    return chunks


def _docling_worker(args: tuple) -> tuple[dict, dict]:
    """Top-level worker — must be picklable (not a nested/lambda function).
    Runs full Docling on a sub-PDF and returns (page_elements, page_sizes)
    with page numbers already shifted to global coordinates."""
    sub_pdf_str, page_offset, images_dir_str = args
    sub_pdf_path = Path(sub_pdf_str)
    images_dir   = Path(images_dir_str)

    result = run_docling(sub_pdf_path)
    doc    = result.document

    page_elements = build_page_element_map(doc, images_dir, page_offset=page_offset)
    page_sizes    = get_page_sizes(doc, page_offset=page_offset)

    return page_elements, page_sizes


# ══════════════════════════════════════════════════════════════════════════════
# 6. VLM enrichment pass  (images only — equations handled by Docling)
# ══════════════════════════════════════════════════════════════════════════════

def _render_bbox_crop(
    page_path: Path, bbox: dict, page_size: dict
) -> Optional[str]:
    try:
        img        = Image.open(page_path)
        w_px, h_px = img.size
        pw = page_size.get("width")  or 595
        ph = page_size.get("height") or 842
        sx, sy = w_px / pw, h_px / ph
        pad    = 4

        pdf_top = max(bbox["y0"], bbox["y1"])
        pdf_bot = min(bbox["y0"], bbox["y1"])

        x0 = max(0,    int(bbox["x0"] * sx) - pad)
        x1 = min(w_px, int(bbox["x1"] * sx) + pad)
        y0 = max(0,    int((ph - pdf_top) * sy) - pad)
        y1 = min(h_px, int((ph - pdf_bot) * sy) + pad)

        if x1 <= x0 or y1 <= y0:
            return None
        return _image_to_b64(img.crop((x0, y0, x1, y1)))
    except Exception:
        return None


def enrich_elements_with_vlm(
    page_elements: dict[int, list[dict]],
    pages_dir: Path,
    page_sizes: dict[int, dict],
) -> None:
    work_items: list[tuple[int, int, str, str]] = []

    for page_no, elements in page_elements.items():
        page_img_path = pages_dir / f"page_{page_no}.png"
        psize         = page_sizes.get(page_no, {})

        for idx, elem in enumerate(elements):
            if elem.get("type") != "picture":
                continue

            b64: Optional[str] = None
            abs_path = elem.get("image_abs")
            if abs_path and Path(abs_path).exists():
                try:
                    b64 = _path_to_b64(Path(abs_path))
                except Exception:
                    pass
            if b64 is None and page_img_path.exists():
                b64 = _render_bbox_crop(page_img_path, elem["bbox"], psize)
            if b64:
                work_items.append((page_no, idx, "image", b64))

    if not work_items:
        print("      (no pictures to enrich)")
        return

    print(f"      → {len(work_items)} VLM tasks  (sequential, one at a time)")

    for done, (page_no, idx, _task_type, b64) in enumerate(work_items, 1):
        value: str = ""
        for attempt in range(3):
            try:
                value = describe_image(b64)
                if not value.startswith("ERROR:"):
                    break
                if attempt < 2:
                    print(f"      ⚠  VLM attempt {attempt+1} failed (p{page_no}): {value[:60]} — retrying…")
                    time.sleep(2 ** attempt)
            except Exception as exc:
                if attempt < 2:
                    print(f"      ⚠  VLM attempt {attempt+1} exception (p{page_no}): {exc} — retrying…")
                    time.sleep(2 ** attempt)
                else:
                    value = f"ERROR: {exc}"

        page_elements[page_no][idx]["description"] = value
        short = value[:70].replace("\n", " ") + ("…" if len(value) > 70 else "")
        status = "⚠ " if value.startswith("ERROR:") else f"[{done:>3}/{len(work_items)}]"
        print(f"      {status} p{page_no}: {short}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Save layout JSON
# ══════════════════════════════════════════════════════════════════════════════

def save_layout_json(
    page_elements: dict[int, list[dict]],
    page_sizes: dict[int, dict],
    out_dir: Path,
    pdf_name: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pages: list[dict] = []
    for page_no, elements in sorted(page_elements.items()):
        clean_elems = [
            {k: v for k, v in e.items() if k not in ("image_abs", "_needs_vlm")}
            for e in elements
        ]
        layout = {
            "pdf_name":      pdf_name,
            "page_no":       page_no,
            "page_image":    f"../pages/page_{page_no}.png",
            **page_sizes.get(page_no, {}),
            "element_count": len(clean_elems),
            "elements":      clean_elems,
        }
        p = out_dir / f"page_{page_no}_layout.json"
        p.write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
        all_pages.append(layout)

    all_pages_path = out_dir / "all_pages.json"
    all_pages_path.write_text(json.dumps(all_pages, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_pages_path


# ══════════════════════════════════════════════════════════════════════════════
# 8. Markdown export
# ══════════════════════════════════════════════════════════════════════════════

def _extract_fig_id(caption: str) -> str:
    m = re.match(r"(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?)", caption, re.IGNORECASE)
    return m.group(1) if m else ""


def _extract_image_type(desc: str) -> str:
    return desc.split("\n")[0].strip().rstrip(" —:-") if desc else ""


def _picture_md_block(elem: dict) -> list[str]:
    ref  = elem.get("image_ref", "")
    cap  = (elem.get("caption") or "").strip()
    desc = (elem.get("description") or "").strip()

    img_type = _extract_image_type(desc) if desc else (cap or "figure")
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

    claimed_captions = {
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
            if content not in claimed_captions:
                lines.append(f"\n*{content}*\n")
        elif t == "picture":
            lines.extend(_picture_md_block(elem))
        elif content:
            lines.append(f"\n{content}\n")

    return "\n".join(lines)


def export_full_markdown(
    page_elements: dict[int, list[dict]],
    pages_md_dir: Path,
    out_dir: Path,
    pdf_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for page_no in sorted(page_elements.keys()):
        p = pages_md_dir / f"page_{page_no}.md"
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    (out_dir / f"{pdf_name}.md").write_text("\n\n---\n\n".join(parts), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Chunking  (Docling HybridChunker / HierarchicalChunker)
# ══════════════════════════════════════════════════════════════════════════════

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


def _build_chunks(
    dl_doc: DoclingDocument,
    chunker_type: ChunkerType,
    tokenizer: str,
    max_tokens: int | None,
    merge_peers: bool,
    include_raw_text: bool,
) -> list[dict[str, Any]]:
    if chunker_type == ChunkerType.HYBRID:
        kwargs: dict[str, Any] = {"tokenizer": tokenizer, "merge_peers": merge_peers}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        chunker = HybridChunker(**kwargs)
    else:
        chunker = HierarchicalChunker()

    chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunker.chunk(dl_doc=dl_doc)):
        contextualized_text = chunker.contextualize(chunk=chunk)
        raw_text = getattr(chunk, "text", None)
        chunk_payload = _to_jsonable(chunk)
        chunk_payload.update({
            "chunk_index": index,
            "chunker":     chunker_type.value,
            "text":        contextualized_text,
        })
        if include_raw_text:
            chunk_payload["raw_text"] = raw_text
        _normalize_chunk_bboxes(chunk_payload)
        chunks.append(chunk_payload)

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 10. Figure chunks  (one chunk per picture, heading + caption + VLM desc)
# ══════════════════════════════════════════════════════════════════════════════

def build_figure_chunks(
    page_elements: dict[int, list[dict]],
) -> list[dict[str, Any]]:
    figure_chunks: list[dict[str, Any]] = []

    for page_no in sorted(page_elements.keys()):
        elements = page_elements[page_no]
        current_heading: str = ""

        for elem in elements:
            t = elem.get("type", "")

            if t in ("title", "section_header"):
                current_heading = (elem.get("content") or "").strip()

            elif t == "picture":
                desc = (elem.get("description") or "").strip()
                if not desc:
                    continue

                cap = (elem.get("caption") or "").strip()

                text_parts: list[str] = []
                if current_heading:
                    text_parts.append(current_heading)
                if cap:
                    text_parts.append(cap)
                text_parts.append(desc)

                figure_chunks.append({
                    "chunker":        "figure",
                    "type":           "figure",
                    "page":           page_no,
                    "parent_heading": current_heading,
                    "caption":        cap,
                    "image_ref":      elem.get("image_ref", ""),
                    "bbox":           elem.get("bbox"),
                    "text":           "\n".join(text_parts),
                })

    return figure_chunks


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_input: str,
    output_root: str = ".",
    skip_vlm: bool = False,
    chunker_type: ChunkerType = ChunkerType.HYBRID,
    tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int | None = None,
    merge_peers: bool = True,
    include_raw_text: bool = True,
    n_docling_workers: int = _DEFAULT_DOCLING_WORKERS,
) -> None:
    pdf_path = Path(pdf_input).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_name = pdf_path.stem
    out      = Path(output_root) / pdf_name

    pages_dir  = out / "pages"
    layout_dir = out / "layout"
    ocr_dir    = out / "pages_md"
    images_dir = out / "images"

    for d in (pages_dir, layout_dir, ocr_dir, images_dir):
        d.mkdir(parents=True, exist_ok=True)

    vlm_label = "DISABLED (--no-vlm)" if skip_vlm else OLLAMA_MODEL
    print(f"\n{'═'*62}")
    print(f"  PDF             →  {pdf_path.name}")
    print(f"  VLM             →  {vlm_label}")
    print(f"  Chunker         →  {chunker_type.value}")
    print(f"  Docling workers →  {n_docling_workers}")
    print(f"  VLM             →  sequential (1 at a time)")
    print(f"{'═'*62}")

    # ── Step 1: render page images ────────────────────────────────────────────
    print("\n[1/7] Rendering page images …")
    page_paths = render_pages(pdf_path, pages_dir, dpi=150)
    print(f"      ✓  {len(page_paths)} pages  →  {pages_dir}/")

    # ── Step 2: parallel Docling OCR ─────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="docling_split_"))
    try:
        chunks_info = _split_pdf(pdf_path, n_docling_workers, tmp_dir)
        actual_workers = len(chunks_info)

        print(f"\n[2/7] Running Docling in parallel ({actual_workers} workers) …")
        for i, (sub_path, offset) in enumerate(chunks_info):
            with fitz.open(sub_path) as f:
                n_pages = f.page_count
            print(f"      worker {i}: {n_pages} pages  (global offset +{offset})")

        worker_args = [
            (sub_path, offset, str(images_dir))
            for sub_path, offset in chunks_info
        ]

        page_elements: dict[int, list[dict]] = {}
        page_sizes:    dict[int, dict]       = {}

        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = {pool.submit(_docling_worker, a): a for a in worker_args}
            for fut in as_completed(futures):
                pe, ps = fut.result()
                page_elements.update(pe)
                page_sizes.update(ps)

        print(f"      ✓  Conversion complete  ({len(page_elements)} pages merged)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Step 3: build element map ─────────────────────────────────────────────
    print("\n[3/7] Sorting reading order …")
    sort_elements_reading_order(page_elements, page_sizes)
    total_elems = sum(len(v) for v in page_elements.values())
    print(f"      ✓  {total_elems} elements across {len(page_elements)} pages")

    # ── Step 4: VLM enrichment ────────────────────────────────────────────────
    if skip_vlm:
        print("\n[4/7] VLM enrichment … SKIPPED")
    else:
        print(f"\n[4/7] VLM enrichment via Ollama ({OLLAMA_MODEL}) …")
        enrich_elements_with_vlm(page_elements, pages_dir, page_sizes)
        print("      ✓  VLM enrichment complete")

    # ── Step 5: layout + markdown ─────────────────────────────────────────────
    print("\n[5/7] Saving layout + markdown outputs …")
    save_layout_json(page_elements, page_sizes, layout_dir, pdf_name)
    print(f"      ✓  Layout JSON   →  {layout_dir}/")

    for page_no, elements in sorted(page_elements.items()):
        md_txt = build_page_markdown(page_no, elements)
        (ocr_dir / f"page_{page_no}.md").write_text(md_txt, encoding="utf-8")
    print(f"      ✓  Page markdown →  {ocr_dir}/")

    export_full_markdown(page_elements, ocr_dir, out, pdf_name)
    print(f"      ✓  Full markdown →  {out}/{pdf_name}.md")

    # ── Step 6: chunking (fast full-doc pass, OCR disabled) ───────────────────
    print(f"\n[6/7] Fast Docling pass for chunking (OCR disabled, full document) …")
    fast_doc = run_docling_fast(pdf_path).document
    print("      ✓  Fast pass complete")

    print(f"      Chunking with {chunker_type.value} chunker …")
    chunks = _build_chunks(
        dl_doc=fast_doc,
        chunker_type=chunker_type,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        merge_peers=merge_peers,
        include_raw_text=include_raw_text,
    )

    figure_chunks = build_figure_chunks(page_elements)
    if figure_chunks:
        def _chunk_page(c: dict) -> int:
            try:
                return c["meta"]["doc_items"][0]["prov"][0]["page_no"]
            except (KeyError, IndexError, TypeError):
                return 0

        all_chunks: list[dict] = []
        reg_by_page: dict[int, list[dict]] = defaultdict(list)
        for c in chunks:
            reg_by_page[_chunk_page(c)].append(c)

        fig_by_page: dict[int, list[dict]] = defaultdict(list)
        for c in figure_chunks:
            fig_by_page[c["page"]].append(c)

        for page_no in sorted(set(list(reg_by_page) + list(fig_by_page))):
            all_chunks.extend(reg_by_page.get(page_no, []))
            all_chunks.extend(fig_by_page.get(page_no, []))

        for i, c in enumerate(all_chunks):
            c["chunk_index"] = i
        chunks = all_chunks

    print(f"      ✓  {len(chunks)} chunks ({len(figure_chunks)} figure)  →  output")

    chunk_meta: dict[str, Any] = {}
    if chunker_type == ChunkerType.HYBRID:
        chunk_meta = {"tokenizer": tokenizer, "max_tokens": max_tokens, "merge_peers": merge_peers}

    chunks_filename = f"{chunker_type.value}_chunks.json"
    chunks_path = out / chunks_filename
    chunks_payload = {
        "source":             str(pdf_path),
        "document_name":      pdf_name,
        "chunker":            chunker_type.value,
        "chunk_count":        len(chunks),
        "figure_chunk_count": len(figure_chunks),
        "vlm_enriched":       not skip_vlm,
        **chunk_meta,
        "chunks":             chunks,
    }
    chunks_path.write_text(
        json.dumps(_to_jsonable(chunks_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"      ✓  {len(chunks)} chunks  →  {chunks_path}")

    # ── Step 7: ingestion report ──────────────────────────────────────────────
    print("\n[7/7] Writing ingestion report …")
    _elapsed  = time.perf_counter() - _PIPELINE_START
    parsed_at = time.strftime("%Y%m%d_%H%M%S")

    type_counts: dict[str, int] = defaultdict(int)
    for elems in page_elements.values():
        for e in elems:
            type_counts[e.get("type", "other")] += 1

    report = {
        "pdf":            str(pdf_path),
        "parsed_at":      parsed_at,
        "strategy":       "parallel_hi_res",
        "vlm_model":      "disabled" if skip_vlm else OLLAMA_MODEL,
        "total_elements": sum(type_counts.values()),
        "total_pages":    max(page_elements.keys(), default=0),
        "element_counts": dict(type_counts),
        "chunker":        chunker_type.value,
        "chunk_count":    len(chunks),
        "runtime": {
            "elapsed_seconds":   round(_elapsed, 2),
            "cpus_pinned":       _PINNED_CPUS,
            "total_cpus":        _TOTAL_CPUS,
            "docling_workers":   actual_workers,
            "vlm_mode":          "sequential",
        },
        "outputs": {
            "layout_json":  str(layout_dir / "all_pages.json"),
            "layout_dir":   str(layout_dir),
            "pages_dir":    str(pages_dir),
            "pages_md_dir": str(ocr_dir),
            "full_md":      str(out / (pdf_name + ".md")),
            "images_dir":   str(images_dir),
            "chunks_json":  str(chunks_path),
        },
    }

    report_path = out / "ingestion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'═'*62}")
    print("  Done!")
    print(f"    pages/       {pages_dir}")
    print(f"    layout/      {layout_dir}")
    print(f"    pages_md/    {ocr_dir}")
    print(f"    full MD      {out / (pdf_name + '.md')}")
    print(f"    images/      {images_dir}")
    print(f"    chunks       {chunks_path}  ({len(chunks)} chunks)")
    print(f"    report       {report_path}")
    print(f"    time         {_elapsed:.2f}s  ({_PINNED_CPUS}/{_TOTAL_CPUS} CPUs, {actual_workers} Docling workers)")
    print(f"{'═'*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF → pages · layout · markdown · images · chunks  (parallel Docling + Ollama VLM)"
    )
    parser.add_argument("pdf",        help="Path to input PDF")
    parser.add_argument(
        "output_root",
        nargs="?",
        default=".",
        help="Output root directory (default: .)",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip Ollama VLM step",
    )
    parser.add_argument(
        "--docling-workers",
        type=int,
        default=_DEFAULT_DOCLING_WORKERS,
        help=f"Number of parallel Docling processes (default: {_DEFAULT_DOCLING_WORKERS})",
    )
    parser.add_argument(
        "--chunker",
        choices=[c.value for c in ChunkerType],
        default=ChunkerType.HYBRID.value,
        help="Chunking strategy: hybrid (default) or hierarchical",
    )
    parser.add_argument(
        "--tokenizer",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Tokenizer model name used by the hybrid chunker",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per chunk (hybrid chunker)",
    )
    parser.add_argument(
        "--merge-peers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge undersized neighboring chunks (hybrid chunker)",
    )
    parser.add_argument(
        "--include-raw-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include raw (non-contextualized) text alongside chunk text",
    )

    args = parser.parse_args()
    process_pdf(
        pdf_input=args.pdf,
        output_root=args.output_root,
        skip_vlm=args.no_vlm,
        chunker_type=ChunkerType(args.chunker),
        tokenizer=args.tokenizer,
        max_tokens=args.max_tokens,
        merge_peers=args.merge_peers,
        include_raw_text=args.include_raw_text,
        n_docling_workers=args.docling_workers,
    )
