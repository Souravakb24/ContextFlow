#!/usr/bin/env python3
"""
PDF Processing Pipeline  ─  Docling + PyMuPDF + Ollama VLM
═══════════════════════════════════════════════════════════
Outputs (all under <output_root>/<pdf_name>/):

  pages/page_<N>.png            ← full-page screenshot (150 DPI)
  layout/page_<N>_layout.json   ← per-page structured layout + VLM descriptions
  pages_md/page_<N>.md          ← per-page markdown w/ image refs + descriptions
  <pdf_name>.md                 ← complete document markdown
  images/                       ← extracted embedded images + figure crops

Handles: text, headings, tables (Markdown + raw data), equations (LaTeX via VLM),
         code blocks, lists, figures/captions w/ AI descriptions.

Install:
    pip install pymupdf docling psutil requests pillow
    # And have Ollama running with: ollama pull qwen3-vl:30b
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import psutil
import requests
import fitz  # PyMuPDF
from PIL import Image

# ── Docling imports ────────────────────────────────────────────────────────────
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types.doc import (
    DocItemLabel,
    ImageRefMode,
    TableItem,
    PictureItem,
)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_MODEL    = "qwen3-vl:30b"
OLLAMA_URL      = "http://localhost:11434/v1/chat/completions"
SAMPLE_INTERVAL = 0.5

CPU_TARGET      = 0.70
MAX_VLM_THREADS = 8
NUM_WORKERS     = max(1, int((os.cpu_count() or 1) * CPU_TARGET))

# Formula-image rendering DPI (higher = clearer but larger payload)
FORMULA_RENDER_DPI = 200


# ── Prompts ───────────────────────────────────────────────────────────────────
IMAGE_DESCRIPTION_PROMPT = """\
Analyze this image carefully and respond according to its type:

1. LOGO — If the image is a logo or brand mark, reply:
   "Logo: <brief description of the logo/brand name if visible>"

2. FLOWCHART / FLOW DIAGRAM — If the image contains boxes, arrows, or \
flow-based connections, describe it using a functional-dependency structure:
   - List every block/node as a bullet in a set of "Blocks:"
   - Then list every connection as: BlockA -> BlockB (label if any)
   - Give a high-level summary of the overall process in 1 paragraph.

3. PLOT / CHART / GRAPH — If the image is a data plot:
   - Describe chart type, axes labels, title, and overall trend.
   - Provide a markdown table summarising the key data points.

4. OTHER IMAGE — Write a thorough descriptive paragraph covering: subject,
   key visual elements, colours, spatial layout, and any visible text/numbers.

Output format must be strictly followed. Always start with the image type.
Respond only with the description — no preamble, no meta-commentary.
"""

FORMULA_EXTRACTION_PROMPT = """\
This image contains a mathematical formula or equation. Your task is to \
transcribe it into valid LaTeX.

Rules:
- Output ONLY the LaTeX code for the formula, nothing else.
- Do NOT wrap the output in $ or $$ or code fences.
- Do NOT add any explanation, preamble, or commentary.
- Use standard LaTeX math commands (\\frac, \\sum, \\int, \\sqrt, etc.).
- If the image contains multiple numbered equations, separate them with \\\\.
- If the formula is unreadable, respond with exactly: UNREADABLE
"""


# ══════════════════════════════════════════════════════════════════════════════
# CPU throttle guard
# ══════════════════════════════════════════════════════════════════════════════

class CPUGuard:
    """Blocks new VLM requests when system CPU is above CPU_TARGET."""
    def __init__(self, target: float = CPU_TARGET, interval: float = SAMPLE_INTERVAL):
        self._target   = target
        self._interval = interval

    def wait_if_busy(self) -> None:
        while True:
            usage = psutil.cpu_percent(interval=self._interval) / 100.0
            if usage < self._target:
                return
            time.sleep(self._interval)


_cpu_guard = CPUGuard()


# ══════════════════════════════════════════════════════════════════════════════
# Ollama VLM helpers
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
    """Call Ollama's OpenAI-compatible vision endpoint. Returns text response."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "stream": False,
    }

    for attempt in range(retries + 1):
        _cpu_guard.wait_if_busy()
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            return "ERROR: Ollama not reachable — is it running on localhost:11434?"
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

def render_pages(pdf_path: Path, out_dir: Path, dpi: int = 150) -> dict[int, Path]:
    """Render every PDF page to a PNG file at <dpi> resolution."""
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
# 2. Extract embedded raster images with PyMuPDF
# ══════════════════════════════════════════════════════════════════════════════

def extract_embedded_images(
    pdf_path: Path, out_dir: Path
) -> dict[int, list[Path]]:
    """Pull every raster image out of the PDF (de-duped by xref)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, list[Path]] = defaultdict(list)
    seen: set[int] = set()

    with fitz.open(str(pdf_path)) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    base = pdf.extract_image(xref)
                    ext  = base["ext"]
                    idx  = len(result[page_no]) + 1
                    p    = out_dir / f"page_{page_no}_img_{idx}.{ext}"
                    p.write_bytes(base["image"])
                    result[page_no].append(p)
                except Exception as exc:
                    print(f"  ⚠ xref={xref} page={page_no}: {exc}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. Run Docling
# ══════════════════════════════════════════════════════════════════════════════

def run_docling(pdf_path: Path):
    """Convert PDF with all Docling capabilities enabled."""
    opts = PdfPipelineOptions()
    opts.do_ocr                                    = True
    opts.do_table_structure                        = True
    opts.table_structure_options.do_cell_matching  = True
    opts.table_structure_options.mode              = TableFormerMode.ACCURATE
    opts.generate_page_images                      = False
    opts.generate_picture_images                   = True
    opts.images_scale                              = 2.0

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return converter.convert(str(pdf_path))


# ══════════════════════════════════════════════════════════════════════════════
# 4. Build per-page element map
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_dict(bbox) -> dict:
    return {
        "x0": round(float(bbox.l), 2),
        "y0": round(float(bbox.t), 2),
        "x1": round(float(bbox.r), 2),
        "y1": round(float(bbox.b), 2),
    }


def build_page_element_map(
    doc,
    images_dir: Path,
) -> dict[int, list[dict]]:
    """
    Iterate Docling document items → structured element list per page
    in Docling reading order. Picture crops are saved to <images_dir>.
    """
    page_elements: dict[int, list[dict]] = defaultdict(list)
    pic_counters:  dict[int, int]        = defaultdict(int)

    for item, level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue

        for prov in item.prov:
            page_no = prov.page_no
            label   = item.label

            elem: dict[str, Any] = {
                "type":  label.value if hasattr(label, "value") else str(label),
                "bbox":  _bbox_dict(prov.bbox),
                "level": level,
            }

            if hasattr(item, "text") and item.text:
                elem["content"] = item.text

            # ── Tables ──────────────────────────────────────────────────────
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

            # ── Pictures / figures ───────────────────────────────────────────
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

                # description filled in later by VLM
                elem["description"] = ""

            # ── Equations ────────────────────────────────────────────────────
            elif label == DocItemLabel.FORMULA:
                elem["type"] = "equation"
                if hasattr(item, "text") and item.text:
                    elem["latex"] = item.text
                # latex_vlm filled in later by VLM
                elem["latex_vlm"] = ""

            page_elements[page_no].append(elem)

    return page_elements


# ══════════════════════════════════════════════════════════════════════════════
# 5. Collect page sizes
# ══════════════════════════════════════════════════════════════════════════════

def get_page_sizes(doc) -> dict[int, dict]:
    sizes: dict[int, dict] = {}
    if not hasattr(doc, "pages"):
        return sizes
    for key, page in doc.pages.items():
        try:
            page_no = int(key)
        except (TypeError, ValueError):
            page_no = getattr(page, "page_no", None)
        if page_no and hasattr(page, "size") and page.size:
            sizes[page_no] = {
                "width":  round(float(page.size.width),  2),
                "height": round(float(page.size.height), 2),
            }
    return sizes


# ══════════════════════════════════════════════════════════════════════════════
# 6. VLM enrichment pass  (parallel, CPU-throttled)
# ══════════════════════════════════════════════════════════════════════════════

def _render_bbox_crop(
    page_path: Path, bbox: dict, page_size: dict
) -> Optional[str]:
    """Crop a bbox region from the rendered page PNG → base64."""
    try:
        img      = Image.open(page_path)
        w_px, h_px = img.size
        pw = page_size.get("width")  or 595
        ph = page_size.get("height") or 842
        sx, sy   = w_px / pw, h_px / ph
        pad      = 4
        x0 = max(0,    int(bbox["x0"] * sx) - pad)
        y0 = max(0,    int(bbox["y0"] * sy) - pad)
        x1 = min(w_px, int(bbox["x1"] * sx) + pad)
        y1 = min(h_px, int(bbox["y1"] * sy) + pad)
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
    """
    For every picture and equation element, call Ollama in parallel and
    attach 'description' (pictures) or 'latex_vlm' (equations) in-place.
    """
    work_items: list[tuple[int, int, str, str]] = []

    for page_no, elements in page_elements.items():
        page_img_path = pages_dir / f"page_{page_no}.png"
        psize         = page_sizes.get(page_no, {})

        for idx, elem in enumerate(elements):
            t   = elem.get("type", "")
            b64: Optional[str] = None

            if t == "picture":
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

            elif t == "equation":
                if page_img_path.exists():
                    b64 = _render_bbox_crop(page_img_path, elem["bbox"], psize)
                    if b64:
                        work_items.append((page_no, idx, "formula", b64))

    if not work_items:
        print("      (no pictures or equations to enrich)")
        return

    print(f"      → {len(work_items)} VLM tasks  "
          f"({MAX_VLM_THREADS} threads, CPU target {int(CPU_TARGET*100)}%)")

    def _process(item):
        page_no, idx, task_type, b64 = item
        if task_type == "image":
            return (page_no, idx, "description", describe_image(b64))
        else:
            return (page_no, idx, "latex_vlm",   extract_formula(b64))

    n_threads = min(MAX_VLM_THREADS, NUM_WORKERS, len(work_items))
    done = 0
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {pool.submit(_process, item): item for item in work_items}
        for fut in as_completed(futures):
            done += 1
            try:
                page_no, idx, field, value = fut.result()
                page_elements[page_no][idx][field] = value
                label = page_elements[page_no][idx].get("type", "")
                short = value[:70].replace("\n", " ") + ("…" if len(value) > 70 else "")
                print(f"      [{done:>3}/{len(work_items)}] p{page_no} {label}: {short}")
            except Exception as exc:
                print(f"      ⚠  VLM task failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Save layout JSON (one file per page)
# ══════════════════════════════════════════════════════════════════════════════

def save_layout_json(
    page_elements: dict[int, list[dict]],
    page_sizes: dict[int, dict],
    out_dir: Path,
    pdf_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pages: list[dict] = []
    for page_no, elements in sorted(page_elements.items()):
        # Sanitise: remove internal abs path (not useful in JSON output)
        clean_elems = []
        for e in elements:
            ce = {k: v for k, v in e.items() if k != "image_abs"}
            clean_elems.append(ce)

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


# ══════════════════════════════════════════════════════════════════════════════
# 8. BBox visualization  (reuses drawing helpers from visualize_layout.py)
# ══════════════════════════════════════════════════════════════════════════════

# Colour map mirrors visualize_layout.py
_VIS_COLORS: dict[str, tuple[int, int, int]] = {
    "text":                (70,  130, 180),
    "section_header":      (255, 140,   0),
    "title":               (220,  20,  60),
    "list_item":           (60,  179, 113),
    "caption":             (147, 112, 219),
    "footnote":            (188, 143, 143),
    "page_header":         (128, 128,   0),
    "page_footer":         (128, 128,   0),
    "table":               (255,  69,   0),
    "document_index":      (255,  99,  71),
    "picture":             (30,  144, 255),
    "formula":             (255,  20, 147),
    "equation":            (255,  20, 147),
    "code":                (0,   206, 209),
    "checkbox_selected":   (50,  205,  50),
    "checkbox_unselected": (169, 169, 169),
    "form":                (255, 215,   0),
    "key_value_region":    (255, 165,   0),
    "paragraph":           (100, 149, 237),
}
_VIS_FALLBACK = (128, 128, 128)


def _vis_color(label: str) -> tuple[int, int, int]:
    return _VIS_COLORS.get(label.lower(), _VIS_FALLBACK)


def _draw_bboxes_on_page(
    page_img_path: Path,
    elements: list[dict],
    page_size: dict,
) -> "Image.Image":
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(page_img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = img.size
    pw = page_size.get("width") or 595
    ph = page_size.get("height") or 842
    sx, sy = img_w / pw, img_h / ph

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except OSError:
            font = ImageFont.load_default()

    for elem in elements:
        label = elem.get("type", "unknown")
        color = _vis_color(label)
        bbox  = elem.get("bbox", {})

        x0 = bbox.get("x0", 0) * sx
        y0 = bbox.get("y0", 0) * sy
        x1 = bbox.get("x1", 0) * sx
        y1 = bbox.get("y1", 0) * sy

        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        draw.rectangle([(x0, y0), (x1, y1)], fill=(*color, 45), outline=(*color, 230), width=2)

        text_y = y0 - 16 if y0 > 18 else y0 + 2
        text_bbox = draw.textbbox((x0, text_y), label, font=font)
        draw.rectangle(
            [(text_bbox[0] - 2, text_bbox[1] - 1), (text_bbox[2] + 2, text_bbox[3] + 1)],
            fill=(*color, 210),
        )
        draw.text((x0, text_y), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


def _build_legend(label_counts: dict[str, int]) -> "Image.Image":
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    row_h, swatch, pad, width = 26, 18, 10, 320
    height = pad + row_h * max(len(label_counts), 1) + pad
    img  = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    y = pad
    for lbl, cnt in sorted(label_counts.items()):
        color = _vis_color(lbl)
        draw.rectangle([(pad, y), (pad + swatch, y + swatch - 2)], fill=color, outline=(80, 80, 80))
        draw.text((pad + swatch + 8, y), f"{lbl}  ×{cnt}", font=font, fill=(30, 30, 30))
        y += row_h

    return img


def save_bbox_visualizations(
    page_elements: dict[int, list[dict]],
    pages_dir: Path,
    page_sizes: dict[int, dict],
    vis_dir: Path,
) -> None:
    from PIL import Image

    vis_dir.mkdir(parents=True, exist_ok=True)

    for page_no, elements in sorted(page_elements.items()):
        page_img_path = pages_dir / f"page_{page_no}.png"
        if not page_img_path.exists():
            print(f"  [vis page {page_no}] page image not found, skipping.")
            continue

        psize = page_sizes.get(page_no, {})
        annotated = _draw_bboxes_on_page(page_img_path, elements, psize)

        label_counts: dict[str, int] = {}
        for e in elements:
            lbl = e.get("type", "unknown")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        legend = _build_legend(label_counts)
        combined_h = max(annotated.height, legend.height)
        combined = Image.new("RGB", (annotated.width + legend.width + 10, combined_h), (255, 255, 255))
        combined.paste(annotated, (0, 0))
        combined.paste(legend, (annotated.width + 10, 0))

        out_path = vis_dir / f"page_{page_no:03d}_bbox.png"
        combined.save(out_path)
        print(f"      [page {page_no}] {sum(label_counts.values())} elements → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Export full document Markdown
# ══════════════════════════════════════════════════════════════════════════════

def export_full_markdown(
    doc,
    page_elements: dict[int, list[dict]],
    out_dir: Path,
    pdf_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        md = doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
    except Exception:
        md = doc.export_to_markdown()

    # Collect pictures in document order to replace <!-- image --> placeholders
    pictures = [
        elem
        for page_no in sorted(page_elements.keys())
        for elem in page_elements[page_no]
        if elem.get("type") == "picture"
    ]

    pic_iter = iter(pictures)
    parts = md.split("<!-- image -->")
    result_parts = [parts[0]]
    for part in parts[1:]:
        try:
            pic   = next(pic_iter)
            block = "\n".join(_picture_md_block(pic, img_prefix=""))
            result_parts.append(block + part)
        except StopIteration:
            result_parts.append("<!-- image -->" + part)

    (out_dir / f"{pdf_name}.md").write_text("".join(result_parts), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 11. Per-page Markdown  (with VLM descriptions inline)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_fig_id(caption: str) -> str:
    """Pull 'Fig. 1' / 'Figure 2a' from the start of a caption string."""
    import re
    m = re.match(r"(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?)", caption, re.IGNORECASE)
    return m.group(1) if m else ""


def _extract_image_type(desc: str) -> str:
    """Return the image-type label (first line of VLM description)."""
    return desc.split("\n")[0].strip().rstrip(" —:-") if desc else ""


def _picture_md_block(elem: dict, img_prefix: str) -> list[str]:
    """
    Render a picture element as Markdown lines.
    img_prefix  — prepended to image_ref  ('../' for pages_md, '' for full MD)
    """
    ref  = elem.get("image_ref", "")
    cap  = (elem.get("caption") or "").strip()
    desc = (elem.get("description") or "").strip()

    img_type = _extract_image_type(desc) if desc else (cap or "figure")
    fig_id   = _extract_fig_id(cap)

    lines: list[str] = []
    if ref:
        lines.append(f"\n![{img_type}]({img_prefix}{ref})\n")
    else:
        lines.append("\n<!-- figure: image not extracted -->\n")

    if fig_id:
        lines.append(f"\n**Figure:** {fig_id}")
    if cap:
        lines.append(f"**Caption:** {cap}")
    if desc:
        lines.append(f"**Description:**\n{desc}")
    lines.append("")          # blank line after block

    return lines


def build_page_markdown(
    page_no: int,
    elements: list[dict],
    embedded_images: list[Path],
) -> str:
    lines: list[str] = [f"# Page {page_no}\n"]

    # Captions already claimed by picture elements — skip standalone rendering
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
            # Prefer VLM extraction (more accurate on rendered crops)
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
            # Skip if this caption belongs to a picture element (rendered inline there)
            if content not in claimed_captions:
                lines.append(f"\n*{content}*\n")

        elif t == "picture":
            lines.extend(_picture_md_block(elem, img_prefix="../"))

        elif content:
            lines.append(f"\n{content}\n")

    if embedded_images:
        lines.append("\n---\n## Embedded Images\n")
        for p in embedded_images:
            lines.append(f"\n![{p.name}](../images/{p.name})\n")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_input: str,
    output_root: str = ".",
    skip_vlm: bool = False,
) -> None:
    pdf_path = Path(pdf_input).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_name = pdf_path.stem
    out      = Path(output_root) / pdf_name

    pages_dir   = out / "pages"
    layout_dir  = out / "layout"
    ocr_dir     = out / "pages_md"
    full_dir    = out
    images_dir  = out / "images"
    bbox_vis_dir = out / "bbox_vis"

    for d in (pages_dir, layout_dir, ocr_dir, images_dir, bbox_vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    vlm_label = f"DISABLED (--no-vlm)" if skip_vlm else OLLAMA_MODEL
    print(f"\n{'═'*62}")
    print(f"  PDF      →  {pdf_path.name}")
    print(f"  VLM      →  {vlm_label}")
    print(f"  Workers  →  {NUM_WORKERS} CPU / {MAX_VLM_THREADS} VLM threads")
    print(f"{'═'*62}")

    print("\n[1/7] Rendering page images …")
    page_paths = render_pages(pdf_path, pages_dir, dpi=150)
    print(f"      ✓  {len(page_paths)} pages  →  {pages_dir}/")

    print("\n[2/6] Extracting embedded images …")
    embedded_images = extract_embedded_images(pdf_path, images_dir)
    total = sum(len(v) for v in embedded_images.values())
    print(f"      ✓  {total} images  →  {images_dir}/")

    print("\n[3/6] Running Docling (OCR · layout · tables · equations) …")
    result = run_docling(pdf_path)
    doc    = result.document
    print("      ✓  Conversion complete")

    print("\n[4/6] Building element map …")
    page_elements = build_page_element_map(doc, images_dir)
    page_sizes    = get_page_sizes(doc)
    total_elems   = sum(len(v) for v in page_elements.values())
    print(f"      ✓  {total_elems} elements across {len(page_elements)} pages")

    if skip_vlm:
        print("\n[5/6] VLM enrichment … SKIPPED")
    else:
        print(f"\n[5/6] VLM enrichment via Ollama ({OLLAMA_MODEL}) …")
        enrich_elements_with_vlm(page_elements, pages_dir, page_sizes)
        print("      ✓  VLM enrichment complete")

    print("\n[6/6] Saving outputs …")
    save_layout_json(page_elements, page_sizes, layout_dir, pdf_name)
    print(f"      ✓  Layout JSON   →  {layout_dir}/")

    export_full_markdown(doc, page_elements, full_dir, pdf_name)
    print(f"      ✓  Full markdown →  {full_dir}/{pdf_name}.md")

    ocr_dir.mkdir(parents=True, exist_ok=True)
    for page_no, elements in sorted(page_elements.items()):
        imgs   = embedded_images.get(page_no, [])
        md_txt = build_page_markdown(page_no, elements, imgs)
        (ocr_dir / f"page_{page_no}.md").write_text(md_txt, encoding="utf-8")
    print(f"      ✓  Page markdown →  {ocr_dir}/")

    print(f"\n{'═'*62}")
    print("  Done!")
    print(f"    pages/     {pages_dir}")
    print(f"    layout/    {layout_dir}")
    print(f"    pages_md/  {ocr_dir}")
    print(f"    full MD    {full_dir / (pdf_name + '.md')}")
    print(f"    images/    {images_dir}")
    print(f"{'═'*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF → pages · layout · markdown · images  (+Ollama VLM)"
    )
    parser.add_argument("pdf",         help="Path to input PDF")
    parser.add_argument("output_root", nargs="?", default=".", help="Output root dir (default: .)")
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip Ollama VLM step (useful when Ollama is not running)",
    )
    args = parser.parse_args()
    process_pdf(args.pdf, args.output_root, skip_vlm=args.no_vlm)


    #python injestion_docling.py two_sides.pdf ./output
