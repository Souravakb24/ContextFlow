"""
proof.py — Draw retrieved chunk bboxes onto page images.
Returns annotated PIL Images for display in the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.config import cfg
from app.logger import get_logger
from app.utils.coord_utils import pdf_to_img_coords

log = get_logger(__name__)

# Colour palette for up to 5 chunks
_CHUNK_COLORS = [
    "#6C3BAF",  # dark purple (primary)
    "#E86C2E",  # orange
    "#2E86C1",  # blue
    "#27AE60",  # green
    "#C0392B",  # red
]

_SCORE_COLORS = {
    "high":   "#27AE60",
    "medium": "#E67E22",
    "low":    "#E74C3C",
}


def _score_color(score: float) -> str:
    if score >= 0.75:
        return _SCORE_COLORS["high"]
    if score >= 0.5:
        return _SCORE_COLORS["medium"]
    return _SCORE_COLORS["low"]


def _get_page_dims(layout_json_path: str | None) -> tuple[float, float]:
    if layout_json_path and Path(layout_json_path).exists():
        try:
            import json
            d = json.loads(Path(layout_json_path).read_text(encoding="utf-8"))
            return float(d.get("width", 595)), float(d.get("height", 842))
        except Exception:
            pass
    return 595.0, 842.0


def draw_bboxes_on_page(
    page_img_path: str,
    bboxes:        list[dict],
    pdf_w: float = 595.0,
    pdf_h: float = 842.0,
    chunk_colors: list[str] | None = None,
    line_width: int = 3,
) -> Image.Image:
    """
    Draw coloured bboxes on a page image.
    bboxes: list of {x0, y0, x1, y1, label (optional), color_idx (optional)}
    Returns annotated PIL Image.
    """
    img  = Image.open(page_img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    colors = chunk_colors or _CHUNK_COLORS

    for i, bbox in enumerate(bboxes):
        color = colors[bbox.get("color_idx", i) % len(colors)]
        x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
        l, t, r, b = pdf_to_img_coords(x0, y0, x1, y1, pdf_w, pdf_h, w, h, padding=3)

        if r <= l or b <= t:
            log.warning("Degenerate bbox skipped: %s", bbox)
            continue

        # Semi-transparent fill
        rgba = _hex_to_rgba(color, alpha=30)
        draw.rectangle([l, t, r, b], fill=rgba, outline=color, width=line_width)

        # Label tag
        label = bbox.get("label", "")
        if label:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            tag_w = len(label) * 7 + 8
            tag_h = 16
            tx, ty = l, max(0, t - tag_h)
            draw.rectangle([tx, ty, tx + tag_w, ty + tag_h], fill=color)
            draw.text((tx + 4, ty + 1), label, fill="white", font=font)

    return img


def _hex_to_rgba(hex_color: str, alpha: int = 80) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


def annotate_hits(
    hits: list[dict],
    top_k: int | None = None,
) -> dict[str, list[dict]]:
    """
    For each unique (document, page) combination in hits, produce annotated page images.

    Returns:
        {
          "doc::page_no": {
            "image": PIL.Image,
            "page_no": int,
            "document_name": str,
            "chunks": [list of chunk summaries on this page],
          }
        }
    """
    hits = hits[:top_k] if top_k else hits

    # Group provenance by (doc, page_no)
    page_data: dict[str, dict] = {}

    for chunk_idx, hit in enumerate(hits):
        doc_name   = hit.get("document_name", "")
        provenance = hit.get("provenance_parsed", [])
        files      = hit.get("files", {})
        score      = hit.get("reranker_score", hit.get("score", 0.0))

        img_by_page: dict[int, str] = {}
        for p in files.get("page_images", []):
            try:
                pno = int(Path(p).stem.split("_")[-1])
                img_by_page[pno] = p
            except ValueError:
                pass

        layout_by_page: dict[int, str] = {}
        for lj in files.get("layout_json", []):
            try:
                parts = Path(lj).stem.split("_")
                pno   = int(parts[1])
                layout_by_page[pno] = lj
            except (ValueError, IndexError):
                pass

        for prov in provenance:
            pno = prov.get("page_no")
            if pno is None:
                continue
            key = f"{doc_name}::{pno}"
            if key not in page_data:
                page_data[key] = {
                    "document_name": doc_name,
                    "page_no":       pno,
                    "img_path":      img_by_page.get(pno, ""),
                    "layout_path":   layout_by_page.get(pno, ""),
                    "bboxes":        [],
                    "chunks":        [],
                }

            page_data[key]["bboxes"].append({
                "x0": prov.get("x0", 0),
                "y0": prov.get("y0", 0),
                "x1": prov.get("x1", 0),
                "y1": prov.get("y1", 0),
                "label": f"[{chunk_idx + 1}]",
                "color_idx": chunk_idx,
            })
            page_data[key]["chunks"].append({
                "chunk_idx":      chunk_idx + 1,
                "chunk_type":     hit.get("chunk_type", ""),
                "headings":       hit.get("headings", ""),
                "reranker_score": score,
                "text_preview":   (hit.get("raw_text") or hit.get("text", ""))[:120],
            })

    # Render each page
    results: dict[str, dict] = {}
    for key, data in page_data.items():
        img_path = data["img_path"]
        if not img_path or not Path(img_path).exists():
            log.warning("Page image not found for %s page %d — skipping.", data["document_name"], data["page_no"])
            continue

        pdf_w, pdf_h = _get_page_dims(data.get("layout_path"))
        try:
            annotated = draw_bboxes_on_page(img_path, data["bboxes"], pdf_w, pdf_h)
            results[key] = {
                "image":         annotated,
                "page_no":       data["page_no"],
                "document_name": data["document_name"],
                "chunks":        data["chunks"],
            }
            log.info("Annotated page %d of '%s' with %d bbox(es).",
                     data["page_no"], data["document_name"], len(data["bboxes"]))
        except Exception as e:
            log.error("Failed to annotate page %d of '%s': %s", data["page_no"], data["document_name"], e)

    return results


def annotate_single_hit(hit: dict, chunk_idx: int = 0) -> list[Image.Image]:
    """
    Annotate all pages referenced by a single hit.
    Returns list of annotated PIL Images.
    """
    result = annotate_hits([hit], top_k=1)
    return [v["image"] for v in result.values()]
