"""
coord_utils.py — PDF bbox coordinate conversion.
PDF origin: bottom-left. Image origin: top-left.
"""

from __future__ import annotations

from app.logger import get_logger

log = get_logger(__name__)


def pdf_to_img_coords(
    x0: float, y0: float, x1: float, y1: float,
    pdf_w: float, pdf_h: float,
    img_w: int,  img_h: int,
    padding: int = 4,
) -> tuple[int, int, int, int]:
    """
    Convert PDF coordinate bbox to image pixel bbox.

    PDF coords: x increases right, y increases UP (origin bottom-left).
    Image coords: x increases right, y increases DOWN (origin top-left).

    Returns (left, top, right, bottom) in pixel space, clamped to image bounds.
    """
    sx = img_w / pdf_w
    sy = img_h / pdf_h

    left  = int(x0 * sx) - padding
    right = int(x1 * sx) + padding

    # Flip Y axis
    top = int((pdf_h - max(y0, y1)) * sy) - padding
    bot = int((pdf_h - min(y0, y1)) * sy) + padding

    # Clamp to image bounds
    left  = max(0, left)
    top   = max(0, top)
    right = min(img_w, right)
    bot   = min(img_h, bot)

    if right <= left or bot <= top:
        log.warning("Degenerate bbox after conversion: pdf(%s,%s,%s,%s) → img(%s,%s,%s,%s)",
                    x0, y0, x1, y1, left, top, right, bot)

    return left, top, right, bot


def normalize_bbox(b: dict | None) -> dict | None:
    """Normalize any bbox dict to {x0, y0, x1, y1} float keys."""
    if not b or not isinstance(b, dict):
        return None
    if "x0" in b:
        return {k: round(float(b[k]), 2) for k in ("x0", "y0", "x1", "y1")}
    if "l" in b:
        return {
            "x0": round(float(b["l"]), 2),
            "y0": round(float(b["t"]), 2),
            "x1": round(float(b["r"]), 2),
            "y1": round(float(b["b"]), 2),
        }
    return None
