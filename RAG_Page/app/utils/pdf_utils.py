"""
pdf_utils.py — PyMuPDF helpers: page count and page-to-PNG rendering.
"""

from __future__ import annotations

from pathlib import Path
from app.logger import get_logger

log = get_logger(__name__)


def get_page_count(pdf_path: Path) -> int:
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        n = doc.page_count
    log.debug("PDF '%s' has %d pages.", pdf_path.name, n)
    return n


def render_pages(pdf_path: Path, out_dir: Path, dpi: int = 150) -> dict[int, Path]:
    """
    Render all pages of a PDF to PNG files.
    Returns {page_no (1-based): Path}.
    Skips pages that already exist on disk (cache hit).
    """
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    mat   = fitz.Matrix(scale, scale)
    paths: dict[int, Path] = {}

    with fitz.open(str(pdf_path)) as pdf:
        total = pdf.page_count
        log.info("Rendering %d pages of '%s' at %d DPI …", total, pdf_path.name, dpi)
        for i, page in enumerate(pdf, start=1):
            p = out_dir / f"page_{i}.png"
            if p.exists():
                log.debug("  page %d — cache hit, skipping render.", i)
            else:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pix.save(str(p))
                log.debug("  page %d rendered → %s", i, p.name)
            paths[i] = p

    log.info("Page rendering complete: %d pages in %s", len(paths), out_dir)
    return paths


def render_single_page(pdf_path: Path, page_no: int, out_dir: Path, dpi: int = 150) -> Path:
    """Render a single page (1-based). Returns the PNG path."""
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"page_{page_no}.png"
    if p.exists():
        log.debug("Single page %d — cache hit.", page_no)
        return p

    scale = dpi / 72.0
    mat   = fitz.Matrix(scale, scale)
    with fitz.open(str(pdf_path)) as pdf:
        page = pdf[page_no - 1]
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(p))
    log.debug("Rendered single page %d → %s", page_no, p)
    return p
