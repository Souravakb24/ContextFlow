"""
file_utils.py — Path scaffolding, JSON I/O, and output directory helpers.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def doc_output_dir(docs_dir: Path, pdf_name: str) -> Path:
    """Return the output directory for a document, creating subdirs."""
    base = docs_dir / pdf_name
    for sub in ("pages", "layout", "pages_md", "images"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def load_json(path: Path) -> Any:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        log.debug("Loaded JSON: %s", path)
        return data
    except Exception as e:
        log.error("Failed to load JSON from %s: %s", path, e)
        raise


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    try:
        path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
        log.debug("Saved JSON: %s", path)
    except Exception as e:
        log.error("Failed to save JSON to %s: %s", path, e)
        raise


def pdf_fingerprint(pdf_path: Path) -> str:
    """SHA-256 fingerprint of first 64 KB + file size — fast cache key."""
    h = hashlib.sha256()
    h.update(str(pdf_path.stat().st_size).encode())
    with open(pdf_path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def is_already_processed(pdf_path: Path, docs_dir: Path) -> bool:
    """
    Cache hit check: returns True if the ingestion report exists AND the
    fingerprint recorded in it matches the current file's fingerprint.
    This prevents re-processing if the same PDF is uploaded again.
    """
    pdf_name = pdf_path.stem
    report_path = docs_dir / pdf_name / "ingestion_report.json"
    if not report_path.exists():
        return False
    try:
        report  = load_json(report_path)
        stored  = report.get("fingerprint", "")
        current = pdf_fingerprint(pdf_path)
        if stored == current:
            log.info("Cache hit: '%s' already processed (fingerprint match). Skipping ingestion.", pdf_name)
            return True
        log.info("Fingerprint mismatch for '%s' — will re-process.", pdf_name)
        return False
    except Exception:
        return False


def chunks_path_for(docs_dir: Path, pdf_name: str, chunker: str = "hybrid") -> Path:
    return docs_dir / pdf_name / f"{chunker}_chunks.json"
