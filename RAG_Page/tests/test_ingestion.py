"""
test_ingestion.py — Unit tests for ingestion pipeline stages.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_pdf_fingerprint_stable(tmp_path):
    from app.utils.file_utils import pdf_fingerprint
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content for testing")
    fp1 = pdf_fingerprint(pdf)
    fp2 = pdf_fingerprint(pdf)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_pdf_fingerprint_changes_on_content(tmp_path):
    from app.utils.file_utils import pdf_fingerprint
    pdf1 = tmp_path / "a.pdf"
    pdf2 = tmp_path / "b.pdf"
    pdf1.write_bytes(b"content A")
    pdf2.write_bytes(b"content B")
    assert pdf_fingerprint(pdf1) != pdf_fingerprint(pdf2)


def test_is_already_processed_no_report(tmp_path):
    from app.utils.file_utils import is_already_processed
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake pdf")
    assert not is_already_processed(pdf, tmp_path)


def test_is_already_processed_with_matching_fingerprint(tmp_path):
    from app.utils.file_utils import is_already_processed, pdf_fingerprint, save_json, ensure_dir
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake pdf content")
    fp  = pdf_fingerprint(pdf)

    doc_dir = tmp_path / "doc"
    ensure_dir(doc_dir)
    save_json(doc_dir / "ingestion_report.json", {"fingerprint": fp})

    assert is_already_processed(pdf, tmp_path)


def test_is_already_processed_mismatched_fingerprint(tmp_path):
    from app.utils.file_utils import is_already_processed, save_json, ensure_dir
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake pdf content")

    doc_dir = tmp_path / "doc"
    ensure_dir(doc_dir)
    save_json(doc_dir / "ingestion_report.json", {"fingerprint": "wrong_hash"})

    assert not is_already_processed(pdf, tmp_path)


def test_normalize_chunk_bboxes():
    from app.core.ingestion import _normalize_chunk_bboxes
    chunk = {
        "meta": {
            "doc_items": [{
                "prov": [{
                    "bbox": {"l": 10.0, "t": 20.0, "r": 100.0, "b": 5.0},
                    "page_no": 3,
                }]
            }]
        }
    }
    _normalize_chunk_bboxes(chunk)
    prov = chunk["meta"]["doc_items"][0]["prov"][0]
    assert "x0" in prov["bbox"]
    assert prov["bbox"]["x0"] == 10.0
    assert chunk.get("page") == 3


def test_build_page_markdown_basic():
    from app.core.ingestion import build_page_markdown
    elements = [
        {"type": "title",          "content": "My Title"},
        {"type": "section_header", "content": "Introduction"},
        {"type": "text",           "content": "Some paragraph text."},
        {"type": "list_item",      "content": "Item A"},
    ]
    md = build_page_markdown(1, elements)
    assert "# My Title" in md
    assert "## Introduction" in md
    assert "Some paragraph text." in md
    assert "- Item A" in md


def test_build_figure_chunks_no_vlm():
    from app.core.ingestion import build_figure_chunks
    page_elements = {
        1: [
            {"type": "picture", "description": "", "caption": "", "image_ref": "", "bbox": {}},
        ]
    }
    chunks = build_figure_chunks(page_elements)
    assert len(chunks) == 0  # skipped because description is empty


def test_build_figure_chunks_with_vlm():
    from app.core.ingestion import build_figure_chunks
    page_elements = {
        1: [
            {"type": "section_header", "content": "Results"},
            {
                "type": "picture",
                "description": "A bar chart showing accuracy over epochs.",
                "caption": "Figure 1",
                "image_ref": "images/page_1_pic_1.png",
                "bbox": {"x0": 10, "y0": 20, "x1": 100, "y1": 80},
            },
        ]
    }
    chunks = build_figure_chunks(page_elements)
    assert len(chunks) == 1
    assert chunks[0]["parent_heading"] == "Results"
    assert "bar chart" in chunks[0]["text"]
