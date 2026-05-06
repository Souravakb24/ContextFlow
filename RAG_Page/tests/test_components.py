"""
test_components.py — Tests for HTML component builders.
"""

import pytest


def test_chunk_card_html_contains_doc_name():
    from app.ui.components import chunk_card_html
    hit = {
        "document_name": "my_paper",
        "page_numbers":  "3,4",
        "headings":      "Introduction",
        "chunk_type":    "text",
        "reranker_score": 0.85,
        "raw_text":      "This is a test chunk with some content.",
        "text":          "This is a test chunk with some content.",
    }
    html = chunk_card_html(hit, 1)
    assert "my_paper" in html
    assert "Introduction" in html
    assert "3,4" in html
    assert "score-high" in html


def test_chunk_card_html_medium_score():
    from app.ui.components import chunk_card_html
    hit = {"document_name": "d", "page_numbers": "1", "headings": "",
           "chunk_type": "text", "reranker_score": 0.62,
           "raw_text": "text", "text": "text"}
    html = chunk_card_html(hit, 1)
    assert "score-medium" in html


def test_chunk_card_html_low_score():
    from app.ui.components import chunk_card_html
    hit = {"document_name": "d", "page_numbers": "1", "headings": "",
           "chunk_type": "text", "reranker_score": 0.3,
           "raw_text": "text", "text": "text"}
    html = chunk_card_html(hit, 1)
    assert "score-low" in html


def test_all_chunks_html_empty():
    from app.ui.components import all_chunks_html
    html = all_chunks_html([])
    assert "Ask a question" in html


def test_all_chunks_html_multiple():
    from app.ui.components import all_chunks_html
    hits = [
        {"document_name": f"doc_{i}", "page_numbers": str(i), "headings": "",
         "chunk_type": "text", "reranker_score": 0.7 - i * 0.1,
         "raw_text": f"content {i}", "text": f"content {i}"}
        for i in range(3)
    ]
    html = all_chunks_html(hits)
    assert "doc_0" in html
    assert "doc_1" in html
    assert "doc_2" in html


def test_pdf_library_html_empty():
    from app.ui.components import pdf_library_html
    html = pdf_library_html([])
    assert "No documents" in html


def test_pdf_library_html_with_docs():
    from app.ui.components import pdf_library_html
    html = pdf_library_html(["paper_a", "paper_b"], active="paper_a")
    assert "paper_a" in html
    assert "paper_b" in html
    assert "active" in html


def test_db_stats_html():
    from app.ui.components import db_stats_html
    html = db_stats_html(5, 1234)
    assert "5" in html
    assert "1234" in html


def test_query_meta_html_simple():
    from app.ui.components import query_meta_html
    meta = {"retrieval_mode": "single", "is_complex": False,
            "candidates_before_rerank": 10, "sub_queries": []}
    html = query_meta_html(meta)
    assert "single" in html
    assert "10" in html


def test_query_meta_html_complex():
    from app.ui.components import query_meta_html
    meta = {
        "retrieval_mode": "multi-query",
        "is_complex": True,
        "candidates_before_rerank": 20,
        "sub_queries": ["What is X?", "How does Y work?"],
    }
    html = query_meta_html(meta)
    assert "multi-query" in html
    assert "What is X?" in html
    assert "How does Y work?" in html
