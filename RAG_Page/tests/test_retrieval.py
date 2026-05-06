"""
test_retrieval.py — Unit tests for hybrid retrieval and RRF fusion.
"""

import pytest


def test_clean_query_removes_stopwords():
    from app.core.retrieval import clean_query
    tokens = clean_query("What is the attention mechanism in transformers?")
    assert "what" not in tokens
    assert "the" not in tokens
    assert "in" not in tokens
    assert "attention" in tokens
    assert "mechanism" in tokens
    assert "transformers" in tokens


def test_rrf_fuse_basic():
    from app.core.retrieval import _rrf_fuse

    id_to_data = {
        "A": {"score": 0.0, "dense_score": 0.9, "bm25_score": 0.0, "text": "a"},
        "B": {"score": 0.0, "dense_score": 0.5, "bm25_score": 0.0, "text": "b"},
        "C": {"score": 0.0, "dense_score": 0.0, "bm25_score": 0.8, "text": "c"},
    }
    dense_ranked = ["A", "B", "C"]
    bm25_ranked  = ["C", "A", "B"]

    fused = _rrf_fuse(dense_ranked, bm25_ranked, id_to_data)
    ids   = [h["score"] for h in fused]

    # A is ranked 1st in dense, 2nd in BM25 → high RRF
    # C is ranked 3rd in dense, 1st in BM25 → high RRF
    # B is ranked 2nd in dense, 3rd in BM25 → lower RRF
    assert len(fused) == 3
    scores = {f["dense_score"]: f for f in fused}


def test_rrf_fuse_top_k():
    from app.core.retrieval import _rrf_fuse

    id_to_data = {str(i): {"score": 0.0, "dense_score": 0.0, "bm25_score": 0.0} for i in range(10)}
    ranked     = [str(i) for i in range(10)]

    fused = _rrf_fuse(ranked, ranked, id_to_data, top_k=3)
    assert len(fused) == 3


def test_parse_hit_structure():
    from app.core.retrieval import _parse_hit
    import json

    prov = [{"page_no": 1, "x0": 10, "y0": 20, "x1": 100, "y1": 80}]
    meta = {
        "document_name": "test_doc",
        "chunk_index":   5,
        "chunk_type":    "text",
        "chunker":       "hybrid",
        "headings":      "Introduction",
        "page_numbers":  "1",
        "provenance":    json.dumps(prov),
        "raw_text":      "Some raw text.",
    }
    hit = _parse_hit("Some text.", meta, dense_score=0.8, bm25_score=0.3)

    assert hit["document_name"] == "test_doc"
    assert hit["chunk_index"]   == 5
    assert hit["dense_score"]   == 0.8
    assert hit["bm25_score"]    == 0.3
    assert hit["provenance_parsed"] == prov


def test_heuristic_is_complex():
    from app.services.llm_service import _heuristic_is_complex

    simple  = "What is KV cache?"
    complex1 = "How does attention work and also explain what are the trade-offs compared to RNN?"

    assert not _heuristic_is_complex(simple)
    assert _heuristic_is_complex(complex1)


def test_heuristic_split():
    from app.services.llm_service import _heuristic_split

    q      = "What is KV cache? And how does it affect memory? Also explain attention."
    parts  = _heuristic_split(q)

    assert len(parts) >= 2
    assert all(len(p.split()) >= 4 for p in parts)
