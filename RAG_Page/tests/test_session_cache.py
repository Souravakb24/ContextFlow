"""
test_session_cache.py — Tests for session cache lifecycle.
"""

import time
import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture(autouse=True)
def patch_cache_dir(tmp_path, monkeypatch):
    """Redirect cache dir to a temp directory for isolation."""
    monkeypatch.setattr("app.config.cfg.cache_dir", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()


def test_new_session_creates_directory():
    from app.core.session_cache import new_session, _session_dir
    sid = new_session()
    assert _session_dir(sid).exists()


def test_save_and_load_retrieval_results():
    from app.core.session_cache import new_session, save_retrieval_results, load_retrieval_results
    sid = new_session()

    hits = [{"document_name": "doc_a", "chunk_index": 0, "text": "hello"}]
    meta = {"original_query": "test", "is_complex": False}

    save_retrieval_results(sid, hits, meta)
    loaded_hits, loaded_meta = load_retrieval_results(sid)

    assert loaded_hits == hits
    assert loaded_meta["original_query"] == "test"


def test_save_and_load_metadata():
    from app.core.session_cache import new_session, save_metadata, load_metadata
    sid  = new_session()
    data = {"proof_pages": [{"doc": "doc_a", "page": 1}]}

    save_metadata(sid, data)
    loaded = load_metadata(sid)
    assert loaded["proof_pages"][0]["page"] == 1


def test_delete_metadata():
    from app.core.session_cache import new_session, save_metadata, delete_metadata, load_metadata, _session_dir
    sid = new_session()
    save_metadata(sid, {"test": True})

    meta_path = _session_dir(sid) / "metadata.json"
    assert meta_path.exists()

    delete_metadata(sid)
    assert not meta_path.exists()

    # load after delete returns empty dict (no error)
    assert load_metadata(sid) == {}


def test_clear_session():
    from app.core.session_cache import new_session, save_metadata, clear_session, _session_dir
    sid = new_session()
    save_metadata(sid, {"test": True})

    clear_session(sid)
    assert not _session_dir(sid).exists()


def test_load_results_nonexistent_session():
    from app.core.session_cache import load_retrieval_results
    hits, meta = load_retrieval_results("nonexistent-uuid")
    assert hits == []
    assert meta == {}
