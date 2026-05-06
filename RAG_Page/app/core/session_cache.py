"""
session_cache.py — Per-session ephemeral metadata store with TTL cleanup.
Each session gets a UUID directory under cache/sessions/.
metadata.json is deleted after proof render.
retrieval_results.json is kept until TTL expiry or a new query.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import cfg
from app.logger import get_logger
from app.utils.file_utils import load_json, save_json

log = get_logger(__name__)

_CLEANUP_INTERVAL = 300  # seconds between cleanup sweeps
_cleanup_thread: threading.Thread | None = None


# ── Session lifecycle ──────────────────────────────────────────────────────────

def new_session() -> str:
    sid = str(uuid.uuid4())
    _session_dir(sid).mkdir(parents=True, exist_ok=True)
    _touch_session(sid)
    log.info("New session created: %s", sid)
    return sid


def _session_dir(sid: str) -> Path:
    return cfg.cache_dir / sid


def _touch_session(sid: str) -> None:
    """Update the session's last-access timestamp."""
    ts_file = _session_dir(sid) / ".last_access"
    ts_file.write_text(str(time.time()), encoding="utf-8")


# ── Write / read / delete ──────────────────────────────────────────────────────

def save_retrieval_results(sid: str, results: list[dict], query_meta: dict) -> None:
    _touch_session(sid)
    path = _session_dir(sid) / "retrieval_results.json"
    save_json(path, {"query_meta": query_meta, "results": results})
    log.info("Session %s: retrieval_results.json saved (%d hits).", sid, len(results))


def load_retrieval_results(sid: str) -> tuple[list[dict], dict]:
    path = _session_dir(sid) / "retrieval_results.json"
    if not path.exists():
        log.warning("Session %s: retrieval_results.json not found.", sid)
        return [], {}
    _touch_session(sid)
    data = load_json(path)
    return data.get("results", []), data.get("query_meta", {})


def save_metadata(sid: str, metadata: dict) -> None:
    """Save bbox provenance metadata for the proof window."""
    _touch_session(sid)
    path = _session_dir(sid) / "metadata.json"
    save_json(path, metadata)
    log.info("Session %s: metadata.json saved.", sid)


def load_metadata(sid: str) -> dict:
    path = _session_dir(sid) / "metadata.json"
    if not path.exists():
        log.warning("Session %s: metadata.json not found.", sid)
        return {}
    _touch_session(sid)
    return load_json(path)


def delete_metadata(sid: str) -> None:
    """Delete bbox metadata after proof window renders it."""
    path = _session_dir(sid) / "metadata.json"
    if path.exists():
        path.unlink()
        log.info("Session %s: metadata.json deleted (post-proof-render).", sid)


def clear_session(sid: str) -> None:
    """Delete all session data (called on new query or explicit clear)."""
    sdir = _session_dir(sid)
    if sdir.exists():
        shutil.rmtree(sdir, ignore_errors=True)
        log.info("Session %s: cleared.", sid)


# ── TTL cleanup thread ─────────────────────────────────────────────────────────

def _cleanup_expired_sessions() -> None:
    now = time.time()
    if not cfg.cache_dir.exists():
        return
    for sdir in cfg.cache_dir.iterdir():
        if not sdir.is_dir():
            continue
        ts_file = sdir / ".last_access"
        try:
            last = float(ts_file.read_text(encoding="utf-8")) if ts_file.exists() else sdir.stat().st_mtime
            if now - last > cfg.session_ttl:
                shutil.rmtree(sdir, ignore_errors=True)
                log.info("Session %s expired and cleaned up.", sdir.name)
        except Exception as e:
            log.warning("Error checking session %s for expiry: %s", sdir.name, e)


def _cleanup_loop() -> None:
    log.info("Session cleanup thread started (TTL=%ds, interval=%ds).", cfg.session_ttl, _CLEANUP_INTERVAL)
    while True:
        time.sleep(_CLEANUP_INTERVAL)
        try:
            _cleanup_expired_sessions()
        except Exception as e:
            log.error("Session cleanup error: %s", e)


def start_cleanup_thread() -> None:
    global _cleanup_thread
    if _cleanup_thread is not None and _cleanup_thread.is_alive():
        return
    _cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True, name="session-cleanup")
    _cleanup_thread.start()
    log.info("Session cleanup daemon thread started.")
