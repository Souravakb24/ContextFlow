"""
logger.py — Centralized rotating file + console logger for ContextFlow.
Usage: from app.logger import get_logger
       log = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_initialized: bool = False


def _setup(log_dir: Path, log_level: str, log_to_file: bool) -> None:
    global _initialized
    if _initialized:
        return

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler (daily rotation, keep 14 days)
    if log_to_file:
        fh = TimedRotatingFileHandler(
            filename=log_dir / "contextflow.log",
            when="midnight",
            backupCount=14,
            encoding="utf-8",
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "fitz"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True


def init_logging(log_dir: Path | None = None, log_level: str = "INFO", log_to_file: bool = True) -> None:
    """Call once from main.py at startup."""
    from app.config import cfg
    _setup(
        log_dir=log_dir or cfg.logs_dir,
        log_level=log_level,
        log_to_file=log_to_file,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Safe to call before init_logging — will auto-init with defaults."""
    if not _initialized:
        try:
            from app.config import cfg
            _setup(cfg.logs_dir, cfg.log_level, cfg.log_to_file)
        except Exception:
            _setup(Path("logs"), "INFO", False)
    return logging.getLogger(name)
