"""
config.py — Centralized configuration for ContextFlow RAG.
All paths, model names, device settings, and tuning parameters live here.
Load via: from app.config import cfg
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project root (RAG_Page/) ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    root:            Path = ROOT
    storage_dir:     Path = ROOT / "storage"
    uploads_dir:     Path = ROOT / "storage" / "uploads"
    docs_dir:        Path = ROOT / "storage" / "Documents-OCR"
    chroma_dir:      Path = ROOT / "storage" / "chroma_db"
    cache_dir:       Path = ROOT / "cache" / "sessions"
    logs_dir:        Path = ROOT / "logs"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_collection: str = "rag_chunks"

    # ── Embedding + Reranker (PyTorch — cuda:1) ───────────────────────────────
    embed_model:     str = "BAAI/bge-large-en-v1.5"
    reranker_model:  str = "BAAI/bge-reranker-v2-m3"
    embed_device:    str = field(default_factory=lambda: os.getenv("EMBED_DEVICE", "cuda:1"))
    reranker_device: str = field(default_factory=lambda: os.getenv("RERANKER_DEVICE", "cuda:1"))
    embed_fp16:      bool = field(default_factory=lambda: os.getenv("EMBED_FP16", "true").lower() == "true")

    # ── Ollama (VLM + LLM — managed by Ollama on GPU 0) ──────────────────────
    ollama_host:     str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    vlm_model:       str = field(default_factory=lambda: os.getenv("VLM_MODEL", "qwen3-vl:30b"))
    llm_model:       str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-oss:20b"))
    ollama_timeout:  int = field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "180")))
    ollama_retries:  int = 2

    # ── Ingestion ─────────────────────────────────────────────────────────────
    page_render_dpi:       int  = 150
    formula_render_dpi:    int  = 300
    docling_workers:       int  = field(default_factory=lambda: int(os.getenv("DOCLING_WORKERS", "4")))
    skip_vlm:              bool = field(default_factory=lambda: os.getenv("SKIP_VLM", "false").lower() == "true")

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunker_type:    str  = field(default_factory=lambda: os.getenv("CHUNKER_TYPE", "hybrid"))
    tokenizer:       str  = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens:      int | None = None
    merge_peers:     bool = True
    include_raw_text: bool = True

    # ── Retrieval ─────────────────────────────────────────────────────────────
    hybrid_candidates:  int   = 15    # top-N per retriever before RRF
    rrf_k:              int   = 60
    multi_query_chunks: int   = 5     # chunks per sub-query in multi-query mode
    default_top_k:      int   = 5
    reranker_max_len:   int   = 512

    # ── Embedding batch sizes (auto-tuned by gpu_utils if not set) ────────────
    batch_size_cuda:    int   = 512
    batch_size_cpu:     int   = 64

    # ── Session cache ─────────────────────────────────────────────────────────
    session_ttl:     int = field(default_factory=lambda: int(os.getenv("SESSION_TTL", "3600")))

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level:       str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_to_file:     bool = True

    # ── UI ────────────────────────────────────────────────────────────────────
    app_name:        str  = "ContextFlow"
    gradio_port:     int  = field(default_factory=lambda: int(os.getenv("GRADIO_PORT", "7860")))
    gradio_share:    bool = field(default_factory=lambda: os.getenv("GRADIO_SHARE", "false").lower() == "true")

    def __post_init__(self) -> None:
        # Ensure all directories exist at import time
        for d in (
            self.uploads_dir, self.docs_dir, self.chroma_dir,
            self.cache_dir, self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
