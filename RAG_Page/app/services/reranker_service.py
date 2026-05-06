"""
reranker_service.py — CrossEncoder bge-reranker-v2-m3 singleton on cuda:1.
"""

from __future__ import annotations

import math
import numpy as np
from app.config import cfg
from app.logger import get_logger
from app.utils.gpu_utils import print_device_info, log_gpu_memory

log = get_logger(__name__)

_RERANKER = None


def init_reranker(device: str | None = None) -> None:
    global _RERANKER
    from sentence_transformers import CrossEncoder

    dev = device or cfg.reranker_device
    log.info("Loading reranker model: %s on %s …", cfg.reranker_model, dev)
    print_device_info(dev)

    _RERANKER = CrossEncoder(
        cfg.reranker_model,
        max_length=cfg.reranker_max_len,
        device=dev,
    )
    log_gpu_memory(dev, "post-reranker-load")
    log.info("Reranker model ready.")


def get_reranker():
    if _RERANKER is None:
        log.warning("Reranker not initialized — auto-initializing with defaults.")
        init_reranker()
    return _RERANKER


def rerank(query: str, hits: list[dict]) -> list[dict]:
    """
    Score all hits against the query using the CrossEncoder.
    Adds 'reranker_score' (sigmoid-normalized) to each hit.
    Returns hits sorted by reranker_score descending.
    """
    if not hits:
        return hits

    reranker = get_reranker()
    log.info("Reranking %d candidates against query …", len(hits))

    pairs      = [[query, h.get("raw_text") or h.get("text", "")] for h in hits]
    raw_scores = reranker.predict(pairs)

    for hit, rs in zip(hits, raw_scores):
        hit["reranker_score"] = round(1.0 / (1.0 + math.exp(-float(rs))), 6)

    hits.sort(key=lambda h: h["reranker_score"], reverse=True)
    log.info(
        "Reranking complete. Top score: %.4f, Bottom score: %.4f",
        hits[0]["reranker_score"],
        hits[-1]["reranker_score"],
    )
    log_gpu_memory(cfg.reranker_device, "post-rerank")
    return hits
