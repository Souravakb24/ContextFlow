"""
embedding_service.py — bge-large-en-v1.5 singleton on cuda:1.
Initialized once at startup via init_embedding_model().
"""

from __future__ import annotations

import numpy as np
from app.config import cfg
from app.logger import get_logger
from app.utils.gpu_utils import print_device_info, auto_batch_size, log_gpu_memory

log = get_logger(__name__)

_MODEL = None
_BATCH_SIZE: int = 64


def init_embedding_model(device: str | None = None, fp16: bool | None = None) -> None:
    global _MODEL, _BATCH_SIZE
    from sentence_transformers import SentenceTransformer
    import torch

    dev = device or cfg.embed_device
    use_fp16 = fp16 if fp16 is not None else cfg.embed_fp16

    log.info("Loading embedding model: %s on %s …", cfg.embed_model, dev)
    print_device_info(dev)

    _MODEL = SentenceTransformer(cfg.embed_model, device=dev)
    if use_fp16:
        _MODEL = _MODEL.half()
        log.info("Embedding model: fp16 enabled.")
    else:
        log.info("Embedding model: fp32.")

    _BATCH_SIZE = auto_batch_size(dev)
    log.info("Embedding batch size: %d", _BATCH_SIZE)

    # Warmup
    if dev.startswith("cuda"):
        _ = _MODEL.encode(["warmup"], batch_size=1, show_progress_bar=False)
        torch.cuda.synchronize()
        log.info("Embedding model warmup complete.")

    log_gpu_memory(dev, "post-embed-load")
    log.info("Embedding model ready.")


def get_model():
    if _MODEL is None:
        log.warning("Embedding model not initialized — auto-initializing with defaults.")
        init_embedding_model()
    return _MODEL


def embed_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """
    Encode a list of texts. Returns a float32 numpy array of shape (N, dim).
    Embeddings are L2-normalized (cosine similarity == dot product).
    """
    model = get_model()
    bs    = batch_size or _BATCH_SIZE
    log.info("Embedding %d texts (batch_size=%d) …", len(texts), bs)

    encode_kwargs = dict(
        batch_size=bs,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if cfg.embed_device.startswith("cuda"):
        encode_kwargs["device"] = cfg.embed_device

    import torch
    vecs = model.encode(texts, **encode_kwargs)
    if cfg.embed_device.startswith("cuda"):
        torch.cuda.synchronize()
        log_gpu_memory(cfg.embed_device, "post-encode")

    log.info("Embedding complete: shape=%s", vecs.shape)
    return vecs


def embed_query(text: str) -> list[float]:
    """Encode a single query string. Returns a plain Python list."""
    vecs = embed_texts([text], batch_size=1)
    return vecs[0].tolist()
