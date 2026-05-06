"""
gpu_utils.py — Device detection, VRAM-based batch sizing, memory logging.
"""

from __future__ import annotations

import sys
from app.logger import get_logger

log = get_logger(__name__)

_VRAM_BATCH: list[tuple[int, int]] = [
    (24 * 1024**3, 512),
    (16 * 1024**3, 384),
    (10 * 1024**3, 256),
    (6  * 1024**3, 128),
    (0,             64),
]


def resolve_device(requested: str | None = None) -> str:
    """
    Return a valid PyTorch device string.
    Priority: explicit request → first CUDA GPU → MPS → exit.
    CPU is not allowed for embedding/reranking.
    """
    import torch

    if requested:
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                log.critical("CUDA requested but no CUDA GPU found. Aborting.")
                sys.exit(1)
            idx = int(requested.split(":")[-1]) if ":" in requested else 0
            if idx >= torch.cuda.device_count():
                log.critical("cuda:%d requested but only %d GPU(s) available.", idx, torch.cuda.device_count())
                sys.exit(1)
            log.info("Using requested device: %s", requested)
            return requested
        if requested == "mps":
            import torch
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                log.critical("MPS requested but not available.")
                sys.exit(1)
            log.info("Using MPS device.")
            return "mps"
        if requested == "cpu":
            log.critical("CPU device is not allowed. Use cuda or mps.")
            sys.exit(1)

    import torch
    if torch.cuda.is_available():
        dev = "cuda:0"
        log.info("Auto-selected device: %s", dev)
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log.info("Auto-selected device: mps")
        return "mps"

    log.critical("No GPU found (CUDA or MPS required). Aborting.")
    sys.exit(1)


def auto_batch_size(device: str) -> int:
    """Return a batch size tuned to available VRAM."""
    import torch

    if not device.startswith("cuda"):
        return 64

    dev_idx = int(device.split(":")[-1]) if ":" in device else 0
    try:
        total = torch.cuda.get_device_properties(dev_idx).total_memory
        for threshold, bs in _VRAM_BATCH:
            if total >= threshold:
                log.debug("VRAM %.1f GB → batch size %d", total / 1024**3, bs)
                return bs
    except Exception as e:
        log.warning("Could not query VRAM: %s. Using default batch size 64.", e)
    return 64


def log_gpu_memory(device: str, label: str = "") -> None:
    """Log current GPU memory allocation and reservation."""
    import torch

    if not device.startswith("cuda"):
        return
    dev_idx = int(device.split(":")[-1]) if ":" in device else 0
    try:
        alloc   = torch.cuda.memory_allocated(dev_idx) / 1024**2
        reserved = torch.cuda.memory_reserved(dev_idx) / 1024**2
        log.debug("GPU[%d] mem %s— allocated: %.0f MB / reserved: %.0f MB", dev_idx, f"({label}) " if label else "", alloc, reserved)
    except Exception as e:
        log.warning("Could not read GPU memory: %s", e)


def print_device_info(device: str) -> None:
    """Log device details at startup."""
    import torch

    if device.startswith("cuda"):
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        props   = torch.cuda.get_device_properties(dev_idx)
        vram_gb = props.total_memory / 1024**3
        try:
            free_gb = torch.cuda.mem_get_info(dev_idx)[0] / 1024**3
        except Exception:
            free_gb = float("nan")
        log.info(
            "Device: %s — %s  (%.1f GB total, %.1f GB free)  CUDA %s  cuDNN %s",
            device, props.name, vram_gb, free_gb,
            torch.version.cuda, torch.backends.cudnn.version(),
        )
    elif device == "mps":
        log.info("Device: MPS (Apple Silicon GPU)")
