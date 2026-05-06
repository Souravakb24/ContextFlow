"""
vlm_service.py — Ollama VLM client (qwen3-vl:30b) for image description and formula extraction.
GPU 0 is managed by the Ollama server process (set via OLLAMA_CUDA_VISIBLE_DEVICES in .env).
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from app.config import cfg
from app.logger import get_logger
from app.prompts import IMAGE_DESCRIPTION_PROMPT, FORMULA_EXTRACTION_PROMPT

log = get_logger(__name__)


def _image_to_b64(img, fmt: str = "PNG") -> str:
    from PIL import Image
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(img)}")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _path_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def call_vlm(b64_image: str, prompt: str) -> str:
    """
    Send a base64-encoded image + prompt to the Ollama VLM.
    Returns the model's response text, or an ERROR: string on failure.
    """
    import urllib.request
    import urllib.error
    import json

    url     = f"{cfg.ollama_host}/v1/chat/completions"
    payload = json.dumps({
        "model":  cfg.vlm_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text",      "text": prompt},
            ],
        }],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(cfg.ollama_retries + 1):
        try:
            log.debug("VLM call attempt %d/%d (model=%s) …", attempt + 1, cfg.ollama_retries + 1, cfg.vlm_model)
            with urllib.request.urlopen(req, timeout=cfg.ollama_timeout) as resp:
                data   = json.loads(resp.read())
                result = data["choices"][0]["message"]["content"].strip()
                log.debug("VLM response (%d chars).", len(result))
                return result
        except urllib.error.URLError as exc:
            if "Connection refused" in str(exc):
                log.error("Ollama not reachable at %s. Is Ollama running?", cfg.ollama_host)
                return f"ERROR: Ollama not reachable at {cfg.ollama_host}"
            if attempt < cfg.ollama_retries:
                wait = 2 ** attempt
                log.warning("VLM attempt %d failed: %s. Retrying in %ds …", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                log.error("VLM all retries exhausted: %s", exc)
                return f"ERROR: {exc}"
        except Exception as exc:
            if attempt < cfg.ollama_retries:
                wait = 2 ** attempt
                log.warning("VLM attempt %d exception: %s. Retrying in %ds …", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                log.error("VLM all retries exhausted: %s", exc)
                return f"ERROR: {exc}"

    return "ERROR: max retries exceeded"


def describe_image(b64: str) -> str:
    log.debug("Describing image via VLM …")
    return call_vlm(b64, IMAGE_DESCRIPTION_PROMPT)


def extract_formula(b64: str) -> str:
    log.debug("Extracting formula via VLM …")
    return call_vlm(b64, FORMULA_EXTRACTION_PROMPT)


def describe_image_from_path(path: Path) -> str:
    try:
        b64 = _path_to_b64(path)
        return describe_image(b64)
    except Exception as e:
        log.error("Could not read image at %s: %s", path, e)
        return f"ERROR: {e}"


def describe_image_from_pil(img) -> str:
    try:
        b64 = _image_to_b64(img)
        return describe_image(b64)
    except Exception as e:
        log.error("Could not encode PIL image: %s", e)
        return f"ERROR: {e}"
