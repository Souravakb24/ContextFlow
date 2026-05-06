"""
llm_service.py — Ollama LLM client (gpt-oss:20b) for query decomposition and RAG synthesis.
GPU 0 is managed by the Ollama server process.
"""

from __future__ import annotations

import json
import re
import time

import ollama

from app.config import cfg
from app.logger import get_logger
from app.prompts import (
    DECOMPOSE_SYSTEM,
    RAG_SYSTEM_PROMPT,
    build_rag_prompt,
)

log = get_logger(__name__)

# Heuristic complexity patterns
_COMPLEXITY_PATTERNS = re.compile(
    r"\b(also explain|additionally|as well as|compared to|vs\b|versus|what trade.offs|"
    r"how does .{5,60} affect|difference between|and what|and how|contrast between)\b",
    re.IGNORECASE,
)


# ── Query decomposition ────────────────────────────────────────────────────────

def _heuristic_is_complex(query: str) -> bool:
    if len(query.split()) > 35 and _COMPLEXITY_PATTERNS.search(query):
        return True
    return len(_COMPLEXITY_PATTERNS.findall(query)) >= 2


def _heuristic_split(query: str) -> list[str]:
    parts = re.split(r"\?\s+|\band\b|\balso\b|\badditionally\b", query, flags=re.IGNORECASE)
    cleaned = []
    for p in parts:
        p = p.strip().strip(".").strip()
        if len(p.split()) >= 4:
            if not p.endswith("?"):
                p += "?"
            cleaned.append(p)
    return cleaned[:4] or [query]


def _call_decompose(messages: list[dict]) -> tuple[bool, list[str]]:
    response   = ollama.chat(
        model=cfg.llm_model,
        messages=messages,
        options={"temperature": 0.0, "num_predict": 512},
    )
    raw = response["message"]["content"].strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    data        = json.loads(raw)
    is_complex  = bool(data.get("complex", False))
    sub_queries = [s.strip() for s in data.get("sub_queries", []) if s.strip()]
    return is_complex, sub_queries


def decompose_query(query: str) -> tuple[bool, list[str]]:
    """
    Use Ollama LLM to detect if query is complex and split it.
    Falls back to heuristic split if LLM is unavailable.
    Returns (is_complex, sub_queries).
    """
    log.info("Analysing query complexity (model=%s) …", cfg.llm_model)
    try:
        messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user",   "content": f"Query: {query}"},
        ]
        is_complex, sub_queries = _call_decompose(messages)
        log.info("LLM verdict: complex=%s, sub_queries=%d", is_complex, len(sub_queries))

        # Heuristic override: LLM said simple but heuristic disagrees
        if not is_complex and _heuristic_is_complex(query):
            log.info("Heuristic override: retrying decomposition with explicit hint …")
            retry_msg = (
                f"This query contains multiple distinct questions. "
                f"It MUST be classified as complex. Split it into focused sub-queries.\n\nQuery: {query}"
            )
            messages2 = [
                {"role": "system", "content": DECOMPOSE_SYSTEM},
                {"role": "user",   "content": retry_msg},
            ]
            is_complex, sub_queries = _call_decompose(messages2)

            if not is_complex and _heuristic_is_complex(query):
                log.info("LLM still returned simple — forcing via heuristic split.")
                is_complex  = True
                sub_queries = _heuristic_split(query)

        return is_complex, sub_queries

    except Exception as e:
        log.warning("Query decomposition LLM call failed: %s. Falling back to heuristic.", e)
        if _heuristic_is_complex(query):
            return True, _heuristic_split(query)
        return False, []


# ── RAG answer synthesis ───────────────────────────────────────────────────────

def synthesize_answer(question: str, hits: list[dict]) -> str:
    """
    Generate a grounded answer from retrieved chunks using the RAG synthesis prompt.
    Returns the LLM's answer string.
    """
    log.info("Synthesizing RAG answer for question: %s …", question[:80])
    user_msg = build_rag_prompt(question, hits)

    try:
        response = ollama.chat(
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            options={"temperature": 0.1, "num_predict": 2048},
        )
        answer = response["message"]["content"].strip()
        log.info("RAG answer generated (%d chars).", len(answer))
        return answer
    except Exception as e:
        log.error("RAG synthesis failed: %s", e)
        return f"Error generating answer: {e}"


def synthesize_answer_stream(question: str, hits: list[dict]):
    """
    Stream RAG answer tokens. Yields string chunks as they arrive from Ollama.
    """
    log.info("Streaming RAG answer for: %s …", question[:80])
    user_msg = build_rag_prompt(question, hits)

    try:
        stream = ollama.chat(
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            options={"temperature": 0.1, "num_predict": 2048},
            stream=True,
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
    except Exception as e:
        log.error("RAG stream failed: %s", e)
        yield f"\n\nError generating answer: {e}"


def check_ollama_ready() -> bool:
    """Ping Ollama to verify the LLM model is loaded and responsive."""
    try:
        ollama.chat(
            model=cfg.llm_model,
            messages=[{"role": "user", "content": "ping"}],
            options={"num_predict": 1},
        )
        log.info("Ollama LLM (%s) is ready.", cfg.llm_model)
        return True
    except Exception as e:
        log.warning("Ollama LLM not ready: %s", e)
        return False
