"""
main.py — ContextFlow entry point.
Initializes logging, warms up all models, starts session cleanup, launches Gradio.
"""

from __future__ import annotations

import sys
import time

from app.logger import init_logging, get_logger
from app.config import cfg

# Init logging first — everything else uses it
init_logging(cfg.logs_dir, cfg.log_level, cfg.log_to_file)
log = get_logger(__name__)


def check_ollama() -> bool:
    from app.services.llm_service import check_ollama_ready
    log.info("Checking Ollama connectivity …")
    ok = check_ollama_ready()
    if not ok:
        log.warning(
            "Ollama not responding. VLM/LLM features will be disabled until Ollama is available."
        )
    return ok


def init_models() -> None:
    from app.services.embedding_service import init_embedding_model
    from app.services.reranker_service import init_reranker
    from app.utils.gpu_utils import print_device_info

    log.info("═" * 60)
    log.info("  ContextFlow — Model Initialization")
    log.info("═" * 60)

    log.info("Initializing embedding model on %s …", cfg.embed_device)
    init_embedding_model(device=cfg.embed_device, fp16=cfg.embed_fp16)

    log.info("Initializing reranker on %s …", cfg.reranker_device)
    init_reranker(device=cfg.reranker_device)

    log.info("All PyTorch models ready.")


def init_chromadb() -> None:
    from app.services.chromadb_service import init_chromadb, collection_count
    init_chromadb()
    log.info("ChromaDB ready. Collection has %d documents.", collection_count())


def start_session_cleanup() -> None:
    from app.core.session_cache import start_cleanup_thread
    start_cleanup_thread()


def main() -> None:
    t0 = time.perf_counter()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║          ContextFlow RAG  —  Starting Up             ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("Config: embed_device=%s  reranker_device=%s  ollama=%s",
             cfg.embed_device, cfg.reranker_device, cfg.ollama_host)

    # 1. ChromaDB
    log.info("── [1/4] ChromaDB …")
    try:
        init_chromadb()
    except Exception as e:
        log.critical("ChromaDB initialization failed: %s", e, exc_info=True)
        sys.exit(1)

    # 2. PyTorch models (embedding + reranker on cuda:1)
    log.info("── [2/4] Embedding + Reranker models …")
    try:
        init_models()
    except Exception as e:
        log.critical("Model initialization failed: %s", e, exc_info=True)
        sys.exit(1)

    # 3. Ollama health check (non-fatal)
    log.info("── [3/4] Ollama health check …")
    check_ollama()

    # 4. Session cleanup thread
    log.info("── [4/4] Session cleanup daemon …")
    start_session_cleanup()

    elapsed = time.perf_counter() - t0
    log.info("Startup complete in %.1fs.", elapsed)
    log.info("Launching Gradio on port %d …", cfg.gradio_port)

    # Build and launch UI
    from app.ui.app import build_app
    from app.ui.theme import make_theme, CUSTOM_CSS
    demo = build_app()
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=cfg.gradio_port,
        share=cfg.gradio_share,
        show_error=True,
        favicon_path=None,
        theme=make_theme(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
