"""
chromadb_service.py — ChromaDB client singleton + collection management.
"""

from __future__ import annotations

import chromadb
from app.config import cfg
from app.logger import get_logger

log = get_logger(__name__)

_CLIENT:     chromadb.ClientAPI | None = None
_COLLECTION: chromadb.Collection | None = None


def init_chromadb() -> None:
    global _CLIENT, _COLLECTION
    log.info("Connecting to ChromaDB at: %s …", cfg.chroma_dir)
    _CLIENT = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    _COLLECTION = _CLIENT.get_or_create_collection(
        name=cfg.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("ChromaDB collection '%s' ready — %d existing docs.", cfg.chroma_collection, _COLLECTION.count())


def get_client() -> chromadb.ClientAPI:
    if _CLIENT is None:
        init_chromadb()
    return _CLIENT


def get_collection() -> chromadb.Collection:
    if _COLLECTION is None:
        init_chromadb()
    return _COLLECTION


def collection_count() -> int:
    return get_collection().count()


def upsert_chunks(
    ids:        list[str],
    documents:  list[str],
    embeddings: list[list[float]],
    metadatas:  list[dict],
    batch_size: int = 512,
) -> None:
    col   = get_collection()
    total = len(ids)
    log.info("Upserting %d chunks to ChromaDB …", total)

    for start in range(0, total, batch_size):
        end = start + batch_size
        col.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
        log.debug("  Upserted batch [%d:%d]", start, min(end, total))

    log.info("Upsert complete. Collection now has %d docs.", col.count())


def query_dense(
    query_embedding: list[float],
    n_results: int,
    where: dict | None = None,
) -> dict:
    col = get_collection()
    log.debug("Dense query: n_results=%d, where=%s", n_results, where)
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    log.debug("Dense query returned %d results.", len(results["ids"][0]))
    return results


def get_all_documents(limit: int | None = None) -> dict:
    col = get_collection()
    total = col.count()
    lim   = limit or total
    log.debug("Fetching all %d documents from ChromaDB …", lim)
    return col.get(limit=lim, include=["documents", "metadatas"])


def list_indexed_documents() -> list[str]:
    """Return unique document_name values in the collection."""
    if collection_count() == 0:
        return []
    result = get_all_documents()
    names  = sorted({m.get("document_name", "") for m in result.get("metadatas", []) if m.get("document_name")})
    log.debug("Indexed documents: %s", names)
    return names


def delete_document(document_name: str) -> None:
    col = get_collection()
    log.info("Deleting all chunks for document '%s' …", document_name)
    col.delete(where={"document_name": document_name})
    log.info("Deleted chunks for '%s'. Collection now has %d docs.", document_name, col.count())
