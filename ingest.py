"""
Ticket / document ingestion pipeline.

Supports:
  - Bulk import from CSV / JSON / JSONL files  (ingest_file)
  - Real-time single-document upsert           (upsert_document)
  - Real-time batch upsert from JSON body      (upsert_documents)
  - Document deletion                          (delete_document)
  - Document lookup                            (get_document)

The active product config supplies the column mapping and ChromaDB
collection name, so the same pipeline works for any product.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

import config
import product as product_cfg

logger = logging.getLogger(__name__)

_META_LIMIT    = 512   # ChromaDB metadata value character limit
_MAX_DOC_CHARS = 8000  # nomic-embed-text context limit safety margin

# ── Status tracking for bulk ingestion ───────────────────────────────────────

_status_lock = threading.Lock()
_ingest_status: Dict[str, Any] = {
    "running":   False,
    "total":     0,
    "processed": 0,
    "failed":    0,
    "state":     "idle",   # idle | running | completed | failed
    "error":     None,
}


def get_status() -> Dict:
    with _status_lock:
        return dict(_ingest_status)


# ── ChromaDB collection ───────────────────────────────────────────────────────

def _get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    ef = embedding_functions.OllamaEmbeddingFunction(
        url=f"{config.OLLAMA_URL}/api/embeddings",
        model_name=config.EMBED_MODEL,
    )
    return client.get_or_create_collection(
        name=product_cfg.get_collection_name(),
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ── Document formatting ───────────────────────────────────────────────────────

def _clean(val: Any) -> str:
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "") else s


def _format_document(row: Dict, col: Dict[str, str]) -> str:
    """Convert a flat dict of field values into a text document for embedding."""
    title       = _clean(row.get(col.get("title",       "title"),       ""))
    category    = _clean(row.get(col.get("category",    "category"),    ""))
    module      = _clean(row.get(col.get("module",      "module"),      ""))
    priority    = _clean(row.get(col.get("priority",    "priority"),    ""))
    description = _clean(row.get(col.get("description", "description"), ""))
    resolution  = _clean(row.get(col.get("resolution",  "resolution"),  ""))
    tags        = _clean(row.get(col.get("tags",        "tags"),        ""))

    parts = []
    if title:
        parts.append(f"Title: {title}")
    meta = " | ".join(p for p in [
        f"Category: {category}" if category else "",
        f"Module: {module}"     if module   else "",
        f"Priority: {priority}" if priority else "",
    ] if p)
    if meta:
        parts.append(meta)
    if description:
        parts.append(f"Issue: {description}")
    if resolution:
        parts.append(f"Resolution: {resolution}")
    if tags:
        parts.append(f"Tags: {tags}")
    text = "\n".join(parts)
    return text[:_MAX_DOC_CHARS]


def _build_metadata(row: Dict, col: Dict[str, str]) -> Dict[str, str]:
    return {
        field: _clean(row.get(column, ""))[:_META_LIMIT]
        for field, column in col.items()
        if _clean(row.get(column, ""))
    }


# ── Real-time single document ─────────────────────────────────────────────────

def upsert_document(doc_id: str, data: Dict[str, str]) -> None:
    """
    Upsert a single document using the product's column mapping.
    `data` keys should match the *standard* field names
    (ticket_id, title, description, resolution, …).
    """
    col  = product_cfg.get_column_map()
    text = _format_document(data, col)
    if not text.strip():
        raise ValueError("Document produced empty text — check field values")
    meta = _build_metadata(data, col)
    _get_collection().upsert(documents=[text], ids=[doc_id], metadatas=[meta])


def upsert_documents(docs: List[Dict[str, str]], batch_size: int = 100) -> Dict:
    """
    Upsert a list of documents (JSON body, no file required).
    Each dict must have a 'doc_id' key (used as the ChromaDB ID).
    """
    col = product_cfg.get_column_map()
    processed = failed = 0
    collection = _get_collection()

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts, ids, metas = [], [], []

        for doc in batch:
            doc_id = _clean(doc.get("doc_id", ""))
            if not doc_id:
                failed += 1
                continue
            text = _format_document(doc, col)
            if not text.strip():
                failed += 1
                continue
            texts.append(text)
            ids.append(doc_id)
            metas.append(_build_metadata(doc, col))

        if texts:
            collection.upsert(documents=texts, ids=ids, metadatas=metas)
            processed += len(texts)

    return {"processed": processed, "failed": failed, "total": len(docs)}


def delete_document(doc_id: str) -> bool:
    """Delete a document from ChromaDB. Returns True if it existed."""
    try:
        result = _get_collection().get(ids=[doc_id])
        if not result["ids"]:
            return False
        _get_collection().delete(ids=[doc_id])
        return True
    except Exception as exc:
        logger.error("delete_document error: %s", exc)
        return False


def get_document(doc_id: str) -> Optional[Dict]:
    """Fetch a document by ID. Returns None if not found."""
    try:
        result = _get_collection().get(
            ids=[doc_id], include=["documents", "metadatas"]
        )
        if not result["ids"]:
            return None
        return {
            "doc_id":   doc_id,
            "text":     result["documents"][0],
            "metadata": result["metadatas"][0],
        }
    except Exception as exc:
        logger.error("get_document error: %s", exc)
        return None


# ── Bulk file ingestion ───────────────────────────────────────────────────────

def ingest_file(filepath: str, column_map: Optional[Dict] = None, batch_size: int = 100) -> Dict:
    """
    Ingest a ticket file (CSV / JSON / JSONL) into ChromaDB.
    Designed to be called from a background thread.
    column_map overrides the product config's column mapping for this run.
    """
    global _ingest_status

    with _status_lock:
        if _ingest_status["running"]:
            return {"error": "Ingestion already in progress"}
        _ingest_status.update({
            "running": True, "total": 0,
            "processed": 0,  "failed": 0,
            "state": "running", "error": None,
        })

    # Merge: explicit argument > product config
    col = {**product_cfg.get_column_map(), **(column_map or {})}
    path = Path(filepath)

    if not path.exists():
        with _status_lock:
            _ingest_status.update({"running": False, "state": "failed", "error": f"File not found: {filepath}"})
        return {"error": f"File not found: {filepath}"}

    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(filepath, dtype=str, on_bad_lines="skip").fillna("")
        elif suffix == ".json":
            df = pd.read_json(filepath, dtype=str).fillna("")
        elif suffix == ".jsonl":
            df = pd.read_json(filepath, lines=True, dtype=str).fillna("")
        else:
            with _status_lock:
                _ingest_status.update({"running": False, "state": "failed", "error": f"Unsupported file type: {path.suffix}"})
            return {"error": f"Unsupported file type: {path.suffix}"}
    except Exception as exc:
        with _status_lock:
            _ingest_status.update({"running": False, "state": "failed", "error": f"Failed to read file: {exc}"})
        return {"error": f"Failed to read file: {exc}"}

    with _status_lock:
        _ingest_status["total"] = len(df)

    id_col     = col.get("ticket_id", "ticket_id")
    collection = _get_collection()

    # Fetch all existing IDs so we can skip already-indexed documents
    try:
        existing_ids = set(collection.get(include=[])["ids"])
        logger.info("Skipping %d already-indexed documents", len(existing_ids))
    except Exception:
        existing_ids = set()

    try:
        for batch_start in range(0, len(df), batch_size):
            batch = df.iloc[batch_start : batch_start + batch_size]
            docs, ids, metas = [], [], []

            for idx, row in batch.iterrows():
                row_dict  = row.to_dict()
                ticket_id = _clean(row_dict.get(id_col, "")) or f"ticket-{idx}"

                if ticket_id in existing_ids:
                    with _status_lock:
                        _ingest_status["processed"] += 1
                    continue

                text      = _format_document(row_dict, col)

                if not text.strip():
                    with _status_lock:
                        _ingest_status["failed"] += 1
                    continue

                docs.append(text)
                ids.append(ticket_id)
                metas.append(_build_metadata(row_dict, col))

            if docs:
                try:
                    collection.upsert(documents=docs, ids=ids, metadatas=metas)
                    with _status_lock:
                        _ingest_status["processed"] += len(docs)
                except Exception as batch_exc:
                    # Retry one by one so a single bad doc doesn't kill the batch
                    logger.warning("Batch upsert failed (%s), retrying individually", batch_exc)
                    for doc, doc_id, meta in zip(docs, ids, metas):
                        try:
                            collection.upsert(documents=[doc], ids=[doc_id], metadatas=[meta])
                            with _status_lock:
                                _ingest_status["processed"] += 1
                        except Exception as single_exc:
                            logger.warning("Skipping %s: %s", doc_id, single_exc)
                            with _status_lock:
                                _ingest_status["failed"] += 1
                logger.info("Ingested %d / %d", _ingest_status["processed"], _ingest_status["total"])

    except Exception as exc:
        logger.error("Ingestion error: %s", exc)
        with _status_lock:
            _ingest_status.update({"running": False, "state": "failed", "error": str(exc)})
        return {"error": str(exc)}

    with _status_lock:
        _ingest_status.update({"running": False, "state": "completed"})

    return {
        "state":     "completed",
        "processed": _ingest_status["processed"],
        "failed":    _ingest_status["failed"],
        "total":     _ingest_status["total"],
    }
