"""
Core RAG logic: retrieval, prompt construction, and streaming inference.
"""

import logging
import threading
from typing import Dict, Generator, List, Optional

import chromadb
import ollama
from chromadb.utils import embedding_functions

import config
import product as product_cfg

logger = logging.getLogger(__name__)

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        ef = embedding_functions.OllamaEmbeddingFunction(
            url=f"{config.OLLAMA_URL}/api/embeddings",
            model_name=config.EMBED_MODEL,
        )
        _collection = client.get_or_create_collection(
            name=product_cfg.get_collection_name(),
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_doc_count() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0


def retrieve_context(query: str, n_results: Optional[int] = None) -> List[Dict]:
    """
    Return the top-n most relevant documents for the query.
    Each result carries ticket metadata and a text excerpt.
    """
    n          = n_results or config.N_RESULTS
    collection = _get_collection()
    count      = collection.count()
    if count == 0:
        return []

    n = min(n, count)
    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    sources = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        relevance = round(max(0.0, 1.0 - dist), 3)
        sources.append({
            "ticket_id": meta.get("ticket_id", "N/A"),
            "title":     meta.get("title",     "Untitled"),
            "category":  meta.get("category",  ""),
            "module":    meta.get("module",     ""),
            "priority":  meta.get("priority",   ""),
            "status":    meta.get("status",     ""),
            "excerpt":   doc[: config.EXCERPT_LEN],
            "relevance": relevance,
        })
    # Drop sources below the confidence threshold — LLM will say "no match"
    sources = [s for s in sources if s["relevance"] >= config.MIN_RELEVANCE]
    return sources


def list_docs(limit: int = 20, offset: int = 0) -> dict:
    """Paginated document listing for the admin panel."""
    try:
        collection = _get_collection()
        total  = collection.count()
        result = collection.get(
            limit=min(limit, 100),
            offset=offset,
            include=["documents", "metadatas"],
        )
        docs = [
            {"doc_id": did, "text_preview": doc[:200], "metadata": meta}
            for did, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]
        return {"total": total, "offset": offset, "limit": limit, "docs": docs}
    except Exception as exc:
        logger.error("list_docs error: %s", exc)
        return {"total": 0, "offset": offset, "limit": limit, "docs": [], "error": str(exc)}


def build_messages(history: List[Dict], sources: List[Dict], query: str) -> List[Dict]:
    """
    Assemble the Ollama chat message list from:
      - product-specific system prompt
      - retrieved context documents
      - recent conversation history
      - the current user query
    """
    system_prompt = product_cfg.get_system_prompt()
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]

    if sources:
        ctx = "\n\n---\n\n".join(
            f"[{s['ticket_id']}] {s['excerpt']}" for s in sources
        )
        messages.append({
            "role":    "system",
            "content": f"Relevant documents from knowledge base:\n\n{ctx}",
        })
    else:
        messages.append({
            "role":    "system",
            "content": (
                "No matching documents found in the knowledge base. "
                "Tell the user you could not find relevant information and suggest they contact support."
            ),
        })

    for msg in history[-config.HISTORY_MSGS:]:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})
    return messages


def stream_response(
    messages: List[Dict],
    stop_event: threading.Event,
) -> Generator[str, None, None]:
    """Stream token-by-token from the LLM; honour stop_event for cancellation."""
    try:
        stream = ollama.chat(
            model=config.LLM_MODEL,
            messages=messages,
            stream=True,
            options={
                "temperature": 0.0,                    # deterministic, no hallucination
                "num_predict": config.NUM_PREDICT,     # cap output length
                # num_ctx omitted — use model's own optimal default
            },
        )
        for chunk in stream:
            if stop_event.is_set():
                logger.info("Generation cancelled by user")
                break
            token: str = chunk.message.content
            if token:
                yield token
    except Exception as exc:
        logger.error("LLM streaming error: %s", exc)
        yield f"\n\n*[Error: {exc}]*"
