"""
Product configuration loader.

Each product (ERP, FinTech, HR, etc.) has its own JSON config file in
the products/ directory. The active product is selected via the PRODUCT
environment variable (default: "erp").

A product config drives:
  - Which ChromaDB collection to use
  - The LLM system prompt
  - Column mapping for CSV/JSON ingestion
  - UI labels, placeholder text, and suggested prompts
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import config

logger = logging.getLogger(__name__)

_DEFAULT_COLUMN_MAP: Dict[str, str] = {
    "ticket_id":   "ticket_id",
    "title":       "title",
    "description": "description",
    "resolution":  "resolution",
    "category":    "category",
    "module":      "module",
    "priority":    "priority",
    "status":      "status",
    "created_at":  "created_at",
    "tags":        "tags",
}

_FALLBACK: Dict[str, Any] = {
    "name":       "Support AI",
    "collection": config.COLLECTION_NAME,
    "system_prompt": (
        "You are a support assistant. Answer using ONLY the provided context. "
        "Cite document IDs when referencing a source. Give numbered steps for fixes. "
        "If the context is insufficient, say so clearly. Never invent IDs. Be concise."
    ),
    "column_map": _DEFAULT_COLUMN_MAP,
    "ui": {
        "brand":            "Support AI",
        "welcome_title":    "Support Assistant",
        "welcome_subtitle": "Ask me anything about your product. I'll search the knowledge base for relevant answers.",
        "input_placeholder": "Ask a question…",
        "suggested_prompts": [],
    },
}

_cache: Dict[str, Any] | None = None


def load() -> Dict[str, Any]:
    """Return the active product config, cached after first load."""
    global _cache
    if _cache is not None:
        return _cache

    path = Path(f"products/{config.PRODUCT}.json")
    if not path.exists():
        logger.warning(
            "Product config not found: %s — using built-in fallback", path
        )
        _cache = _FALLBACK
        return _cache

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Merge with fallback so missing keys always have a value
        merged = {**_FALLBACK, **data}
        merged["ui"] = {**_FALLBACK["ui"], **data.get("ui", {})}
        merged["column_map"] = {**_DEFAULT_COLUMN_MAP, **data.get("column_map", {})}
        _cache = merged
        logger.info("Loaded product config: %s (%s)", data.get("name"), path)
    except Exception as exc:
        logger.error("Failed to load product config %s: %s — using fallback", path, exc)
        _cache = _FALLBACK

    return _cache


def get_collection_name() -> str:
    return load().get("collection", config.COLLECTION_NAME)


def get_system_prompt() -> str:
    return load().get("system_prompt", _FALLBACK["system_prompt"])


def get_column_map() -> Dict[str, str]:
    return load().get("column_map", _DEFAULT_COLUMN_MAP)


def get_ui() -> Dict[str, Any]:
    return load().get("ui", _FALLBACK["ui"])
