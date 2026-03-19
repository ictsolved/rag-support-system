"""
ERP / Support AI — FastAPI application entry point.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import secrets

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import config
import ingest as ingest_module
import product as product_cfg
import rag_core
import sessions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Support AI", version="2.0.0")

# req_id → threading.Event  (lets clients cancel in-flight generation)
_active: Dict[str, threading.Event] = {}


# ── Admin auth ────────────────────────────────────────────────────────────────

def _require_admin(x_admin_token: Optional[str] = Header(None)):
    """Dependency: validates X-Admin-Token header against ADMIN_PASSWORD."""
    if not config.ADMIN_PASSWORD:
        return          # no password configured — open access (dev mode)
    if not x_admin_token or not secrets.compare_digest(x_admin_token, config.ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str

class StopRequest(BaseModel):
    req_id: str

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Conversation"

class IngestFileRequest(BaseModel):
    filepath: str
    column_map: Optional[dict] = None

class UpsertDocRequest(BaseModel):
    """Upsert a single document in real time."""
    doc_id: str
    data:   Dict[str, str]   # standard field names (title, description, resolution …)

class BatchUpsertRequest(BaseModel):
    """Upsert multiple documents in real time (no file needed)."""
    documents: List[Dict[str, str]]  # each dict must contain a 'doc_id' key

class FeedbackRequest(BaseModel):
    message_id: str
    session_id: str
    rating:     int          # 1 = helpful, -1 = not helpful
    comment:    Optional[str] = ""

class WebhookRequest(BaseModel):
    action:    str                          # upsert | upsert_batch | delete
    doc_id:    Optional[str]       = None
    data:      Optional[Dict[str, str]] = None
    documents: Optional[List[Dict[str, str]]] = None


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path("templates/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI template not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Product config ────────────────────────────────────────────────────────────

@app.get("/api/product")
async def get_product():
    """Return UI-safe product config fields for the active product."""
    prod = product_cfg.load()
    return {
        "name": prod.get("name", "Support AI"),
        "ui":   prod.get("ui",   {}),
    }


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest):
    return sessions.create_session(req.title)

@app.get("/api/sessions")
async def list_sessions():
    return sessions.list_sessions()

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = sessions.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if not sessions.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not sessions.get_session(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    req_id     = str(uuid4())
    stop_event = threading.Event()
    _active[req_id] = stop_event
    history    = sessions.get_history(req.session_id)

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'req_id', 'req_id': req_id})}\n\n"

            sources = rag_core.retrieve_context(req.message)
            # Strip raw content before sending to client — LLM uses full sources internally
            safe_sources = [
                {k: v for k, v in s.items() if k != "excerpt"}
                for s in sources
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': safe_sources})}\n\n"

            messages = rag_core.build_messages(history, sources, req.message)
            full_response = ""
            for token in rag_core.stream_response(messages, stop_event):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            sessions.add_message(req.session_id, "user", req.message)
            ai_msg = sessions.add_message(req.session_id, "assistant", full_response, safe_sources)
            msg_id = ai_msg["id"] if ai_msg else ""

            yield f"data: {json.dumps({'type': 'done', 'message_id': msg_id})}\n\n"

        except Exception as exc:
            logger.error("Chat error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            _active.pop(req_id, None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/stop")
async def stop_generation(req: StopRequest):
    event = _active.get(req.req_id)
    if event:
        event.set()
        return {"ok": True}
    return {"ok": False, "detail": "Request ID not found"}


# ── Live data CRUD ────────────────────────────────────────────────────────────

@app.post("/api/data", status_code=201, dependencies=[Depends(_require_admin)])
async def upsert_doc(req: UpsertDocRequest):
    """
    Add or update a single document in the knowledge base.

    Example body:
        {
            "doc_id": "TKT-999",
            "data": {
                "title":       "Cannot export to Excel",
                "description": "Export button does nothing in Firefox",
                "resolution":  "Cleared browser cache; updated to Firefox 125",
                "category":    "UI",
                "priority":    "Medium"
            }
        }
    """
    try:
        ingest_module.upsert_document(req.doc_id, req.data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True, "doc_id": req.doc_id}

@app.post("/api/data/batch", status_code=201, dependencies=[Depends(_require_admin)])
async def batch_upsert(req: BatchUpsertRequest):
    """
    Add or update multiple documents in one call.

    Each object in `documents` must have a 'doc_id' key plus any data fields.
    Runs synchronously; for very large batches use /api/ingest (file-based).
    """
    result = ingest_module.upsert_documents(req.documents)
    return result

@app.delete("/api/data/{doc_id}", dependencies=[Depends(_require_admin)])
async def delete_doc(doc_id: str):
    """Remove a document from the knowledge base by its ID."""
    if not ingest_module.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True, "doc_id": doc_id}

@app.get("/api/data/{doc_id}")
async def get_doc(doc_id: str):
    """Retrieve a document's stored text and metadata."""
    doc = ingest_module.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ── Feedback ─────────────────────────────────────────────────────────────────

@app.post("/api/feedback", status_code=201)
async def submit_feedback(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=422, detail="rating must be 1 or -1")
    result = sessions.add_feedback(req.message_id, req.session_id, req.rating, req.comment or "")
    return result


# ── Webhook (external push) ───────────────────────────────────────────────────

@app.post("/api/webhook")
async def webhook(req: WebhookRequest, x_api_key: Optional[str] = Header(None)):
    """
    Receive real-time document updates from external systems (ERP, CRM, etc.)
    Set WEBHOOK_SECRET in .env to require authentication via X-Api-Key header.

    Supported actions:
      upsert        – add / update a single document
      upsert_batch  – add / update multiple documents
      delete        – remove a document by ID
    """
    if config.WEBHOOK_SECRET and x_api_key != config.WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Api-Key")

    if req.action == "upsert":
        if not req.doc_id or not req.data:
            raise HTTPException(status_code=422, detail="doc_id and data are required")
        try:
            ingest_module.upsert_document(req.doc_id, req.data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, "action": "upsert", "doc_id": req.doc_id}

    elif req.action == "upsert_batch":
        if not req.documents:
            raise HTTPException(status_code=422, detail="documents list is required")
        result = ingest_module.upsert_documents(req.documents)
        return {"ok": True, "action": "upsert_batch", **result}

    elif req.action == "delete":
        if not req.doc_id:
            raise HTTPException(status_code=422, detail="doc_id is required")
        if not ingest_module.delete_document(req.doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"ok": True, "action": "delete", "doc_id": req.doc_id}

    else:
        raise HTTPException(status_code=422, detail=f"Unknown action: {req.action!r}")


# ── Admin UI + API ────────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    html_path = Path("templates/admin.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Admin template not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


class AdminLoginRequest(BaseModel):
    password: str

@app.post("/api/admin/login")
async def admin_login(req: AdminLoginRequest):
    """Validate admin password. Returns 200 if correct, 401 if not."""
    if not config.ADMIN_PASSWORD:
        return {"ok": True}   # no password configured
    if not secrets.compare_digest(req.password, config.ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Incorrect password")
    return {"ok": True}


@app.get("/api/admin/stats", dependencies=[Depends(_require_admin)])
async def admin_stats():
    db_stats  = sessions.get_stats()
    fb_stats  = sessions.get_feedback_summary()
    prod      = product_cfg.load()
    return {
        "product":      prod.get("name"),
        "collection":   product_cfg.get_collection_name(),
        "doc_count":    rag_core.get_doc_count(),
        "llm_model":    config.LLM_MODEL,
        **db_stats,
        **fb_stats,
    }


@app.get("/api/admin/docs", dependencies=[Depends(_require_admin)])
async def admin_list_docs(limit: int = 20, offset: int = 0):
    return rag_core.list_docs(limit=limit, offset=offset)


@app.get("/api/admin/docs/search", dependencies=[Depends(_require_admin)])
async def admin_search_docs(q: str, limit: int = 20):
    """Semantic search across the knowledge base (for admin browsing)."""
    try:
        collection = rag_core._get_collection()
        n = min(limit, collection.count())
        if n == 0:
            return {"query": q, "results": []}
        raw = collection.query(
            query_texts=[q],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for did, doc, meta, dist in zip(
            raw["ids"][0], raw["documents"][0],
            raw["metadatas"][0], raw["distances"][0],
        ):
            docs.append({
                "doc_id":      did,
                "title":       meta.get("title", "Untitled"),
                "category":    meta.get("category", ""),
                "module":      meta.get("module",   ""),
                "priority":    meta.get("priority", ""),
                "relevance":   round(max(0.0, 1.0 - dist), 3),
                "text_preview": doc[:200],
                "metadata":    meta,
            })
        return {"query": q, "results": docs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/admin/feedback", dependencies=[Depends(_require_admin)])
async def admin_feedback(limit: int = 50, offset: int = 0):
    return {
        "summary":  sessions.get_feedback_summary(),
        "feedback": sessions.list_feedback(limit, offset),
    }


@app.get("/api/admin/sessions", dependencies=[Depends(_require_admin)])
async def admin_list_sessions():
    return sessions.list_sessions()


@app.get("/api/admin/sessions/{session_id}", dependencies=[Depends(_require_admin)])
async def admin_get_session(session_id: str):
    session = sessions.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ── Bulk file ingestion ───────────────────────────────────────────────────────

@app.post("/api/ingest/upload", status_code=202, dependencies=[Depends(_require_admin)])
async def upload_ingest(file: UploadFile = File(...)):
    """
    Upload a ticket file directly from the browser.
    The file is saved to a temp path and ingested in a background thread.
    Supports .csv, .json, .jsonl
    """
    status = ingest_module.get_status()
    if status["running"]:
        raise HTTPException(status_code=409, detail="Ingestion already in progress")

    suffix = Path(file.filename or "upload.csv").suffix.lower()
    if suffix not in (".csv", ".json", ".jsonl"):
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {suffix!r}")

    import tempfile
    content = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()

    def _run():
        try:
            ingest_module.ingest_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    threading.Thread(target=_run, daemon=True, name="ingest-worker").start()
    return {"message": "Ingestion started", "filename": file.filename}


@app.post("/api/ingest", dependencies=[Depends(_require_admin)])
async def start_ingest(req: IngestFileRequest):
    status = ingest_module.get_status()
    if status["running"]:
        raise HTTPException(status_code=409, detail="Ingestion already in progress")

    def _run():
        ingest_module.ingest_file(req.filepath, req.column_map)

    threading.Thread(target=_run, daemon=True, name="ingest-worker").start()
    return {"message": "Ingestion started", "filepath": req.filepath}

@app.get("/api/ingest/status")
async def ingest_status():
    return ingest_module.get_status()


# ── System status ─────────────────────────────────────────────────────────────

@app.get("/api/status")
async def system_status():
    prod = product_cfg.load()
    return {
        "ok":           True,
        "product":      prod.get("name"),
        "collection":   product_cfg.get_collection_name(),
        "llm_model":    config.LLM_MODEL,
        "embed_model":  config.EMBED_MODEL,
        "ticket_count": rag_core.get_doc_count(),
        "version":      "2.0.0",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    os.makedirs(config.DATA_DIR, exist_ok=True)
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False, log_level="info")
