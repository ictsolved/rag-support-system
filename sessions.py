"""
Session management backed by SQLite.

Replaces the previous flat sessions.json approach with a proper relational
store. SQLite handles concurrent reads/writes safely (WAL mode) and scales
to millions of conversations without memory pressure.

Schema
------
sessions  : id, title, created_at, updated_at
messages  : id, session_id (FK), role, content, sources (JSON), timestamp
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

import config


# ── DB bootstrap ─────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT 'New Conversation',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content    TEXT NOT NULL,
    sources    TEXT NOT NULL DEFAULT '[]',
    timestamp  TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedback (
    id         TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    rating     INTEGER NOT NULL CHECK(rating IN (1, -1)),
    comment    TEXT NOT NULL DEFAULT '',
    timestamp  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session  ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated  ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_message  ON feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_feedback_session  ON feedback(session_id);
"""


@contextmanager
def _db():
    """Open a short-lived connection, commit on exit, rollback on error."""
    conn = sqlite3.connect(config.SESSIONS_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Initialise schema on import
with _db() as _c:
    _c.executescript(_SCHEMA)


# ── Public API ────────────────────────────────────────────────────────────────

def create_session(title: str = "New Conversation") -> Dict:
    sid = str(uuid4())
    now = _now()
    with _db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (sid, title, now, now),
        )
    return {"id": sid, "title": title, "created_at": now, "updated_at": now, "messages": []}


def get_session(session_id: str) -> Optional[Dict]:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        msg_rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()

    session = dict(row)
    session["messages"] = [
        {**dict(m), "sources": json.loads(m["sources"])} for m in msg_rows
    ]
    return session


def list_sessions() -> List[Dict]:
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) AS message_count
            FROM   sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP  BY s.id
            ORDER  BY s.updated_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    with _db() as conn:
        result = conn.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
    return result.rowcount > 0


def add_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List] = None,
) -> Optional[Dict]:
    with _db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not exists:
            return None

        # Auto-title: count existing user messages before this one
        if role == "user":
            user_count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'",
                (session_id,),
            ).fetchone()[0]

        msg_id = str(uuid4())
        now = _now()
        conn.execute(
            """INSERT INTO messages (id, session_id, role, content, sources, timestamp)
               VALUES (?,?,?,?,?,?)""",
            (msg_id, session_id, role, content, json.dumps(sources or []), now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )

        # Set session title from the very first user message
        if role == "user" and user_count == 0:
            title = content[:60] + ("…" if len(content) > 60 else "")
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
            )

    return {
        "id": msg_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "timestamp": now,
    }


def get_history(session_id: str, limit: int = None) -> List[Dict]:
    n = limit or config.HISTORY_MSGS
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
    msgs = [{**dict(r), "sources": json.loads(r["sources"])} for r in rows]
    return msgs[-n:] if len(msgs) > n else msgs


# ── Feedback ──────────────────────────────────────────────────────────────────

def add_feedback(
    message_id: str,
    session_id: str,
    rating: int,
    comment: str = "",
) -> Dict:
    fid = str(uuid4())
    now = _now()
    with _db() as conn:
        conn.execute(
            """INSERT INTO feedback (id, message_id, session_id, rating, comment, timestamp)
               VALUES (?,?,?,?,?,?)""",
            (fid, message_id, session_id, rating, comment, now),
        )
    return {"id": fid, "message_id": message_id, "rating": rating}


def get_feedback_summary() -> Dict:
    with _db() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        positive = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating =  1").fetchone()[0]
        negative = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = -1").fetchone()[0]
    return {"total": total, "positive": positive, "negative": negative}


def list_feedback(limit: int = 50, offset: int = 0) -> List[Dict]:
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT f.id, f.message_id, f.session_id, f.rating, f.comment, f.timestamp,
                   substr(m.content, 1, 120) AS message_preview,
                   s.title                   AS session_title
            FROM   feedback f
            LEFT JOIN messages m ON m.id = f.message_id
            LEFT JOIN sessions s ON s.id = f.session_id
            ORDER  BY f.timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> Dict:
    with _db() as conn:
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    return {"session_count": session_count, "message_count": message_count}
