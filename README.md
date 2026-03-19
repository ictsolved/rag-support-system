# Support AI

A plug-and-play RAG chatbot that turns any historical support data (tickets, docs, FAQs) into an intelligent assistant. Runs fully **locally** — no data leaves your server.

Switch between products with a single environment variable. Each product gets its own knowledge base, system prompt, and UI.

---

## Architecture

```
Browser (Chat UI)
      │  SSE streaming
      ▼
FastAPI (main.py)
      ├── /api/product   →  product.py     (loads products/{PRODUCT}.json)
      ├── /api/sessions  →  sessions.py    (SQLite conversation storage)
      ├── /api/data      →  ingest.py      (real-time document CRUD)
      ├── /api/ingest    →  ingest.py      (bulk file import)
      └── /api/chat      →  rag_core.py
                              ├── ChromaDB  (vector search)
                              └── Ollama    (local LLM inference)
```

| Component | Default | Notes |
|---|---|---|
| LLM | `phi3:mini` | Any Ollama model — local or remote |
| Embeddings | `nomic-embed-text` | Configurable via `EMBED_MODEL` |
| Vector DB | ChromaDB | Persisted in `./db/` |
| Sessions | SQLite | Stored in `./sessions.db` |

---

## Prerequisites

```bash
# Install Ollama: https://ollama.com
ollama pull phi3:mini
ollama pull nomic-embed-text
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (or on a remote server)

## Setup

```bash
git clone https://github.com/ictsolved/rag-support-system.git
cd rag-support-system

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # edit models/ports if needed
cp products/example.json products/myproduct.json   # configure your product

PRODUCT=myproduct python main.py
# → http://localhost:8000
```

---

## Adding a product

1. Copy the template:
   ```bash
   cp products/example.json products/myproduct.json
   ```

2. Edit `products/myproduct.json`:
   ```json
   {
     "name": "My Product Support AI",
     "collection": "myproduct_tickets",
     "system_prompt": "You are a support assistant for My Product...",
     "column_map": {
       "ticket_id":   "id",
       "title":       "subject",
       "description": "body",
       "resolution":  "fix_notes",
       "category":    "dept",
       "module":      "area",
       "priority":    "severity",
       "status":      "state",
       "created_at":  "date",
       "tags":        "labels"
     },
     "ui": {
       "brand":             "My Product AI",
       "welcome_title":     "My Product Assistant",
       "welcome_subtitle":  "Ask me anything. I'll search past tickets for solutions.",
       "input_placeholder": "Describe your issue…",
       "suggested_prompts": [
         { "label": "Login issue",  "prompt": "User cannot log in" },
         { "label": "Data missing", "prompt": "Records are not showing up" }
       ]
     }
   }
   ```

3. Run: `PRODUCT=myproduct python main.py`

`column_map` maps the standard field names the system uses to whatever your CSV/JSON columns are actually named.

---

## Importing data

### Via admin panel
Go to `http://localhost:8000/admin` → **Import** tab → upload a CSV/JSON/JSONL file.

### Via API
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "X-Admin-Token: your_password" \
  -H "Content-Type: application/json" \
  -d '{"filepath": "data/tickets.csv"}'

# Poll progress
curl http://localhost:8000/api/ingest/status
```

### Real-time single document
```bash
curl -X POST http://localhost:8000/api/data \
  -H "X-Admin-Token: your_password" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "TKT-999",
    "data": {
      "title":       "Cannot export report to Excel",
      "description": "Export button does nothing in Firefox 124",
      "resolution":  "Cleared browser cache and updated to Firefox 125",
      "category":    "UI",
      "module":      "Reporting",
      "priority":    "Medium",
      "status":      "Resolved"
    }
  }'
```

### Real-time batch
```bash
curl -X POST http://localhost:8000/api/data/batch \
  -H "X-Admin-Token: your_password" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      { "doc_id": "TKT-100", "title": "...", "description": "...", "resolution": "..." },
      { "doc_id": "TKT-101", "title": "...", "description": "...", "resolution": "..." }
    ]
  }'
```

---

## Extracting data from your system

If your data is in a database or custom format, write a one-off extraction script in `tools/` (git-ignored) that outputs an ingest-ready CSV. See `tools/README.md` for the expected format and an example.

---

## Expected data format

| Column | Required | Description |
|---|---|---|
| `ticket_id` | Yes | Unique ID |
| `title` | Yes | Short summary |
| `description` | Yes | Full issue description |
| `resolution` | Yes | How it was resolved *(most important for RAG)* |
| `category` | No | e.g. Finance, HR, Inventory |
| `module` | No | e.g. Payroll, Inventory, Reports |
| `priority` | No | Low / Medium / High / Critical |
| `status` | No | Open / Resolved / Closed |
| `created_at` | No | Ticket date |
| `tags` | No | Comma-separated keywords |

Column names in your file can differ — remap them via `column_map` in your product config.

---

## Configuration (`.env`)

```env
PRODUCT=myproduct         # loads products/myproduct.json

OLLAMA_URL=http://localhost:11434
LLM_MODEL=phi3:mini
EMBED_MODEL=nomic-embed-text

N_RESULTS=3               # documents retrieved per query   (↓ = faster)
EXCERPT_LEN=250           # chars of each doc shown to LLM  (↓ = faster)
HISTORY_MSGS=6            # last N messages kept in context
NUM_PREDICT=512           # max tokens the LLM can generate
MIN_RELEVANCE=0.35        # similarity threshold (0–1, ↑ = stricter)

ADMIN_PASSWORD=           # set this in production
WEBHOOK_SECRET=           # optional webhook auth

HOST=0.0.0.0
PORT=8000
```

---

## API reference

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | — | Chat UI |
| `GET` | `/admin` | Admin | Admin panel |
| `GET` | `/api/product` | — | Active product config |
| `GET` | `/api/status` | — | System status + ticket count |
| `POST` | `/api/sessions` | — | Create session |
| `GET` | `/api/sessions` | — | List sessions |
| `GET` | `/api/sessions/{id}` | — | Get session with messages |
| `DELETE` | `/api/sessions/{id}` | — | Delete session |
| `POST` | `/api/chat` | — | Send message (SSE streaming) |
| `POST` | `/api/stop` | — | Cancel active generation |
| `POST` | `/api/feedback/{id}` | — | Submit thumbs up/down |
| `POST` | `/api/data` | Admin | Upsert single document |
| `POST` | `/api/data/batch` | Admin | Upsert multiple documents |
| `DELETE` | `/api/data/{id}` | Admin | Delete document |
| `GET` | `/api/data/{id}` | Admin | Retrieve document |
| `POST` | `/api/ingest` | Admin | Start bulk file ingestion |
| `POST` | `/api/ingest/upload` | Admin | Upload + ingest file |
| `GET` | `/api/ingest/status` | — | Ingestion progress |

### Chat SSE events

```
data: {"type": "req_id",  "req_id": "..."}      ← pass to /api/stop to cancel
data: {"type": "sources", "sources": [...]}      ← matched documents
data: {"type": "token",   "content": "..."}      ← one per LLM token
data: {"type": "done",    "message_id": "..."}
data: {"type": "error",   "message": "..."}
```

---

## Project structure

```
├── main.py                  FastAPI app + all API routes
├── rag_core.py              Retrieval, prompt building, LLM streaming
├── ingest.py                Bulk + real-time document ingestion
├── sessions.py              SQLite-backed conversation history
├── product.py               Product config loader
├── config.py                Environment-based configuration
├── templates/
│   ├── index.html           Chat UI
│   └── admin.html           Admin panel
├── products/
│   └── example.json         Product config template — copy and customise
├── tools/
│   └── README.md            Guide for writing project-specific extractors
├── data/
│   └── sample_tickets.csv   25 sample tickets to test with
├── db/                      ChromaDB vector store  (auto-created, git-ignored)
└── sessions.db              Conversation history   (auto-created, git-ignored)
```
