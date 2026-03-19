import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL  = os.getenv("LLM_MODEL",  "phi3:mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHROMA_PATH  = os.getenv("CHROMA_PATH", "./db")
# COLLECTION_NAME is the fallback; the active product config overrides it
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "support_tickets")

N_RESULTS    = int(os.getenv("N_RESULTS",    "3"))    # tickets retrieved per query
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.35"))  # below this → LLM gets "no match"
EXCERPT_LEN  = int(os.getenv("EXCERPT_LEN",  "250"))  # chars of each ticket shown to LLM
HISTORY_MSGS = int(os.getenv("HISTORY_MSGS", "6"))    # last N messages kept in context
NUM_PREDICT  = int(os.getenv("NUM_PREDICT",  "512"))  # max tokens the LLM can generate

# Which product config to load from products/{PRODUCT}.json
PRODUCT = os.getenv("PRODUCT", "erp")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")   # empty = no auth required
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")   # empty = no auth required (dev only)
SESSIONS_DB    = os.getenv("SESSIONS_DB", "./sessions.db")
DATA_DIR    = os.getenv("DATA_DIR",    "./data")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
