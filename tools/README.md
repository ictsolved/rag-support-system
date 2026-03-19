# tools/

This directory is for **project-specific data extraction scripts** — they are git-ignored and not part of the core framework.

## Purpose

The core system ingests data via CSV, JSON, or JSONL files. If your data lives in a database or a proprietary format, you write a one-off extraction script here to produce an ingest-ready file.

## Expected output format

Your script should produce a CSV (or JSON/JSONL) with these columns:

| Column | Required | Description |
|---|---|---|
| `ticket_id` | Yes | Unique ID for the document |
| `title` | Yes | Short summary |
| `description` | Yes | Full issue description |
| `resolution` | Yes | How it was resolved |
| `category` | No | e.g. Finance, HR, Inventory |
| `module` | No | e.g. Payroll, Inventory, Reports |
| `priority` | No | Low / Medium / High / Critical |
| `status` | No | Open / Resolved / Closed |
| `created_at` | No | Date of the ticket |
| `tags` | No | Comma-separated keywords |

Column names in your file can differ — set `column_map` in your product config to remap them.

## Example

```python
# tools/extract_from_mydb.py
import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine("mysql+pymysql://user:pass@host/db")

df = pd.read_sql("""
    SELECT id AS ticket_id, subject AS title, body AS description,
           resolution, category, module, priority, status, created_at
    FROM support_tickets
    WHERE status = 'Closed'
""", engine)

df.to_csv("data/my_tickets.csv", index=False)
print(f"Exported {len(df)} tickets")
```

Then ingest:
```bash
PRODUCT=my_product python main.py
# Upload data/my_tickets.csv via admin panel → Import
```
