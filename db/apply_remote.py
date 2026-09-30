"""Apply the app schema to DATABASE_URL_UNPOOLED from an env file. Prints no secrets."""

from pathlib import Path
import sys

import psycopg

env_path, sql_path = sys.argv[1:]
env = {}
for line in Path(env_path).read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    env[key] = value.strip().strip('"')
url = env.get("DATABASE_URL_UNPOOLED") or env.get("DATABASE_URL")
if not url:
    raise SystemExit("missing database url")
if "sslmode=" not in url:
    url += ("&" if "?" in url else "?") + "sslmode=require"
sql = Path(sql_path).read_text(encoding="utf-8")
extra = """
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS merged_key TEXT;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS merged_bytes BIGINT;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS court_corners TEXT;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS eta_seconds DOUBLE PRECISION;
CREATE TABLE IF NOT EXISTS corrections (
  id UUID PRIMARY KEY,
  job_id UUID NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  segments JSONB NOT NULL
);
"""
with psycopg.connect(url) as conn:
    conn.execute(sql)
    conn.execute(extra)
    conn.commit()
    tables = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"
    ).fetchall()
print("schema ok:", ", ".join(row[0] for row in tables))
