from __future__ import annotations

import time
import uuid

import psycopg
from psycopg.rows import dict_row

from config import DATABASE_URL


def connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def wait_for_db():
    last_error: Exception | None = None
    for _ in range(30):
        try:
            conn = connect()
            conn.execute("SELECT 1")
            return conn
        except Exception as exc:  # noqa: BLE001 — startup retry
            last_error = exc
            time.sleep(1)
    raise SystemExit(f"database not ready: {last_error}")


def claim(conn, from_status: str, to_status: str, phase: str):
    row = conn.execute(
        """
        UPDATE jobs
        SET status = %s, progress = 0, phase = %s, error = NULL, updated_at = now()
        WHERE id = (
          SELECT id FROM jobs
          WHERE status = %s
          ORDER BY created_at
          FOR UPDATE SKIP LOCKED
          LIMIT 1
        )
        RETURNING *
        """,
        (to_status, phase, from_status),
    ).fetchone()
    conn.commit()
    return row


def set_progress(conn, job_id: str, progress: float, phase: str, eta_seconds: float | None = None) -> None:
    conn.execute(
        """
        UPDATE jobs
        SET progress = %s, phase = %s, eta_seconds = %s, updated_at = now()
        WHERE id = %s
        """,
        (progress, phase, eta_seconds, job_id),
    )
    conn.commit()


def fail_job(conn, job, message: str) -> None:
    from storage import abort_upload, delete

    delete(job.get("source_key"))
    delete(job.get("output_key"))
    delete(job.get("merged_key"))
    abort_upload(job.get("source_key"), job.get("upload_id"))
    conn.execute(
        """
        UPDATE jobs
        SET status = 'failed',
            error = %s,
            phase = 'Failed',
            size_bytes = 0,
            output_bytes = NULL,
            merged_bytes = NULL,
            progress = 0,
            updated_at = now(),
            finished_at = now(),
            expires_at = now() + interval '1 hour'
        WHERE id = %s
        """,
        (message[:500], job["id"]),
    )
    conn.commit()


def replace_segments(conn, job_id: str, segments: list[dict]) -> None:
    conn.execute("DELETE FROM segments WHERE job_id = %s", (job_id,))
    for position, segment in enumerate(segments):
        conn.execute(
            """
            INSERT INTO segments (id, job_id, start_seconds, end_seconds, keep, origin, position)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                job_id,
                segment["start"],
                segment["end"],
                segment["keep"],
                segment.get("origin", "auto"),
                position,
            ),
        )


def load_segments(conn, job_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT start_seconds, end_seconds, keep
        FROM segments
        WHERE job_id = %s
        ORDER BY position
        """,
        (job_id,),
    ).fetchall()
    return [
        {
            "start": float(row["start_seconds"]),
            "end": float(row["end_seconds"]),
            "keep": bool(row["keep"]),
        }
        for row in rows
    ]
