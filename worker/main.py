from __future__ import annotations

import os
import shutil
import sys
import time
import traceback

import json

from analyze import DurationError, propose_for_file
from config import RETENTION_HOURS, WORK_DIR
from db import claim, fail_job, load_segments, replace_segments, set_progress, wait_for_db
from export import export_keeps
from storage import abort_upload, delete, download, ensure_bucket, upload


def _work_path(job_id: str) -> str:
    path = os.path.join(WORK_DIR, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def _free_work(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def analyze_job(conn, job) -> None:
    folder = _work_path(str(job["id"]))
    source = os.path.join(folder, "source")
    try:
        usage = shutil.disk_usage(WORK_DIR)
        needed = int(job["size_bytes"]) * 2 + 1024**3
        if usage.free < needed:
            raise RuntimeError("Not enough free disk to process this game. It was removed.")
        set_progress(conn, job["id"], 0.05, "Checking duration")
        download(job["source_key"], source)

        def on_progress(fraction: float, phase: str = "Finding rallies") -> None:
            set_progress(conn, job["id"], min(0.98, 0.08 + 0.9 * fraction), phase)

        raw_corners = job.get("court_corners")
        corners = json.loads(raw_corners) if raw_corners else None
        segments, warning, duration = propose_for_file(source, corners=corners, on_progress=on_progress)
        with conn.transaction():
            replace_segments(conn, job["id"], segments)
            conn.execute(
                """
                UPDATE jobs
                SET status = 'ready',
                    duration_seconds = %s,
                    warning = %s,
                    progress = 1,
                    phase = 'Review the rallies',
                    finished_at = now(),
                    expires_at = now() + (%s * interval '1 hour'),
                    updated_at = now()
                WHERE id = %s
                """,
                (duration, warning, RETENTION_HOURS, job["id"]),
            )
    except DurationError as exc:
        fail_job(conn, job, str(exc))
    except Exception as exc:  # noqa: BLE001 — record and free the file
        traceback.print_exc()
        fail_job(conn, job, str(exc) or "Analysis failed.")
    finally:
        _free_work(folder)


def export_job(conn, job) -> None:
    folder = _work_path(str(job["id"]))
    source = os.path.join(folder, "source")
    dest = os.path.join(folder, "rallies.zip")
    merged = os.path.join(folder, "full-game.mp4")
    try:
        segments = load_segments(conn, job["id"])
        keeps = [(seg["start"], seg["end"]) for seg in segments if seg["keep"]]
        if not keeps:
            raise RuntimeError("Nothing is marked to keep.")
        set_progress(conn, job["id"], 0.02, "Preparing export")
        download(job["source_key"], source)

        def on_progress(fraction: float, phase: str | None = None) -> None:
            done = int(fraction * len(keeps))
            set_progress(
                conn,
                job["id"],
                min(0.95, fraction),
                phase or f"Exporting clip {min(len(keeps), max(1, done))} of {len(keeps)}",
            )

        export_keeps(source, keeps, dest, merged, on_progress)
        output_key = f"outputs/{job['id']}.zip"
        merged_key = f"outputs/{job['id']}-full.mp4"
        upload(dest, output_key, "application/zip")
        upload(merged, merged_key, "video/mp4")
        output_bytes = os.path.getsize(dest)
        merged_bytes = os.path.getsize(merged)
        conn.execute(
            """
            UPDATE jobs
            SET status = 'done',
                output_key = %s,
                output_bytes = %s,
                merged_key = %s,
                merged_bytes = %s,
                progress = 1,
                phase = 'Clips are ready',
                error = NULL,
                finished_at = now(),
                expires_at = now() + (%s * interval '1 hour'),
                updated_at = now()
            WHERE id = %s
            """,
            (output_key, output_bytes, merged_key, merged_bytes, RETENTION_HOURS, job["id"]),
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        conn.execute(
            """
            UPDATE jobs
            SET status = 'ready',
                error = %s,
                phase = 'Export failed',
                progress = 0,
                updated_at = now()
            WHERE id = %s
            """,
            ((str(exc) or "Export failed.")[:500], job["id"]),
        )
        conn.commit()
    finally:
        _free_work(folder)


def recover(conn) -> None:
    # A restarted worker is the only ffmpeg process, so a job left in
    # analyzing or exporting was abandoned mid-run.
    conn.execute(
        """
        UPDATE jobs
        SET status = 'queued', phase = 'Waiting to analyze', progress = 0, updated_at = now()
        WHERE status = 'analyzing'
        """
    )
    conn.execute(
        """
        UPDATE jobs
        SET status = 'export_queued', phase = 'Waiting to export', progress = 0, updated_at = now()
        WHERE status = 'exporting'
        """
    )
    conn.commit()


def cleanup(conn) -> None:
    rows = conn.execute(
        """
        SELECT id, source_key, output_key, merged_key, upload_id, status
        FROM jobs
        WHERE (expires_at IS NOT NULL AND expires_at < now())
           OR (status = 'uploading' AND created_at < now() - interval '2 hours')
        """
    ).fetchall()
    for row in rows:
        abort_upload(row["source_key"], row["upload_id"])
        delete(row["source_key"])
        delete(row["output_key"])
        delete(row.get("merged_key"))
        conn.execute("DELETE FROM jobs WHERE id = %s", (row["id"],))
    conn.execute(
        "DELETE FROM rate_limits WHERE window_start < now() - interval '2 days'"
    )
    conn.commit()


def main() -> None:
    once = "--once" in sys.argv
    os.makedirs(WORK_DIR, exist_ok=True)
    ensure_bucket()
    with open("/tmp/ready", "w", encoding="utf-8") as handle:
        handle.write("ok\n")
    conn = wait_for_db()
    conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS merged_key TEXT")
    conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS merged_bytes BIGINT")
    conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS court_corners TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS corrections (
          id UUID PRIMARY KEY,
          job_id UUID NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          segments JSONB NOT NULL
        )
        """
    )
    conn.commit()
    recover(conn)
    while True:
        cleanup(conn)
        job = claim(conn, "queued", "analyzing", "Checking the file")
        if job:
            print(f"analyzing {job['id']}", flush=True)
            analyze_job(conn, job)
            if once:
                return
            continue
        job = claim(conn, "export_queued", "exporting", "Waiting to export")
        if job:
            print(f"exporting {job['id']}", flush=True)
            export_job(conn, job)
            if once:
                return
            continue
        if once:
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
