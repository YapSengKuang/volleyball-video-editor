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


class StepClock:
    """Estimates remaining time from how fast the current step is moving."""

    def __init__(self) -> None:
        self.phase = ""
        self.started = 0.0

    def remaining(self, phase: str, fraction: float) -> float | None:
        now = time.time()
        if phase != self.phase:
            self.phase = phase
            self.started = now
            return None
        if fraction < 0.04:
            return None
        elapsed = now - self.started
        if elapsed < 3:
            return None
        return max(0.0, elapsed * (1.0 - fraction) / fraction)


def analyze_job(conn, job) -> None:
    folder = _work_path(str(job["id"]))
    source = os.path.join(folder, "source")
    try:
        usage = shutil.disk_usage(WORK_DIR)
        needed = int(job["size_bytes"]) * 2 + 1024**3
        if usage.free < needed:
            raise RuntimeError("Not enough free disk to process this game. It was removed.")
        set_progress(conn, job["id"], 0.02, "Checking the file")
        download(job["source_key"], source)
        clock = StepClock()

        def on_progress(fraction: float, phase: str = "Finding rallies", step: float | None = None) -> None:
            key = phase.split(" — ")[0]
            eta = clock.remaining(key, step if step is not None else fraction)
            # The ball scan takes about ten times as long as the preview.
            if key == "Compressing a preview" and eta is not None:
                eta = eta * 11
            set_progress(conn, job["id"], min(0.98, fraction), phase, eta)

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
        set_progress(conn, job["id"], 0.02, "Preparing the original video")
        download(job["source_key"], source)
        clock = StepClock()

        def on_progress(fraction: float, phase: str | None = None) -> None:
            done = int(fraction * len(keeps))
            label = phase or f"Exporting clip {min(len(keeps), max(1, done))} of {len(keeps)}"
            key = "Joining the full game" if label.startswith("Joining") else "Exporting clips"
            eta = clock.remaining(key, fraction)
            set_progress(conn, job["id"], min(0.95, fraction), label, eta)

        export_keeps(source, keeps, dest, merged, on_progress)
        output_key = upload(dest, f"outputs/{job['id']}.zip", "application/zip")
        merged_key = upload(merged, f"outputs/{job['id']}-full.mp4", "video/mp4")
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
    conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS eta_seconds DOUBLE PRECISION")
    conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS progress DOUBLE PRECISION NOT NULL DEFAULT 0")
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
