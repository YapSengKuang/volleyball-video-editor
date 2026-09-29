from __future__ import annotations

import os
import subprocess
import tempfile
import zipfile

from config import EXPORT_TIMEOUT_SECONDS, FFMPEG_THREADS

from analyze import has_audio_stream


def _run(cmd: list[str], timeout: int) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Export timed out.") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace")[-800:]
        raise RuntimeError(detail or "ffmpeg failed during export.")


def export_keeps(
    source: str,
    keeps: list[tuple[float, float]],
    zip_path: str,
    merged_path: str,
    on_progress,
) -> None:
    if not keeps:
        raise RuntimeError("Nothing is marked to keep.")
    audio = has_audio_stream(source)
    total = len(keeps)
    with tempfile.TemporaryDirectory(prefix="clips-") as tmp:
        names: list[str] = []
        for index, (start, end) in enumerate(keeps):
            duration = max(0.1, end - start)
            name = f"rally-{index + 1:03d}.mp4"
            clip = os.path.join(tmp, name)
            cmd = [
                "ffmpeg",
                "-y",
                "-threads",
                str(FFMPEG_THREADS),
                "-ss",
                f"{start:.3f}",
                "-i",
                source,
                "-t",
                f"{duration:.3f}",
                "-threads",
                str(FFMPEG_THREADS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
            if audio:
                cmd += ["-c:a", "aac", "-b:a", "128k"]
            else:
                cmd += ["-an"]
            cmd.append(clip)
            _run(cmd, timeout=min(EXPORT_TIMEOUT_SECONDS, max(90, int(duration * 20))))
            names.append(name)
            if on_progress:
                on_progress((index + 1) / total)
        if on_progress:
            on_progress(0.98, "Joining the full game")
        listing = os.path.join(tmp, "list.txt")
        with open(listing, "w", encoding="utf-8") as handle:
            for name in names:
                handle.write(f"file '{os.path.join(tmp, name)}'\n")
        _run(
            [
                "ffmpeg",
                "-y",
                "-threads",
                str(FFMPEG_THREADS),
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                listing,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                merged_path,
            ],
            timeout=600,
        )
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in names:
                archive.write(os.path.join(tmp, name), name)
