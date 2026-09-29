"""Rally detection for a fixed camera volleyball game.

A rally is sustained play inside the marked court. It starts about two seconds
before the action and ends about three seconds after the court goes quiet.
Other courts in the frame are ignored once their corners are outside the box.
Audio can support a quiet contact. It never creates a rally by itself.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from collections.abc import Iterable

import numpy as np

from config import ANALYZE_TIMEOUT_SECONDS, FFMPEG_THREADS, MAX_DURATION_SECONDS, PROXY_FPS, PROXY_HEIGHT

WHISTLE_LOW_HZ = 2500
WHISTLE_HIGH_HZ = 4500
WHISTLE_WIN = 1024
WHISTLE_HOP = 256
WHISTLE_MIN_GAP = 3.0
PAD_BEFORE = 2.0
PAD_AFTER = 3.0
MIN_RALLY = 3.0
DEAD_RUN = 2
SAMPLE_RATE = 16000


class DurationError(Exception):
    def __init__(self, duration: float, limit: float) -> None:
        super().__init__(
            f"This game is {duration / 3600:.1f} hours. "
            f"Games over {limit / 3600:.1f} hours are deleted without processing."
        )
        self.duration = duration
        self.limit = limit


def assert_duration(duration: float, limit: float) -> None:
    if duration > limit:
        raise DurationError(duration, limit)


def probe_duration(path: str) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-800:])
    return float(proc.stdout.decode().strip())


def has_video(path: str) -> bool:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    return proc.returncode == 0 and b"video" in proc.stdout


def has_audio_stream(path: str) -> bool:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    return proc.returncode == 0 and b"audio" in proc.stdout


def _band_mask(sample_rate: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(WHISTLE_WIN, 1 / sample_rate)
    return (freqs >= WHISTLE_LOW_HZ) & (freqs <= WHISTLE_HIGH_HZ)


def band_energy(samples: Iterable[np.ndarray], sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    window = np.hanning(WHISTLE_WIN).astype(np.float32)
    band = _band_mask(sample_rate)
    energies: list[float] = []
    pending = np.zeros(0, dtype=np.float32)
    for chunk in samples:
        if chunk.size == 0:
            continue
        pending = np.concatenate([pending, np.asarray(chunk, dtype=np.float32)])
        pos = 0
        while pos + WHISTLE_WIN <= len(pending):
            frame = pending[pos : pos + WHISTLE_WIN] * window
            spec = np.abs(np.fft.rfft(frame))
            energies.append(float(np.mean(spec[band])))
            pos += WHISTLE_HOP
        pending = pending[pos:]
    if not energies:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(energies, dtype=np.float64)


def peaks_from_energy(energy: np.ndarray, hop_seconds: float) -> list[float]:
    if energy.size == 0:
        return []
    peak = float(np.max(energy))
    if peak <= 1e-6:
        return []
    median = float(np.median(energy))
    mad = float(np.median(np.abs(energy - median))) + 1e-12
    floor = max(median + 8 * mad, 0.2 * peak)
    times: list[float] = []
    index = 0
    while index < len(energy):
        if energy[index] < floor:
            index += 1
            continue
        end = index
        while end < len(energy) and energy[end] >= floor:
            end += 1
        duration = (end - index) * hop_seconds
        center = ((index + end) / 2) * hop_seconds
        if 0.05 <= duration <= 1.5 and (not times or center - times[-1] >= WHISTLE_MIN_GAP):
            times.append(center)
        index = end
    return times


def detect_whistles(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[float]:
    energy = band_energy([samples], sample_rate)
    return peaks_from_energy(energy, WHISTLE_HOP / sample_rate)


def iter_audio(path: str):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        str(FFMPEG_THREADS),
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    started = time.time()
    try:
        while True:
            if time.time() - started > ANALYZE_TIMEOUT_SECONDS:
                proc.kill()
                raise RuntimeError("Reading the soundtrack timed out.")
            buf = proc.stdout.read(SAMPLE_RATE * 4 * 10)
            if not buf:
                break
            yield np.frombuffer(buf, dtype=np.float32).copy()
    finally:
        if proc.stdout:
            proc.stdout.close()
        code = proc.wait(timeout=30)
    if code != 0:
        raise RuntimeError("Could not read audio.")


def whistles_from_file(path: str) -> tuple[list[float], bool]:
    if not has_audio_stream(path):
        return [], False
    try:
        energy = band_energy(iter_audio(path), SAMPLE_RATE)
    except RuntimeError:
        return [], False
    return peaks_from_energy(energy, WHISTLE_HOP / SAMPLE_RATE), True


def smooth(scores: np.ndarray) -> np.ndarray:
    if scores.size < 3:
        return scores
    smoothed = scores.copy()
    smoothed[1:-1] = (scores[:-2] + scores[1:-1] + scores[2:]) / 3
    return smoothed


def bucket_scores(diffs: list[float], fps: float, duration: float) -> np.ndarray:
    seconds = max(1, int(math.ceil(duration)))
    buckets: list[list[float]] = [[] for _ in range(seconds)]
    for index, value in enumerate(diffs):
        stamp = (index + 1) / fps
        slot = min(seconds - 1, int(stamp))
        buckets[slot].append(value)
    scores = np.zeros(seconds, dtype=np.float64)
    for index, values in enumerate(buckets):
        if values:
            scores[index] = float(np.mean(values))
    return smooth(scores)


def order_corners(points: list) -> np.ndarray:
    pairs = []
    for point in points:
        if isinstance(point, dict):
            pairs.append((float(point["x"]), float(point["y"])))
        else:
            pairs.append((float(point[0]), float(point[1])))
    pts = np.asarray(pairs, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError("Mark all four corners of the court.")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def court_mask(height: int, width: int, corners: list) -> np.ndarray:
    """True for pixels inside the court quadrilateral. Corners are 0–1."""
    poly = order_corners(corners) * np.array([width, height])
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = poly.mean(axis=0)
    sign = 0.0
    edges = []
    for index in range(4):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % 4]
        cross = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
        edges.append(cross)
        sign += (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
    want_positive = sign >= 0
    mask = np.ones((height, width), dtype=bool)
    for cross in edges:
        mask &= cross >= -0.5 if want_positive else cross <= 0.5
    if int(mask.sum()) < 16:
        return np.ones((height, width), dtype=bool)
    return mask


def motion_scores(path: str, duration: float, corners: list | None = None, on_progress=None) -> np.ndarray:
    width, height = 192, 108
    frame_size = width * height
    expected = max(1, int(duration * 8))
    mask = court_mask(height, width, corners) if corners else None
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        str(FFMPEG_THREADS),
        "-i",
        path,
        "-vf",
        f"fps=8,scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    diffs: list[float] = []
    prev: np.ndarray | None = None
    started = time.time()
    try:
        while True:
            if time.time() - started > ANALYZE_TIMEOUT_SECONDS:
                proc.kill()
                raise RuntimeError("Reading court motion timed out.")
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            if mask is not None:
                frame = frame.reshape(height, width)
                frame = frame[mask]
            if prev is not None:
                diffs.append(float(np.mean(np.abs(frame - prev))))
            prev = frame
            if on_progress and len(diffs) % 40 == 0:
                on_progress(min(0.95, len(diffs) / expected))
    finally:
        if proc.stdout:
            proc.stdout.close()
        code = proc.wait(timeout=30)
    if code != 0:
        raise RuntimeError("Could not read video.")
    return bucket_scores(diffs, 8, duration)


def motion_threshold(scores: np.ndarray) -> float:
    if scores.size == 0:
        return 1e9
    low = float(np.percentile(scores, 20))
    peak = float(np.percentile(scores, 98))
    if peak < 4:
        return 1e9
    return low + 0.35 * (peak - low)


def activity_flags(scores: np.ndarray, audio: np.ndarray | None = None) -> np.ndarray:
    """A second is live when the court is moving. A sound only helps a quiet contact."""
    thresh = motion_threshold(scores)
    active = scores >= thresh
    if audio is None or audio.size != scores.size or float(np.max(audio)) <= 1e-6:
        return active
    loud = audio >= max(float(np.percentile(audio, 85)), 0.35 * float(np.max(audio)))
    return active | ((scores >= thresh * 0.55) & loud)


def rallies_from_activity(active: np.ndarray, duration: float) -> list[tuple[float, float]]:
    """Stay in a rally until the court has been quiet for a couple of seconds."""
    ranges: list[tuple[float, float]] = []
    start: int | None = None
    low = 0
    for index, flag in enumerate(active):
        if start is None:
            if flag:
                start = index
                low = 0
            continue
        if flag:
            low = 0
            continue
        low += 1
        if low >= DEAD_RUN:
            end = index - DEAD_RUN + 1
            if end - start >= MIN_RALLY:
                ranges.append((float(start), float(end)))
            start = None
            low = 0
    if start is not None:
        end = len(active) - low
        if end - start >= MIN_RALLY:
            ranges.append((float(start), float(min(duration, end))))
    return ranges


def apply_pads(ranges: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """Two seconds before the serve and three after the ball dies, without crossing the next rally."""
    padded: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(ranges):
        if index == 0:
            before = min(PAD_BEFORE, start)
        else:
            midpoint = (ranges[index - 1][1] + start) / 2
            before = min(PAD_BEFORE, max(0.0, start - midpoint))
        if index == len(ranges) - 1:
            after = min(PAD_AFTER, max(0.0, duration - end))
        else:
            midpoint = (end + ranges[index + 1][0]) / 2
            after = min(PAD_AFTER, max(0.0, midpoint - end))
        padded.append((max(0.0, start - before), min(duration, end + after)))
    return padded


def to_clips(keeps: list[tuple[float, float]], duration: float) -> list[dict]:
    clips: list[dict] = []
    for start, end in keeps:
        start = max(0.0, start)
        end = min(duration, end)
        if end - start < 0.5:
            continue
        clips.append({"start": start, "end": end, "keep": True, "origin": "auto"})
    return clips


def propose(
    scores: np.ndarray,
    duration: float,
    audio_energy: np.ndarray | None = None,
    had_audio: bool = False,
    used_court: bool = False,
) -> tuple[list[dict], str | None]:
    warning = None
    if not had_audio:
        warning = (
            "No soundtrack detected. Rallies are based on movement inside the court, "
            "so warmup hitting can still be included."
        )
    elif not used_court:
        warning = (
            "The whole frame was used, so a game on the next court may be included. "
            "Mark your court and find the rallies again."
        )
    raw = rallies_from_activity(activity_flags(scores, audio_energy), duration)
    clips = to_clips(apply_pads(raw, duration), duration)
    if not clips and warning is None:
        warning = "No sustained rally was found inside the court. Add any that were missed."
    return clips, warning


def build_proxy(source: str, dest: str, duration: float, on_progress=None) -> None:
    """One fast, small encode. Rally times are measured on this, not the full file."""
    audio = has_audio_stream(source)
    cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        str(FFMPEG_THREADS),
        "-i",
        source,
        "-map",
        "0:v:0",
        "-vf",
        f"scale=-2:{PROXY_HEIGHT},fps={PROXY_FPS:g}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "32",
        "-pix_fmt",
        "yuv420p",
    ]
    if audio:
        cmd += ["-map", "0:a:0", "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000"]
    else:
        cmd += ["-an"]
    cmd += ["-progress", "pipe:2", "-nostats", dest]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stderr is not None
    started = time.time()
    try:
        for raw in proc.stderr:
            if time.time() - started > ANALYZE_TIMEOUT_SECONDS:
                proc.kill()
                raise RuntimeError("Compressing the preview timed out.")
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("out_time_ms=") or duration <= 0 or not on_progress:
                continue
            try:
                microseconds = int(line.split("=", 1)[1])
            except ValueError:
                continue
            on_progress(min(0.99, (microseconds / 1_000_000) / duration))
    finally:
        proc.stderr.close()
        code = proc.wait(timeout=30)
    if code != 0:
        raise RuntimeError("Could not compress a preview of this game.")


def audio_energy(path: str, duration: float) -> tuple[np.ndarray | None, bool]:
    if not has_audio_stream(path):
        return None, False
    seconds = max(1, int(math.ceil(duration)))
    totals = np.zeros(seconds, dtype=np.float64)
    counts = np.zeros(seconds, dtype=np.float64)
    try:
        cursor = 0
        for chunk in iter_audio(path):
            if chunk.size == 0:
                continue
            energy = np.square(chunk.astype(np.float64))
            for offset in range(0, energy.size, SAMPLE_RATE):
                second = cursor // SAMPLE_RATE
                if second >= seconds:
                    break
                piece = energy[offset : offset + SAMPLE_RATE]
                totals[second] += float(piece.sum())
                counts[second] += piece.size
                cursor += piece.size
    except RuntimeError:
        return None, False
    if float(counts.sum()) == 0:
        return None, False
    safe = np.maximum(counts, 1)
    return np.sqrt(totals / safe), True


def propose_for_file(
    path: str, corners: list | None = None, on_progress=None
) -> tuple[list[dict], str | None, float]:
    duration = probe_duration(path)
    assert_duration(duration, MAX_DURATION_SECONDS)
    if not has_video(path):
        raise RuntimeError("This file has no video track.")
    proxy = f"{path}.proxy.mkv"

    def report(overall: float, phase: str, step: float) -> None:
        if on_progress:
            on_progress(overall, phase, step)

    try:
        report(0.02, "Checking duration", 0.0)

        def clock_label(seconds: float) -> str:
            whole = max(0, int(seconds))
            return f"{whole // 60}:{whole % 60:02d}"

        report(0.04, "Compressing a preview", 0.0)
        build_proxy(
            path,
            proxy,
            duration,
            on_progress=lambda fraction: report(
                0.04 + 0.14 * fraction,
                f"Compressing a preview — {clock_label(fraction * duration)} of {clock_label(duration)}",
                fraction,
            ),
        )
        total_frames = max(1, int(duration * PROXY_FPS))
        report(0.18, "Looking for the ball", 0.0)
        ball_clips: list[tuple[float, float]] = []
        try:
            from ball import rallies_from_hits, scan_ball

            hits = scan_ball(
                proxy,
                corners,
                duration,
                on_progress=lambda fraction: report(
                    0.18 + 0.74 * fraction,
                    f"Looking for the ball — {int(fraction * total_frames):,} of {total_frames:,} frames",
                    fraction,
                ),
            )
            ball_clips = rallies_from_hits(hits, duration, corners)
        except Exception:
            import traceback

            traceback.print_exc()
            hits = []
            ball_clips = []
        if ball_clips:
            report(0.96, "Marking the rallies", 1.0)
            warning = None if corners else (
                "The ball was tracked in the whole frame. Mark your court so the next court's ball is left out."
            )
            return to_clips(ball_clips, duration), warning, duration
        report(0.2, "Finding rallies from movement", 0.0)
        scores = motion_scores(
            proxy,
            duration,
            corners=corners,
            on_progress=lambda fraction: report(
                0.2 + 0.7 * fraction,
                "Finding rallies from movement",
                fraction,
            ),
        )
        levels, had_audio = audio_energy(proxy, duration)
        segments, warning = propose(
            scores,
            duration,
            audio_energy=levels,
            had_audio=had_audio,
            used_court=bool(corners),
        )
        if warning:
            warning = "The ball was hard to see, so rallies were based on movement inside the court. " + warning
        else:
            warning = "The ball was hard to see, so rallies were based on movement inside the court."
        return segments, warning, duration
    finally:
        if os.path.exists(proxy):
            os.remove(proxy)
