"""Rally clips from motion inside the marked court, with sound as a feature.

Audio is never a rule by itself: a whistle on the next court can be loud.
The court polygon drops background games. A small model, trained on labeled
matches, decides each second. Post-processing favors keeping a rally.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from analyze import (
    SAMPLE_RATE,
    WHISTLE_HIGH_HZ,
    WHISTLE_HOP,
    WHISTLE_LOW_HZ,
    WHISTLE_WIN,
    iter_audio,
    order_corners,
)

FEATURE_NAMES = (
    "motion_mean",
    "motion_p90",
    "motion_active",
    "motion_spread",
    "rms",
    "flux",
    "band_ratio",
    "motion_delta",
    "rms_delta",
)
WINDOW = 4  # seconds on each side of the current second
ENTER = 0.55
LEAVE = 0.40
MIN_DUR = 3.0
MERGE_GAP = 2.5
PRE_PAD = 2.0
POST_PAD = 2.0
MODEL_PATH = Path(__file__).resolve().parent / "models" / "rally_logit.json"


def _mask(height: int, width: int, corners: list) -> np.ndarray:
    poly = order_corners(corners)
    if float(np.max(poly)) <= 1.5:
        poly = poly * np.array([width - 1, height - 1])
    yy, xx = np.mgrid[0:height, 0:width]
    inside = np.zeros((height, width), dtype=bool)
    count = len(poly)
    for index in range(count):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % count]
        cond = (y1 > yy) != (y2 > yy)
        cross = (x2 - x1) * (yy - y1) / (y2 - y1 + 1e-9) + x1
        inside ^= cond & (xx < cross)
    if int(inside.sum()) < 16:
        return np.ones((height, width), dtype=bool)
    return inside


def motion_features(path: str, corners: list, duration: float) -> np.ndarray:
    """Per-second motion inside the court. Columns: mean, p90, active fraction."""
    import subprocess

    width, height = 320, 180
    mask = _mask(height, width, corners)
    command = [
        "ffmpeg", "-v", "error", "-t", f"{duration:.1f}", "-i", path,
        "-vf", f"fps=4,scale={width}:{height},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    frame_size = width * height
    previous = None
    buckets: list[list[tuple[np.ndarray, float]]] = [[] for _ in range(max(1, int(math.ceil(duration))))]
    frame_index = 0
    try:
        while True:
            raw = process.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8)
            if previous is not None:
                diff = np.abs(frame.astype(np.int16) - previous).reshape(height, width)
                changed = (diff > 12) & mask
                _, cols = np.nonzero(changed)
                spread = float(np.std(cols) / width) if cols.size > 8 else 0.0
                second = min(len(buckets) - 1, int(frame_index / 4))
                buckets[second].append((diff[mask], spread))
            previous = frame
            frame_index += 1
    finally:
        process.stdout.close()
        process.wait(timeout=30)
    rows = np.zeros((len(buckets), 4), dtype=np.float64)
    for index, samples in enumerate(buckets):
        if not samples:
            continue
        stacked = np.concatenate([values for values, _spread in samples])
        rows[index, 0] = float(np.mean(stacked))
        rows[index, 1] = float(np.percentile(stacked, 90))
        rows[index, 2] = float(np.mean(stacked > 12))
        rows[index, 3] = float(np.mean([spread for _values, spread in samples]))
    return rows


def _audio_tracks_clean(path: str, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seconds = max(1, int(math.ceil(duration)))
    rms_sum = np.zeros(seconds)
    rms_count = np.zeros(seconds)
    flux_buckets: list[list[float]] = [[] for _ in range(seconds)]
    ratio_buckets: list[list[float]] = [[] for _ in range(seconds)]
    window = np.hanning(WHISTLE_WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WHISTLE_WIN, 1 / SAMPLE_RATE)
    band = (freqs >= WHISTLE_LOW_HZ) & (freqs <= WHISTLE_HIGH_HZ)
    pending = np.zeros(0, dtype=np.float32)
    sample_at = 0
    previous = None
    try:
        stream = iter_audio(path)
    except Exception:
        return np.zeros(seconds), np.zeros(seconds), np.zeros(seconds)
    for chunk in stream:
        if chunk.size == 0:
            continue
        start_sample = sample_at
        for offset in range(0, chunk.size, SAMPLE_RATE):
            piece = chunk[offset : offset + SAMPLE_RATE]
            second = (start_sample + offset) // SAMPLE_RATE
            if second >= seconds:
                break
            rms_sum[second] += float(np.sqrt(np.mean(np.square(piece))))
            rms_count[second] += 1
        pending = np.concatenate([pending, chunk])
        pending_origin = sample_at - (len(pending) - chunk.size)
        cursor = 0
        limit = len(pending) - WHISTLE_WIN
        while cursor <= limit:
            frame = pending[cursor : cursor + WHISTLE_WIN] * window
            mag = np.abs(np.fft.rfft(frame))
            rest = float(np.mean(mag[~band])) + 1e-8
            second = int((pending_origin + cursor) / SAMPLE_RATE)
            if 0 <= second < seconds:
                ratio_buckets[second].append(float(np.mean(mag[band]) / rest))
                if previous is not None:
                    delta = np.log1p(mag) - previous
                    flux_buckets[second].append(float(delta[delta > 0].sum()))
            previous = np.log1p(mag)
            cursor += WHISTLE_HOP
        sample_at += chunk.size
        pending = pending[cursor:]
    rms = np.divide(rms_sum, np.maximum(rms_count, 1))
    flux = np.array([float(np.percentile(bucket, 95)) if bucket else 0.0 for bucket in flux_buckets])
    ratio = np.array([float(np.percentile(bucket, 95)) if bucket else 0.0 for bucket in ratio_buckets])
    return rms, flux, ratio


def feature_table(path: str, corners: list, duration: float) -> np.ndarray:
    motion = motion_features(path, corners, duration)
    rms, flux, ratio = _audio_tracks_clean(path, duration)
    count = max(len(motion), len(rms))
    table = np.zeros((count, len(FEATURE_NAMES)), dtype=np.float64)
    motion_n = min(len(motion), count)
    audio_n = min(len(rms), count)
    table[:motion_n, 0:4] = motion[:motion_n]
    table[:audio_n, 4] = rms[:audio_n]
    table[:audio_n, 5] = flux[:audio_n]
    table[:audio_n, 6] = ratio[:audio_n]
    table[1:, 7] = np.diff(table[:, 0])
    table[1:, 8] = np.diff(table[:, 4])
    return table


def window_matrix(table: np.ndarray, radius: int = WINDOW) -> np.ndarray:
    padded = np.pad(table, ((radius, radius), (0, 0)), mode="edge")
    rows = []
    for index in range(len(table)):
        rows.append(padded[index : index + 2 * radius + 1].ravel())
    return np.vstack(rows) if rows else np.zeros((0, table.shape[1] * (2 * radius + 1)))


def fit_logit(features: np.ndarray, labels: np.ndarray, steps: int = 500, lr: float = 0.15, l2: float = 0.2) -> dict:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = (features - mean) / std
    bias = np.ones((len(scaled), 1))
    design = np.concatenate([scaled, bias], axis=1)
    weights = np.zeros(design.shape[1])
    target = labels.astype(np.float64)
    for _ in range(steps):
        prob = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30, 30)))
        example_weight = np.where(target > 0.5, 1.25, 1.0)
        grad = design.T @ ((prob - target) * example_weight) / len(target)
        grad[:-1] += l2 * weights[:-1]
        weights -= lr * grad
    return {"mean": mean.tolist(), "std": std.tolist(), "weights": weights.tolist(), "names": list(FEATURE_NAMES), "window": WINDOW}


def predict_proba(features: np.ndarray, model: dict) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    scaled = (features - mean) / std
    design = np.concatenate([scaled, np.ones((len(scaled), 1))], axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30, 30)))


def rallies_from_probs(
    probs: np.ndarray,
    duration: float,
    enter: float = ENTER,
    leave: float = LEAVE,
    min_dur: float = MIN_DUR,
    merge_gap: float = MERGE_GAP,
    pre_pad: float = PRE_PAD,
    post_pad: float = POST_PAD,
) -> list[tuple[float, float]]:
    """Hysteresis, drop short runs, merge small gaps, then pad. Recall is favored."""
    state = False
    raw: list[tuple[float, float]] = []
    start = 0.0
    for index, prob in enumerate(np.asarray(probs, dtype=np.float64)):
        if not state and prob >= enter:
            state = True
            start = float(index)
        elif state and prob < leave:
            raw.append((start, float(index)))
            state = False
    if state:
        raw.append((start, float(len(probs))))
    merged: list[list[float]] = []
    for begin, finish in raw:
        if merged and begin - merged[-1][1] <= merge_gap:
            merged[-1][1] = finish
        else:
            merged.append([begin, finish])
    kept = [(begin, finish) for begin, finish in merged if finish - begin >= min_dur]
    padded = [(max(0.0, begin - pre_pad), min(duration, finish + post_pad)) for begin, finish in kept]
    collapsed: list[list[float]] = []
    for begin, finish in padded:
        if finish - begin < 0.5:
            continue
        if collapsed and begin <= collapsed[-1][1]:
            collapsed[-1][1] = max(collapsed[-1][1], finish)
        else:
            collapsed.append([begin, finish])
    return [(begin, finish) for begin, finish in collapsed]


def load_model(path: Path | None = None) -> dict | None:
    model_path = path or MODEL_PATH
    if not model_path.exists():
        return None
    return json.loads(model_path.read_text(encoding="utf-8"))


def learned_rallies(path: str, corners: list, duration: float, model: dict | None = None) -> list[tuple[float, float]]:
    spec = model or load_model()
    if spec is None or not corners:
        return []
    table = feature_table(path, corners, duration)
    probs = predict_proba(window_matrix(table, int(spec.get("window", WINDOW))), spec)
    return rallies_from_probs(probs, duration)
