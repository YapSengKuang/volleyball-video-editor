"""Smoothing, thresholding, static-speck rejection, and the shared state machine.

These functions do not read or write files.
"""

from __future__ import annotations

import numpy as np


def hold_pulses(signal: np.ndarray, seconds: int) -> np.ndarray:
    """Keep each positive sample high for `seconds` samples after it starts."""
    values = np.asarray(signal, dtype=np.float64)
    if values.size == 0 or seconds <= 1:
        return values.copy()
    held = values.copy()
    length = len(values)
    for index, value in enumerate(values):
        if value <= 0:
            continue
        stop = min(length, index + int(seconds))
        held[index:stop] = np.maximum(held[index:stop], value)
    return held


def onset_indices(ratios: np.ndarray, percentile: float = 92.0, jump: float = 1.4) -> list[int]:
    """Frames where the band ratio jumps and is among the loudest peaks."""
    data = np.asarray(ratios, dtype=np.float64)
    if data.size < 2:
        return []
    floor = float(np.percentile(data, percentile))
    found = []
    for index in range(1, len(data)):
        previous = data[index - 1]
        if data[index] > 0 and data[index] >= floor and (previous <= 1e-8 or data[index] >= previous * jump):
            found.append(index)
    return found


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded moving average. Window is in samples (seconds, at 1 Hz)."""
    values = np.asarray(signal, dtype=np.float64)
    if values.size == 0 or window <= 1:
        return values.copy()
    window = int(window)
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def otsu_threshold(signal: np.ndarray) -> float:
    """Otsu split of a 1-D signal. Returns the threshold value itself."""
    data = np.asarray(signal, dtype=np.float64)
    if data.size == 0:
        return 0.0
    lo = float(data.min())
    hi = float(data.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(data, bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return lo
    prob = hist / total
    omega = np.cumsum(prob)
    centers = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(prob * centers)
    mu_t = float(mu[-1])
    between = np.zeros_like(prob)
    denom = omega * (1.0 - omega)
    valid = denom > 0
    between[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]
    return float(centers[int(np.argmax(between))])


def normalise_percentile(signal: np.ndarray, low: float = 5, high: float = 95) -> np.ndarray:
    data = np.asarray(signal, dtype=np.float64)
    if data.size == 0:
        return data.copy()
    lo = float(np.percentile(data, low))
    hi = float(np.percentile(data, high))
    if hi <= lo:
        return np.zeros_like(data)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def runs_above(signal: np.ndarray, origin: float, threshold: float, bin_s: float = 1.0) -> list[tuple[float, float]]:
    raw: list[tuple[float, float]] = []
    start = 0.0
    end = 0.0
    open_run = False
    for index, value in enumerate(np.asarray(signal, dtype=np.float64)):
        t0 = origin + index * bin_s
        t1 = t0 + bin_s
        if value >= threshold:
            if not open_run:
                open_run = True
                start = t0
            end = t1
        elif open_run:
            raw.append((start, end))
            open_run = False
    if open_run:
        raw.append((start, end))
    return raw


def merge_gaps(segments: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    if not segments:
        return []
    merged: list[list[float]] = [[segments[0][0], segments[0][1]]]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(item[0], item[1]) for item in merged]


def segment_signal(
    signal: np.ndarray,
    origin: float,
    threshold: float,
    min_dur: float,
    merge_gap: float,
    pad: float,
    limit_start: float,
    limit_end: float,
) -> list[tuple[float, float]]:
    """Runs above threshold, merge gaps, drop short runs, pad, merge overlaps."""
    raw = runs_above(signal, origin, threshold)
    merged = merge_gaps(raw, merge_gap)
    kept = [(start, end) for start, end in merged if end - start >= min_dur]
    padded = [
        (max(limit_start, start - pad), min(limit_end, end + pad))
        for start, end in kept
        if min(limit_end, end + pad) - max(limit_start, start - pad) > 0
    ]
    return merge_gaps(padded, 0.0)


def _cell(x: float, y: float, grid: int) -> tuple[int, int]:
    return int(x) // grid, int(y) // grid


def fired_cells(points: list[tuple[float, float]], grid: int = 32) -> set[tuple[int, int]]:
    """A cell fires when a candidate sits in it or in one of its 8 neighbours."""
    cells: set[tuple[int, int]] = set()
    for x, y in points:
        cx, cy = _cell(x, y, grid)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cells.add((cx + dx, cy + dy))
    return cells


def suppress_static(
    candidates: list[tuple[int, float, float]],
    n_frames: int,
    fps: float,
    window_s: float = 20.0,
    max_fraction: float = 0.25,
    grid: int = 32,
) -> list[tuple[int, float, float]]:
    """Drop candidates whose grid cell fires on too much of a sliding window.

    candidates are (frame_index, x_px, y_px). A stationary speck is removed.
    A candidate that only visits a cell briefly is kept.
    """
    if n_frames <= 0 or not candidates:
        return []
    by_frame: dict[int, list[tuple[float, float]]] = {}
    for frame, x, y in candidates:
        by_frame.setdefault(frame, []).append((x, y))
    frame_cells = {frame: fired_cells(points, grid) for frame, points in by_frame.items()}
    half = max(1, int(round(window_s * fps)) // 2)
    kept: list[tuple[int, float, float]] = []
    last = n_frames - 1
    for frame, x, y in candidates:
        cell = _cell(x, y, grid)
        lo = max(0, frame - half)
        hi = min(last, frame + half)
        total = hi - lo + 1
        fired = sum(1 for index in range(lo, hi + 1) if cell in frame_cells.get(index, ()))
        if total > 0 and fired / total <= max_fraction:
            kept.append((frame, x, y))
    return kept


def temporal_support(
    frame_indices: list[int],
    fps: float,
    radius_s: float = 1.0,
    minimum: int = 3,
) -> set[int]:
    """Keep a frame only when enough surviving hits sit within +/- radius_s."""
    ordered = sorted(set(frame_indices))
    if not ordered:
        return set()
    radius = radius_s * fps
    kept: set[int] = set()
    left = 0
    for right, frame in enumerate(ordered):
        while ordered[left] < frame - radius:
            left += 1
        near = right
        while near + 1 < len(ordered) and ordered[near + 1] <= frame + radius:
            near += 1
        if near - left + 1 >= minimum:
            kept.add(frame)
    return kept
