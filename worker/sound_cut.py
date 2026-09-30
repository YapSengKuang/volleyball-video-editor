"""Propose rallies from a sharp hit in the whistle band, not from crowd loudness.

A roar lifts every frequency, so the band divided by the rest of the spectrum
stays flat. A whistle or a ball hit jumps. Each jump holds the clip open.
"""

from __future__ import annotations

import math

import numpy as np

from analyze import (
    SAMPLE_RATE,
    WHISTLE_HIGH_HZ,
    WHISTLE_HOP,
    WHISTLE_LOW_HZ,
    WHISTLE_WIN,
    iter_audio,
)

HOLD_SECONDS = 8
ONSET_PERCENTILE = 92.0
ONSET_JUMP = 1.4


def _ratios(path: str) -> tuple[list[float], list[float]]:
    window = np.hanning(WHISTLE_WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WHISTLE_WIN, 1 / SAMPLE_RATE)
    band = (freqs >= WHISTLE_LOW_HZ) & (freqs <= WHISTLE_HIGH_HZ)
    pending = np.zeros(0, dtype=np.float32)
    consumed = 0
    times: list[float] = []
    ratios: list[float] = []
    for chunk in iter_audio(path):
        if chunk.size == 0:
            continue
        pending = np.concatenate([pending, np.asarray(chunk, dtype=np.float32)])
        cursor = 0
        limit = len(pending) - WHISTLE_WIN
        while cursor <= limit:
            mag = np.abs(np.fft.rfft(pending[cursor : cursor + WHISTLE_WIN] * window))
            rest = float(np.mean(mag[~band])) + 1e-8
            ratios.append(float(np.mean(mag[band]) / rest))
            times.append((consumed + cursor + WHISTLE_WIN / 2) / SAMPLE_RATE)
            cursor += WHISTLE_HOP
        consumed += cursor
        pending = pending[cursor:]
    return times, ratios


def _onsets(ratios: list[float]) -> list[int]:
    data = np.asarray(ratios, dtype=np.float64)
    if data.size < 2:
        return []
    floor = float(np.percentile(data, ONSET_PERCENTILE))
    found = []
    for index in range(1, len(data)):
        previous = data[index - 1]
        if data[index] > 0 and data[index] >= floor and (previous <= 1e-8 or data[index] >= previous * ONSET_JUMP):
            found.append(index)
    return found


def _hold(signal: np.ndarray, seconds: int) -> np.ndarray:
    held = signal.copy()
    for index, value in enumerate(signal):
        if value <= 0:
            continue
        stop = min(len(held), index + seconds)
        held[index:stop] = np.maximum(held[index:stop], value)
    return held


def _smooth(signal: np.ndarray, window: int = 3) -> np.ndarray:
    if signal.size == 0 or window <= 1:
        return signal.copy()
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(signal, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def _segments(signal: np.ndarray, duration: float) -> list[tuple[float, float]]:
    raw: list[tuple[float, float]] = []
    start = 0.0
    end = 0.0
    open_run = False
    for index, value in enumerate(signal):
        if value >= 0.5:
            if not open_run:
                open_run = True
                start = float(index)
            end = float(index + 1)
        elif open_run:
            raw.append((start, end))
            open_run = False
    if open_run:
        raw.append((start, end))
    merged: list[list[float]] = []
    for begin, finish in raw:
        if merged and begin - merged[-1][1] <= 4:
            merged[-1][1] = max(merged[-1][1], finish)
        else:
            merged.append([begin, finish])
    kept = [(begin, finish) for begin, finish in merged if finish - begin >= 4]
    padded = [(max(0.0, begin - 2), min(duration, finish + 2)) for begin, finish in kept]
    collapsed: list[list[float]] = []
    for begin, finish in padded:
        if collapsed and begin <= collapsed[-1][1]:
            collapsed[-1][1] = max(collapsed[-1][1], finish)
        else:
            collapsed.append([begin, finish])
    return [(begin, finish) for begin, finish in collapsed if finish - begin >= 0.5]


def sound_rallies(path: str, duration: float, hold: int = HOLD_SECONDS) -> list[tuple[float, float]]:
    """Rally windows opened by a sharp band-ratio jump."""
    times, ratios = _ratios(path)
    if len(ratios) < 2:
        return []
    bins = max(1, int(math.ceil(duration)))
    pulses = np.zeros(bins, dtype=np.float64)
    for index in _onsets(ratios):
        second = int(times[index])
        if 0 <= second < bins:
            pulses[second] = 1.0
    if float(pulses.sum()) == 0:
        return []
    return _segments(_smooth(_hold(pulses, hold)), duration)
