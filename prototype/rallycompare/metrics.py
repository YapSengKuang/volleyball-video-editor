"""Per-second and per-rally scores. No file I/O."""

from __future__ import annotations

import numpy as np


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def in_play_mask(
    segments: list[tuple[float, float]],
    origin: float,
    n_bins: int,
    bin_s: float = 1.0,
) -> np.ndarray:
    flags = np.zeros(n_bins, dtype=bool)
    for index in range(n_bins):
        t0 = origin + index * bin_s
        t1 = t0 + bin_s
        if any(_overlap(t0, t1, start, end) > 0 for start, end in segments):
            flags[index] = True
    return flags


def second_scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(predicted, dtype=bool)
    true = np.asarray(truth, dtype=bool)
    count = min(len(pred), len(true))
    pred = pred[:count]
    true = true[:count]
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "tp": tp, "fp": fp, "fn": fn}


def _covered(inner: tuple[float, float], covers: list[tuple[float, float]], fraction: float) -> bool:
    start, end = inner
    length = end - start
    if length <= 0:
        return False
    overlap = sum(_overlap(start, end, other[0], other[1]) for other in covers)
    return overlap / length >= fraction


def segment_scores(
    predicted: list[tuple[float, float]],
    truth: list[tuple[float, float]],
    fraction: float = 0.5,
) -> dict[str, float]:
    seg_recall = (
        sum(1 for rally in truth if _covered(rally, predicted, fraction)) / len(truth) if truth else 0.0
    )
    seg_precision = (
        sum(1 for rally in predicted if _covered(rally, truth, fraction)) / len(predicted) if predicted else 0.0
    )
    return {
        "segment_recall": seg_recall,
        "segment_precision": seg_precision,
        "n_pred": len(predicted),
        "n_true": len(truth),
    }


def duration_stats(segments: list[tuple[float, float]], window: float) -> dict[str, float]:
    lengths = [end - start for start, end in segments if end > start]
    kept = float(sum(lengths))
    return {
        "n_rallies": len(lengths),
        "in_play_pct": (100.0 * kept / window) if window > 0 else 0.0,
        "mean_dur": float(np.mean(lengths)) if lengths else 0.0,
        "median_dur": float(np.median(lengths)) if lengths else 0.0,
    }


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    count = min(len(a), len(b))
    a = a[:count]
    b = b[:count]
    union = int(np.sum(a | b))
    if union == 0:
        return 1.0
    return float(np.sum(a & b) / union)
