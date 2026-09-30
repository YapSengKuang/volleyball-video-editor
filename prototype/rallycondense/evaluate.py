"""Score predicted rallies against hand-labeled start and end times."""

from __future__ import annotations

from rallycondense.segment import Segment


def _seconds(segments: list[Segment], duration: float) -> list[bool]:
    flags = [False] * max(1, int(duration) + 1)
    for segment in segments:
        begin = max(0, int(segment.start))
        finish = min(len(flags), int(segment.end) + 1)
        for index in range(begin, finish):
            flags[index] = True
    return flags


def in_play_scores(predicted: list[Segment], labeled: list[Segment], duration: float) -> dict:
    pred = _seconds(predicted, duration)
    truth = _seconds(labeled, duration)
    count = min(len(pred), len(truth))
    tp = fp = fn = 0
    for index in range(count):
        if pred[index] and truth[index]:
            tp += 1
        elif pred[index]:
            fp += 1
        elif truth[index]:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def _iou(left: Segment, right: Segment) -> float:
    overlap = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    if union <= 0:
        return 0.0
    return overlap / union


def boundary_error(predicted: list[Segment], labeled: list[Segment]) -> dict:
    """Match each labeled rally to the prediction with the highest overlap."""
    if not labeled:
        return {"matches": 0, "mean_start_error": 0.0, "mean_end_error": 0.0}
    start_errors = []
    end_errors = []
    for truth in labeled:
        if not predicted:
            continue
        best = max(predicted, key=lambda item: _iou(item, truth))
        if _iou(best, truth) <= 0:
            continue
        start_errors.append(abs(best.start - truth.start))
        end_errors.append(abs(best.end - truth.end))
    if not start_errors:
        return {"matches": 0, "mean_start_error": None, "mean_end_error": None}
    return {
        "matches": len(start_errors),
        "mean_start_error": sum(start_errors) / len(start_errors),
        "mean_end_error": sum(end_errors) / len(end_errors),
    }


def report(predicted: list[Segment], labeled: list[Segment], duration: float) -> dict:
    scores = in_play_scores(predicted, labeled, duration)
    bounds = boundary_error(predicted, labeled)
    start = bounds["mean_start_error"]
    end = bounds["mean_end_error"]
    typical = None if start is None or end is None else (start + end) / 2
    passed = scores["recall"] >= 0.90 and typical is not None and typical <= 1.5
    return {
        **scores,
        **bounds,
        "typical_boundary_error": typical,
        "meets_exit_bar": passed,
    }
