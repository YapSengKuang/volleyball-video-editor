"""Ball-in-play rally windows.

YOLOv8 looks for the ball on the preview. A rally starts when the ball is
moving and ends after it has been gone for a few seconds. A ball that is
only being held, bounced for a serve, or carried back is too slow to keep
the rally open, which is where ball detection on its own runs long.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from analyze import order_corners
from config import BALL_DEVICE, BALL_IMGSZ, BALL_THREADS, PROXY_FPS

COCO_SPORTS_BALL = 32
DETECT_FPS = PROXY_FPS
DEAD_GAP_S = 3.0
MIN_RALLY_S = 1.5
PRE_ROLL_S = 1.5
POST_ROLL_S = 2.0
MIN_SPEED = 0.22  # fraction of the frame width per second
# predict() confidence is lower so the explicit gate below is what keeps a box.
PREDICT_CONF = 0.15
BALL_MIN_CONF = 0.25
BALL_MIN_WIDTH_PX = 8.0
BALL_MAX_WIDTH_FRAC = 0.18


@dataclass
class Hit:
    time_s: float
    cx: float
    cy: float
    width_px: float
    conf: float


def fill_short_gaps(mask: np.ndarray, gap_frames: int) -> np.ndarray:
    filled = mask.copy()
    seen = False
    index = 0
    while index < len(filled):
        if filled[index]:
            seen = True
            index += 1
            continue
        end = index
        while end < len(filled) and not filled[end]:
            end += 1
        if seen and end < len(filled) and (end - index) <= gap_frames:
            filled[index:end] = True
        index = end
    return filled


def detect_rallies(
    mask: np.ndarray,
    fps: float,
    duration: float,
    dead_gap_s: float = DEAD_GAP_S,
    min_rally_s: float = MIN_RALLY_S,
    pre_roll_s: float = PRE_ROLL_S,
    post_roll_s: float = POST_ROLL_S,
) -> list[tuple[float, float]]:
    """DEAD until the ball is seen, then RALLY until it stays missing."""
    if fps <= 0 or mask.size == 0:
        return []
    dead_gap_frames = max(1, int(dead_gap_s * fps))
    min_rally_frames = max(1, int(min_rally_s * fps))
    raw: list[tuple[float, float]] = []
    in_rally = False
    rally_start = 0
    last_detect = -dead_gap_frames - 1
    for index, detected in enumerate(mask):
        if detected:
            if not in_rally:
                in_rally = True
                rally_start = index
            last_detect = index
        elif in_rally and (index - last_detect) > dead_gap_frames:
            if (last_detect - rally_start) >= min_rally_frames:
                raw.append((rally_start / fps, last_detect / fps))
            in_rally = False
    if in_rally and (last_detect - rally_start) >= min_rally_frames:
        raw.append((rally_start / fps, last_detect / fps))
    padded = [
        (max(0.0, start - pre_roll_s), min(duration, end + post_roll_s))
        for start, end in raw
    ]
    return _merge(padded)


def _merge(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _inside_court(cx: float, cy: float, corners: list) -> bool:
    poly = order_corners(corners)
    sign = 0.0
    crosses = []
    cx_c, cy_c = poly.mean(axis=0)
    for index in range(4):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % 4]
        crosses.append((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1))
        sign += (x2 - x1) * (cy_c - y1) - (y2 - y1) * (cx_c - x1)
    want_positive = sign >= 0
    return all(cross >= -1e-6 if want_positive else cross <= 1e-6 for cross in crosses)


def moving_hits(hits: list[Hit], min_speed: float = MIN_SPEED) -> list[Hit]:
    """Keep hits that are part of a fast move. A carried ball stays out."""
    if len(hits) < 2:
        return []
    kept: list[Hit] = []
    previous = hits[0]
    for hit in hits[1:]:
        dt = hit.time_s - previous.time_s
        if 0 < dt <= 0.6:
            speed = math.hypot(hit.cx - previous.cx, hit.cy - previous.cy) / dt
            if speed >= min_speed:
                kept.extend((previous, hit))
        previous = hit
    unique: list[Hit] = []
    seen: set[float] = set()
    for hit in kept:
        if hit.time_s in seen:
            continue
        seen.add(hit.time_s)
        unique.append(hit)
    return unique


def hits_to_mask(hits: list[Hit], duration: float, fps: float = DETECT_FPS) -> np.ndarray:
    count = max(1, int(math.ceil(duration * fps)) + 1)
    mask = np.zeros(count, dtype=bool)
    for hit in hits:
        index = int(round(hit.time_s * fps))
        if 0 <= index < count:
            mask[index] = True
    return fill_short_gaps(mask, max(1, int(0.5 * fps)))


def rallies_from_hits(hits: list[Hit], duration: float, corners: list | None) -> list[tuple[float, float]]:
    filtered = []
    for hit in hits:
        if corners is not None and not _inside_court(hit.cx, hit.cy, corners):
            continue
        filtered.append(hit)
    play = moving_hits(filtered)
    if len(play) < 2:
        return []
    mask = hits_to_mask(play, duration)
    return detect_rallies(mask, DETECT_FPS, duration)


def _threads() -> int:
    if BALL_THREADS > 0:
        return BALL_THREADS
    return max(1, os.cpu_count() or 1)


def _device() -> str:
    if BALL_DEVICE != "auto":
        return BALL_DEVICE
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def scan_ball(path: str, corners: list | None, duration: float, on_progress=None) -> list[Hit]:
    os.environ.setdefault("YOLO_VERBOSE", "False")
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(_threads())
    model_path = os.environ.get("BALL_MODEL", "/app/models/yolov8n.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
    model = YOLO(model_path)
    names = model.names if isinstance(model.names, dict) else {}
    ball_class = COCO_SPORTS_BALL if COCO_SPORTS_BALL in names else 0
    expected = max(1, int(duration * DETECT_FPS))
    hits: list[Hit] = []
    stream = model.predict(
        source=path,
        classes=[ball_class],
        conf=PREDICT_CONF,
        imgsz=BALL_IMGSZ,
        device=_device(),
        stream=True,
        verbose=False,
    )
    for index, result in enumerate(stream):
        if on_progress and index % 20 == 0:
            on_progress(min(0.98, index / expected))
        if result.boxes is None or len(result.boxes) == 0:
            continue
        height, width = result.orig_shape
        best = result.boxes[int(result.boxes.conf.argmax())]
        conf = float(best.conf[0])
        x1, y1, x2, y2 = (float(value) for value in best.xyxy[0].tolist())
        box_w = x2 - x1
        if conf < BALL_MIN_CONF or box_w < BALL_MIN_WIDTH_PX or box_w > BALL_MAX_WIDTH_FRAC * width:
            continue
        cy = ((y1 + y2) / 2) / height
        if corners is None and cy > 0.72:
            continue
        hits.append(
            Hit(
                time_s=index / DETECT_FPS,
                cx=((x1 + x2) / 2) / width,
                cy=cy,
                width_px=box_w,
                conf=conf,
            )
        )
    return hits
