"""Per-second features inside the court polygon."""

from __future__ import annotations

import subprocess

import numpy as np

from rallycondense.court import order_corners
from rallycondense.segment import Second


def _mask(height: int, width: int, corners: list) -> np.ndarray:
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
    return mask if int(mask.sum()) >= 16 else np.ones((height, width), dtype=bool)


def motion_per_second(video: str, corners: list, duration: float) -> list[float]:
    width, height = 160, 90
    mask = _mask(height, width, corners)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-t",
        f"{duration:.1f}",
        "-i",
        video,
        "-vf",
        f"fps=4,scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    frame_size = width * height
    previous = None
    buckets: list[list[float]] = [[] for _ in range(max(1, int(duration) + 1))]
    frame_index = 0
    while True:
        raw = process.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).astype(np.float32).reshape(height, width)[mask]
        if previous is not None:
            second = min(len(buckets) - 1, int((frame_index / 4)))
            buckets[second].append(float(np.mean(np.abs(frame - previous))))
        previous = frame
        frame_index += 1
    process.stdout.close()
    process.wait(timeout=30)
    return [float(np.mean(values)) if values else 0.0 for values in buckets]


def samples_from_motion(motion: list[float]) -> list[Second]:
    return [Second(motion=value) for value in motion]
