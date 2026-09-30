"""Score full-resolution 5 fps frames. No spatial downscale.

Every sports-ball box is kept, then a box that sits in one place is dropped.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from ball import Hit, detect_rallies, hits_to_mask, moving_hits
from rallycondense.evaluate import report
from rallycondense.segment import Segment

from ultralytics import YOLO

JOBS = [
    ("/desk/volleyball test.mov", "/work/Untitled spreadsheet - Volleyball Test.Mov.csv", "/work/prototype/samples/test1-native.json"),
    ("/desk/Volleyball Test2.mov", "/work/Untitled spreadsheet - Volleyball test 2.csv", "/work/prototype/samples/test2-native.json"),
    ("/desk/Volleyball test 3].mov", "/work/Untitled spreadsheet - Volleyball test 3.csv", "/work/prototype/samples/test3-native.json"),
]


def parse_clock(value: str) -> float:
    parts = [float(piece) for piece in value.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def load_csv(path: str) -> list[Segment]:
    rallies = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = row.get("Rally Start Timestamp") or ""
            end = row.get("Rally End TimeStamp") or row.get("Rally End Timestamp") or ""
            if not start or not end:
                continue
            rallies.append(Segment(parse_clock(start), parse_clock(end), "user"))
    return rallies


def build_native(source: str, dest: str, window: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-t",
            f"{window:.1f}",
            "-i",
            source,
            "-an",
            "-vf",
            "fps=5",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "16",
            dest,
        ],
        check=True,
    )


def collect(model, path: str) -> list[Hit]:
    hits: list[Hit] = []
    for index, result in enumerate(
        model.predict(source=path, classes=[32], conf=0.10, imgsz=1280, device="cpu", stream=True, verbose=False)
    ):
        if result.boxes is None:
            continue
        height, width = result.orig_shape
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            box_w = x2 - x1
            if conf < 0.12 or box_w < 8 or box_w > 0.18 * width:
                continue
            hits.append(
                Hit(
                    time_s=index / 5,
                    cx=((x1 + x2) / 2) / width,
                    cy=((y1 + y2) / 2) / height,
                    width_px=box_w,
                    conf=conf,
                )
            )
    return hits


def drop_static(hits: list[Hit], radius: float = 0.02, min_count: int = 8) -> list[Hit]:
    static = set()
    for index, hit in enumerate(hits):
        nearby = [
            other.time_s
            for other in hits
            if abs(other.cx - hit.cx) <= radius and abs(other.cy - hit.cy) <= radius
        ]
        if len(nearby) >= min_count and max(nearby) - min(nearby) >= 1.5:
            static.add(index)
    return [hit for index, hit in enumerate(hits) if index not in static]


def as_segments(ranges: list[tuple[float, float]]) -> list[Segment]:
    return [Segment(start, end, "ball") for start, end in ranges]


def score_one(model, video: str, labels_path: str, out_path: str) -> None:
    labeled = load_csv(labels_path)
    window = labeled[-1].end + 20
    proxy = "/tmp/native-score.mp4"
    build_native(video, proxy, window)
    raw_hits = collect(model, proxy)
    kept = drop_static(raw_hits)
    play = moving_hits(kept)
    mask = hits_to_mask(play, window) if play else None
    if mask is None:
        raw_ranges: list[tuple[float, float]] = []
        padded: list[tuple[float, float]] = []
    else:
        raw_ranges = detect_rallies(mask, 5, window, pre_roll_s=0, post_roll_s=0)
        padded = detect_rallies(mask, 5, window)
    raw_score = report(as_segments(raw_ranges), labeled, window)
    padded_score = report(as_segments(padded), labeled, window)
    payload = {
        "duration": window,
        "players_per_side": 6,
        "resolution": "1920x1080",
        "fps": 5,
        "imgsz": 1280,
        "hits": len(raw_hits),
        "moving_hits": len(play),
        "raw_rallies": [{"start": start, "end": end} for start, end in raw_ranges],
        "padded_rallies": [{"start": start, "end": end} for start, end in padded],
        "raw_score": raw_score,
        "padded_score": padded_score,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "video": Path(video).name,
                "hits": len(raw_hits),
                "moving": len(play),
                "raw": raw_score,
                "padded": padded_score,
            }
        ),
        flush=True,
    )


def main() -> None:
    model = YOLO("/app/models/yolov8n.pt")
    for video, labels, out in JOBS:
        score_one(model, video, labels, out)


if __name__ == "__main__":
    main()
