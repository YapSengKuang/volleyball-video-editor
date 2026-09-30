"""Score the ball tracker on the labeled stretch of one match.

The spreadsheets stop before the recording ends, so only that window is decoded.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from ball import rallies_from_hits, scan_ball
from rallycondense.evaluate import report
from rallycondense.segment import Segment

# The labeled camera looks down one court from the end line, so the near
# baseline sits at the bottom of the frame. Passing corners keeps those hits.
FULL_FRAME = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def parse_clock(value: str) -> float:
    parts = [float(piece) for piece in value.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Cannot read time {value}")


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


def build_proxy(source: str, dest: str, window: float) -> None:
    command = [
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
        "scale=-2:640,fps=5",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "32",
        dest,
    ]
    subprocess.run(command, check=True)


def main() -> int:
    video, labels_path, out_path = sys.argv[1:]
    labeled = load_csv(labels_path)
    window = labeled[-1].end + 20
    proxy = "/tmp/score-proxy.mp4"
    build_proxy(video, proxy, window)
    hits = scan_ball(proxy, FULL_FRAME, window)
    predicted_raw = rallies_from_hits(hits, window, FULL_FRAME)
    predicted = [Segment(start, end, "ball") for start, end in predicted_raw]
    result = report(predicted, labeled, window)
    payload = {
        "duration": window,
        "players_per_side": 6,
        "hits": len(hits),
        "rallies": [{"start": item.start, "end": item.end} for item in predicted],
        "labels": [{"start": item.start, "end": item.end} for item in labeled],
        "score": result,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"video": Path(video).name, "labeled": len(labeled), "hits": len(hits), "predicted": len(predicted), **result},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
