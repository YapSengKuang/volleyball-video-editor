"""Turn the spreadsheet timestamps into seconds and score one match."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from rallycondense.evaluate import report
from rallycondense.features import motion_per_second, samples_from_motion
from rallycondense.segment import Segment, clean_segments, detect_rallies

FULL_COURT = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


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
            start = row.get("Rally Start Timestamp") or row.get("start")
            end = row.get("Rally End TimeStamp") or row.get("Rally End Timestamp") or row.get("end")
            if not start or not end:
                continue
            rallies.append(Segment(parse_clock(start), parse_clock(end), "user"))
    return rallies


def main() -> int:
    video, labels_path, out_path = sys.argv[1:]
    labeled = load_csv(labels_path)
    window = labeled[-1].end + 20
    motion = motion_per_second(video, FULL_COURT, window)
    predicted = clean_segments(detect_rallies(samples_from_motion(motion)), window)
    result = report(predicted, labeled, window)
    payload = {
        "duration": window,
        "players_per_side": 6,
        "rallies": [{"start": item.start, "end": item.end, "source": item.source} for item in predicted],
        "score": result,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"video": video, "labeled": len(labeled), "predicted": len(predicted), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
