"""Does court motion, on its own, line up with the labeled rallies?"""

import csv
import sys

import numpy as np

from rallycondense.evaluate import report
from rallycondense.features import motion_per_second
from rallycondense.segment import Segment


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
            if start and end:
                rallies.append(Segment(parse_clock(start), parse_clock(end), "user"))
    return rallies


def windows(flags: list[bool], quiet_s: int, min_s: int) -> list[Segment]:
    raw: list[tuple[int, int]] = []
    start = None
    quiet = 0
    for index, on in enumerate(flags):
        if on:
            if start is None:
                start = index
            quiet = 0
            continue
        if start is None:
            continue
        quiet += 1
        if quiet >= quiet_s:
            raw.append((start, index - quiet + 1))
            start = None
            quiet = 0
    if start is not None:
        raw.append((start, len(flags) - quiet))
    return [Segment(float(start), float(end), "motion") for start, end in raw if end - start >= min_s]


def main() -> None:
    video, labels_path = sys.argv[1:]
    labeled = load_csv(labels_path)
    window = labeled[-1].end + 20
    motion = np.array(motion_per_second(video, [(0, 0), (1, 0), (1, 1), (0, 1)], window))
    truth = np.zeros(len(motion), dtype=bool)
    for rally in labeled:
        truth[int(rally.start) : int(rally.end) + 1] = True
    play = motion[truth[: len(motion)]]
    dead = motion[~truth[: len(motion)]]
    print(
        {
            "video": video,
            "play_median": float(np.median(play)) if play.size else None,
            "dead_median": float(np.median(dead)) if dead.size else None,
            "play_p25": float(np.percentile(play, 25)) if play.size else None,
            "dead_p75": float(np.percentile(dead, 75)) if dead.size else None,
        },
        flush=True,
    )
    for quiet, minimum in ((2, 3), (3, 3), (2, 2)):
        threshold = float(np.percentile(motion, 60))
        flags = (motion >= threshold).tolist()
        predicted = windows(flags, quiet, minimum)
        scored = report(predicted, labeled, window)
        print({"quiet": quiet, "min": minimum, "threshold": round(threshold, 2), "clips": len(predicted), **scored}, flush=True)


if __name__ == "__main__":
    main()
