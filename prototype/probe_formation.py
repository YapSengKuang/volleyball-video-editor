"""See which person-formation feature separates labeled rallies from the gaps."""

from __future__ import annotations

import csv
import statistics
import subprocess
import sys

from ultralytics import YOLO

PERSON = 0


def parse_clock(value: str) -> float:
    parts = [float(piece) for piece in value.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def load_csv(path: str) -> list[tuple[float, float]]:
    rallies = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = row.get("Rally Start Timestamp") or ""
            end = row.get("Rally End TimeStamp") or row.get("Rally End Timestamp") or ""
            if start and end:
                rallies.append((parse_clock(start), parse_clock(end)))
    return rallies


def in_label(second: float, rallies: list[tuple[float, float]]) -> bool:
    return any(start <= second <= end for start, end in rallies)


def main() -> None:
    video, labels_path = sys.argv[1:]
    rallies = load_csv(labels_path)
    window = rallies[-1][1] + 20
    proxy = "/tmp/people.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-t", f"{window:.1f}", "-i", video,
            "-an", "-vf", "fps=2,scale=-2:640", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            proxy,
        ],
        check=True,
    )
    model = YOLO("/app/models/yolov8n.pt")
    play = {"n": [], "span": [], "near_foot": [], "spread": []}
    dead = {"n": [], "span": [], "near_foot": [], "spread": []}
    for index, result in enumerate(
        model.predict(source=proxy, classes=[PERSON], conf=0.35, imgsz=640, device="cpu", stream=True, verbose=False)
    ):
        height, width = result.orig_shape
        people = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                people.append((((x1 + x2) / 2) / width, y2 / height, (y2 - y1) / height))
        court = [person for person in people if 0.10 < person[0] < 0.95 and person[2] > 0.08]
        second = index / 2
        bucket = play if in_label(second, rallies) else dead
        if len(court) < 4:
            bucket["n"].append(len(court))
            continue
        court.sort(key=lambda person: person[2], reverse=True)
        near = court[:3]
        xs = [person[0] for person in court]
        mean_x = sum(xs) / len(xs)
        bucket["n"].append(len(court))
        bucket["span"].append(max(xs) - min(xs))
        bucket["near_foot"].append(sum(person[1] for person in near) / len(near))
        bucket["spread"].append((sum((value - mean_x) ** 2 for value in xs) / len(xs)) ** 0.5)

    def summarize(name: str, values: dict) -> None:
        print(name)
        for key, series in values.items():
            if not series:
                print(f"  {key}: empty")
                continue
            print(f"  {key}: median {statistics.median(series):.3f}  p25 {sorted(series)[len(series)//4]:.3f}  n {len(series)}")

    summarize("play", play)
    summarize("dead", dead)


if __name__ == "__main__":
    main()
