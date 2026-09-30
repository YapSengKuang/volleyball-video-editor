"""Print one formation sample per second so a rally start is visible."""

from __future__ import annotations

import subprocess
import sys

from ultralytics import YOLO


def main() -> None:
    video, start_s, seconds = sys.argv[1:]
    start = float(start_s)
    duration = float(seconds)
    proxy = "/tmp/timeline.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-ss", f"{start:.1f}", "-t", f"{duration:.1f}", "-i", video,
            "-an", "-vf", "fps=2,scale=-2:640", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            proxy,
        ],
        check=True,
    )
    model = YOLO("/app/models/yolov8n.pt")
    previous = None
    for index, result in enumerate(
        model.predict(source=proxy, classes=[0], conf=0.35, imgsz=640, device="cpu", stream=True, verbose=False)
    ):
        height, width = result.orig_shape
        people = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                cx, foot, box_h = ((x1 + x2) / 2) / width, y2 / height, (y2 - y1) / height
                if 0.10 < cx < 0.95 and box_h > 0.08:
                    people.append((cx, foot, box_h))
        people.sort(key=lambda person: person[2], reverse=True)
        near = people[:3]
        xs = [person[0] for person in people] or [0.0]
        shift = 0.0
        if previous:
            shifts = []
            for cx, foot, _box_h in people:
                nearest = min(previous, key=lambda other: abs(other[0] - cx) + abs(other[1] - foot))
                shifts.append(abs(nearest[0] - cx) + abs(nearest[1] - foot))
            shift = sum(shifts) / len(shifts) if shifts else 0.0
        previous = people
        near_foot = sum(person[1] for person in near) / len(near) if near else 0.0
        print(
            f"{start + index / 2:6.1f}  n={len(people):2d}  span={max(xs) - min(xs):.2f}  "
            f"near_foot={near_foot:.2f}  shift={shift:.3f}"
        )


if __name__ == "__main__":
    main()
