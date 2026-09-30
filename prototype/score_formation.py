"""Start a rally when the six are set, then track the ball inside the court.

People are scanned on a small frame for the whole clip. The ball is scanned
only after that, on the court crop scaled back up. Prints minutes removed.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from ball import Hit, moving_hits
from rallycondense.evaluate import report
from rallycondense.segment import Segment

JOBS = [
    ("/desk/volleyball test.mov", "/work/Untitled spreadsheet - Volleyball Test.Mov.csv", "/work/prototype/samples/test1-formation.json"),
    ("/desk/Volleyball Test2.mov", "/work/Untitled spreadsheet - Volleyball test 2.csv", "/work/prototype/samples/test2-formation.json"),
    ("/desk/Volleyball test 3].mov", "/work/Untitled spreadsheet - Volleyball test 3.csv", "/work/prototype/samples/test3-formation.json"),
]
PERSON_FPS = 2
BALL_FPS = 5


@dataclass
class PersonFrame:
    time: float
    near_foot: float
    shift: float
    count: int


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


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def even(value: int) -> int:
    return max(2, value - (value % 2))


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True)


def file_duration(video: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def build_people_proxy(video: str, dest: str, window: float) -> None:
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-v", "error", "-t", f"{window:.1f}", "-i", video, "-an",
            "-vf", f"fps={PERSON_FPS},scale=-2:640", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            dest,
        ]
    )


def build_ball_proxy(video: str, dest: str, window: float) -> None:
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-v", "error", "-t", f"{window:.1f}", "-i", video, "-an",
            "-vf", f"fps={BALL_FPS}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-g", "5",
            dest,
        ]
    )


def read_people(model, proxy: str) -> tuple[list[PersonFrame], list[tuple[float, float, float]]]:
    frames: list[PersonFrame] = []
    court_points: list[tuple[float, float, float]] = []
    previous: list[tuple[float, float]] = []
    for index, result in enumerate(
        model.predict(source=proxy, classes=[0], conf=0.35, imgsz=640, device="cpu", stream=True, verbose=False)
    ):
        height, width = result.orig_shape
        people: list[tuple[float, float, float]] = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                cx = ((x1 + x2) / 2) / width
                foot = y2 / height
                box_h = (y2 - y1) / height
                if 0.10 < cx < 0.95 and box_h > 0.09:
                    people.append((cx, foot, box_h))
                    court_points.append((cx, foot, box_h))
        people.sort(key=lambda person: person[2], reverse=True)
        near = people[:3]
        shift = 0.0
        if previous and people:
            shifts = []
            for cx, foot, _box_h in people:
                nearest = min(previous, key=lambda other: abs(other[0] - cx) + abs(other[1] - foot))
                shifts.append(abs(nearest[0] - cx) + abs(nearest[1] - foot))
            shift = sum(shifts) / len(shifts)
        previous = [(cx, foot) for cx, foot, _box_h in people]
        frames.append(
            PersonFrame(
                time=index / PERSON_FPS,
                near_foot=sum(person[1] for person in near) / len(near) if near else 0.0,
                shift=shift,
                count=len(people),
            )
        )
    return frames, court_points


def court_box(points: list[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    if len(points) < 20:
        return (0.0, 0.0, 1.0, 1.0)
    xs = [point[0] for point in points]
    feet = [point[1] for point in points]
    tops = [point[1] - point[2] for point in points]
    x0 = max(0.0, percentile(xs, 8) - 0.05)
    x1 = min(1.0, percentile(xs, 92) + 0.05)
    y0 = max(0.0, percentile(tops, 5) - 0.28)
    y1 = min(1.0, percentile(feet, 98) + 0.04)
    if x1 - x0 < 0.35 or y1 - y0 < 0.35:
        return (0.0, 0.0, 1.0, 1.0)
    return (x0, y0, x1, y1)


def formation_starts(frames: list[PersonFrame]) -> list[float]:
    usable = [frame for frame in frames if frame.count >= 6 and frame.time > 0]
    if len(usable) < 8:
        return []
    foot_high = percentile([frame.near_foot for frame in usable], 70)
    foot_low = percentile([frame.near_foot for frame in usable], 42)
    shift_low = percentile([frame.shift for frame in usable if frame.shift > 0], 40)
    starts: list[float] = []
    run = 0
    run_started = 0.0
    armed = True
    for frame in frames:
        quiet_back = frame.count >= 6 and frame.near_foot >= foot_high and 0 < frame.shift <= shift_low
        if quiet_back:
            if run == 0:
                run_started = frame.time
            run += 1
            if armed and run >= PERSON_FPS:
                starts.append(run_started)
                armed = False
        else:
            run = 0
        if frame.near_foot <= foot_low:
            armed = True
    return starts


def crop_clip(proxy: str, dest: str, start: float, duration: float, box: tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = box
    width, height = 1920, 1080
    x = even(int(x0 * width))
    y = even(int(y0 * height))
    crop_w = even(int((x1 - x0) * width))
    crop_h = even(int((y1 - y0) * height))
    crop_w = min(crop_w, width - x)
    crop_h = min(crop_h, height - y)
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-v", "error", "-i", proxy, "-ss", f"{start:.2f}", "-t", f"{duration:.2f}",
            "-vf", f"crop={crop_w}:{crop_h}:{x}:{y},scale=1920:-2",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-an", dest,
        ]
    )


def ball_hits(model, clip: str, offset: float) -> list[Hit]:
    hits: list[Hit] = []
    for index, result in enumerate(
        model.predict(source=clip, classes=[32], conf=0.10, imgsz=1280, device="cpu", stream=True, verbose=False)
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
                    time_s=offset + index / BALL_FPS,
                    cx=((x1 + x2) / 2) / width,
                    cy=((y1 + y2) / 2) / height,
                    width_px=box_w,
                    conf=conf,
                )
            )
    return hits


def drop_static(hits: list[Hit]) -> list[Hit]:
    static = set()
    for index, hit in enumerate(hits):
        nearby = [
            other.time_s
            for other in hits
            if abs(other.cx - hit.cx) <= 0.02 and abs(other.cy - hit.cy) <= 0.02
        ]
        if len(nearby) >= 8 and max(nearby) - min(nearby) >= 1.5:
            static.add(index)
    return [hit for index, hit in enumerate(hits) if index not in static]


def ball_end(times: list[float], start: float) -> float | None:
    kept = sorted(time for time in times if start - 0.4 <= time <= start + 40)
    if len(kept) < 3:
        return None
    end = kept[0]
    for previous, current in zip(kept, kept[1:]):
        if current - previous > 6:
            break
        end = current
    return min(start + 40, end + 0.8)


def release_end(frames: list[PersonFrame], start: float, foot_low: float) -> float:
    run = 0
    for frame in frames:
        if frame.time < start + 4:
            continue
        if frame.near_foot <= foot_low:
            run += 1
            if run >= 3:
                return min(start + 40, frame.time)
        else:
            run = 0
    return start + 18


def build_rallies(frames: list[PersonFrame], starts: list[float], ball_ends: list[float | None]) -> list[Segment]:
    usable = [frame.near_foot for frame in frames if frame.count >= 6]
    foot_low = percentile(usable, 42)
    rallies: list[Segment] = []
    cursor = 0.0
    for start, end_from_ball in zip(starts, ball_ends):
        if start < cursor:
            continue
        end = end_from_ball if end_from_ball and end_from_ball > start + 2 else release_end(frames, start, foot_low)
        end = max(end, start + 3)
        rallies.append(Segment(start, min(end, start + 40), "formation"))
        cursor = end
    return rallies


def minutes_saved(duration: float, segments: list[Segment]) -> float:
    kept = 0.0
    for segment in segments:
        kept += max(0.0, segment.end - segment.start)
    return (duration - kept) / 60


def score_one(people_model, ball_model, video: str, labels_path: str, out_path: str) -> None:
    labeled = load_csv(labels_path)
    window = labeled[-1].end + 20
    people_proxy = "/tmp/people-proxy.mp4"
    ball_proxy = "/tmp/ball-proxy.mp4"
    build_people_proxy(video, people_proxy, window)
    frames, points = read_people(people_model, people_proxy)
    box = court_box(points)
    starts = formation_starts(frames)
    build_ball_proxy(video, ball_proxy, window)
    ends: list[float | None] = []
    for index, start in enumerate(starts):
        clip = f"/tmp/court-crop-{index}.mp4"
        crop_clip(ball_proxy, clip, max(0.0, start - 0.4), 40.4, box)
        hits = drop_static(ball_hits(ball_model, clip, max(0.0, start - 0.4)))
        play = moving_hits(hits)
        ends.append(ball_end([hit.time_s for hit in play], start))
    predicted = build_rallies(frames, starts, ends)
    scored = report(predicted, labeled, window)
    full = file_duration(video)
    saved = minutes_saved(window, predicted)
    labeled_saved = minutes_saved(window, labeled)
    payload = {
        "players_per_side": 6,
        "file_minutes": round(full / 60, 2),
        "analyzed_minutes": round(window / 60, 2),
        "kept_minutes": round(window / 60 - saved, 2),
        "minutes_saved": round(saved, 2),
        "labeled_minutes_saved": round(labeled_saved, 2),
        "court": {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]},
        "rallies": [{"start": item.start, "end": item.end} for item in predicted],
        "score": scored,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "video": Path(video).name,
                "analyzed_minutes": payload["analyzed_minutes"],
                "kept_minutes": payload["kept_minutes"],
                "minutes_saved": payload["minutes_saved"],
                "labeled_minutes_saved": payload["labeled_minutes_saved"],
                "rallies": len(predicted),
                "recall": round(scored["recall"], 3),
                "precision": round(scored["precision"], 3),
                "boundary_error": scored["typical_boundary_error"],
                "meets_exit_bar": scored["meets_exit_bar"],
            }
        ),
        flush=True,
    )


def main() -> None:
    people_model = YOLO("/app/models/yolov8n.pt")
    ball_model = people_model
    for video, labels, out in JOBS:
        score_one(people_model, ball_model, video, labels, out)


if __name__ == "__main__":
    main()
