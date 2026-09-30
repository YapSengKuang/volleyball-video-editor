"""Keep every ball box on the native clip, then drop the one that sits still."""

import os

os.environ.setdefault("YOLO_VERBOSE", "False")

from ultralytics import YOLO

from ball import Hit, detect_rallies, moving_hits, rallies_from_hits

CLIP = "/work/prototype/samples/native-rally.mp4"
# Clip starts at 6.0s of volleyball test.mov. The labeled rally is 6–15s.
LABEL = (0.0, 9.0)


def collect(model, imgsz: int) -> list[Hit]:
    hits: list[Hit] = []
    for index, result in enumerate(
        model.predict(source=CLIP, classes=[32], conf=0.10, imgsz=imgsz, device="cpu", stream=True, verbose=False)
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
    """A logo or a light that stays put is not the ball."""
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


def main() -> None:
    model = YOLO("/app/models/yolov8n.pt")
    for imgsz in (1280, 1920):
        raw = collect(model, imgsz)
        kept = drop_static(raw)
        play = moving_hits(kept, min_speed=0.12)
        rallies = rallies_from_hits(play, 12.0, None) if len(play) >= 2 else []
        covered = 0
        for start, end in rallies:
            covered += max(0.0, min(end, LABEL[1]) - max(start, LABEL[0]))
        print(
            {
                "imgsz": imgsz,
                "raw": len(raw),
                "after_static_drop": len(kept),
                "moving": len(play),
                "rallies": [(round(start, 1), round(end, 1)) for start, end in rallies],
                "labeled_seconds_covered": round(covered, 1),
                "labeled_seconds": LABEL[1] - LABEL[0],
            }
        )


if __name__ == "__main__":
    main()
