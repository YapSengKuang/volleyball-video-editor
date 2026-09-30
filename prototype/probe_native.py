"""See whether full-resolution frames make the ball visible to YOLOv8n."""

import os

os.environ.setdefault("YOLO_VERBOSE", "False")
from ultralytics import YOLO

model = YOLO("/app/models/yolov8n.pt")
clip = "/work/prototype/samples/native-rally.mp4"

for imgsz in (640, 1280, 1920):
    frames = 0
    seen = 0
    strong = 0
    hits = []
    for index, result in enumerate(
        model.predict(
            source=clip,
            classes=[32],
            conf=0.10,
            imgsz=imgsz,
            device="cpu",
            stream=True,
            verbose=False,
        )
    ):
        frames += 1
        if result.boxes is None or len(result.boxes) == 0:
            continue
        height, width = result.orig_shape
        best = result.boxes[int(result.boxes.conf.argmax())]
        conf = float(best.conf[0])
        x1, y1, x2, y2 = (float(value) for value in best.xyxy[0].tolist())
        box_w = x2 - x1
        seen += 1
        if conf >= 0.25 and 8 <= box_w <= 0.18 * width:
            strong += 1
        hits.append((round(index / 5, 1), round(conf, 2), round(box_w, 1), round(((y1 + y2) / 2) / height, 2)))
    print({"imgsz": imgsz, "frames": frames, "any_box": seen, "passes_filter": strong, "hits": hits})
