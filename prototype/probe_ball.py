"""Count raw sports-ball boxes on a short clip, before the size filter."""

import os

os.environ.setdefault("YOLO_VERBOSE", "False")
from ultralytics import YOLO

model = YOLO("/app/models/yolov8n.pt")
clip = "/work/prototype/samples/rally-clip.mp4"

for imgsz, conf in ((640, 0.10), (960, 0.10)):
    boxes = []
    for index, result in enumerate(model.predict(source=clip, classes=[32], conf=conf, imgsz=imgsz, device="cpu", stream=True, verbose=False)):
        if result.boxes is None:
            continue
        height, width = result.orig_shape
        for box in result.boxes:
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            boxes.append((index / 5, float(box.conf[0]), round(x2 - x1, 1), round(((y1 + y2) / 2) / height, 2)))
    print({"imgsz": imgsz, "conf": conf, "boxes": len(boxes), "hits": boxes[:40]})
