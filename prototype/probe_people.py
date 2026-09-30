"""Print person boxes on a few known frames so the formation rule can be set from them."""

import os
import sys

os.environ.setdefault("YOLO_VERBOSE", "False")
from ultralytics import YOLO

model = YOLO("/app/models/yolov8n.pt")
for path in sys.argv[1:]:
    result = next(model.predict(source=path, classes=[0], conf=0.35, imgsz=640, device="cpu", stream=True, verbose=False))
    height, width = result.orig_shape
    people = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            people.append(
                (
                    round(float(box.conf[0]), 2),
                    round(((x1 + x2) / 2) / width, 2),
                    round(y2 / height, 2),
                    round((y2 - y1) / height, 2),
                )
            )
    people.sort(key=lambda item: item[2])
    print(path, len(people))
    for person in people:
        print(" ", person)
