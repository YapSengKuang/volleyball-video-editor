FROM python:3.12-slim-bookworm

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps \
  && mkdir -p /app/models \
  && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
  && mv yolov8n.pt /app/models/yolov8n.pt
COPY . .
ENV BALL_MODEL=/app/models/yolov8n.pt

CMD ["python", "-u", "main.py"]
