"""Read one sampled pass of the video and turn it into per-second signals."""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from rallycompare.ballcfg import (
    BALL_MAX_WIDTH_FRAC,
    BALL_MIN_CONF,
    BALL_MIN_WIDTH_PX,
    COCO_SPORTS_BALL,
    MIN_SPEED,
    PREDICT_CONF,
)
from rallycompare.geometry import (
    full_frame_polygon,
    point_in_polygon,
    raster_mask,
    scale_polygon,
    zone_above,
)
from rallycompare.post import normalise_percentile, suppress_static, temporal_support

ALL_METHODS = (
    "ball_baseline",
    "ball_filtered",
    "frame_diff",
    "player_track",
    "audio",
    "contact",
    "ratio",
    "onset",
    "fused",
)
FUSE_INPUTS = ("frame_diff", "player_track", "audio")
BALL_METHODS = {"ball_baseline", "ball_filtered"}


@dataclass
class ExtractResult:
    origin: float
    start: float
    end: float
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    width: int = 0
    height: int = 0


def parse_clock(value: str) -> float:
    text = value.strip()
    if not text:
        raise ValueError("empty time")
    if ":" not in text:
        return float(text)
    parts = [float(piece) for piece in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Cannot read time {value}")


def parse_labels(path: str) -> list[tuple[float, float]]:
    rallies: list[tuple[float, float]] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        cells = [cell.strip() for cell in row.replace(";", ",").split(",")]
        if len(cells) < 2:
            continue
        try:
            start = parse_clock(cells[0])
            end = parse_clock(cells[1])
        except ValueError:
            continue
        if end > start:
            rallies.append((start, end))
    return rallies


def video_bounds(path: str) -> tuple[float, float, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    return duration, fps, width, height


def iter_frames(path: str, start: float, end: float, sample_fps: float, log):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f = max(0, int(start * native))
    end_f = int(end * native) if end > 0 else total
    if total:
        end_f = min(end_f, total)
    step = max(1, int(round(native / sample_fps)))
    expected = max(1, (end_f - start_f) // step)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frame_no = start_f
    emitted = 0
    try:
        while frame_no < end_f:
            if not cap.grab():
                break
            if (frame_no - start_f) % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield frame_no / native, frame
                    emitted += 1
                    if emitted == 1 or emitted % 20 == 0:
                        log(f"extract {emitted}/{expected}  t={frame_no / native:.1f}s")
            frame_no += 1
    finally:
        cap.release()
    log(f"extract {emitted}/{expected} done")


def _bin(times: list[float], values: list[float], origin: float, n_bins: int) -> np.ndarray:
    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    for stamp, value in zip(times, values):
        index = int(math.floor(stamp - origin))
        if 0 <= index < n_bins:
            sums[index] += value
            counts[index] += 1
    out = np.zeros(n_bins, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load_models(weights: str, need_ball: bool, need_person: bool):
    from ultralytics import YOLO

    ball_model = YOLO(weights) if need_ball else None
    person_model = YOLO(weights) if need_person else None
    return ball_model, person_model


def _boxes(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    height, width = result.orig_shape
    found = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        found.append(
            {
                "conf": float(box.conf[0]),
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "w": x2 - x1,
                "h": y2 - y1,
                "frame_w": width,
                "frame_h": height,
                "id": int(box.id[0]) if box.id is not None else -1,
            }
        )
    return found


class _BallBaseline:
    def __init__(self) -> None:
        self.times: list[float] = []
        self.values: list[float] = []
        self._prev: tuple[float, float, float] | None = None

    def consume(self, stamp: float, boxes: list[dict]) -> None:
        hit = 0.0
        best = max(boxes, key=lambda item: item["conf"]) if boxes else None
        if best is None:
            self._prev = None
        else:
            if self._prev is not None and best["conf"] >= BALL_MIN_CONF:
                dt = stamp - self._prev[0]
                width = best["frame_w"]
                if dt > 0 and BALL_MIN_WIDTH_PX <= best["w"] <= BALL_MAX_WIDTH_FRAC * width:
                    speed = math.hypot(
                        (best["cx"] - self._prev[1]) / width,
                        (best["cy"] - self._prev[2]) / best["frame_h"],
                    ) / dt
                    if speed >= MIN_SPEED:
                        hit = 1.0
            self._prev = (stamp, best["cx"], best["cy"])
        self.times.append(stamp)
        self.values.append(hit)


class _BallFiltered:
    def __init__(self, air: np.ndarray) -> None:
        self.air = air
        self.times: list[float] = []
        self.candidates: list[tuple[int, float, float]] = []
        self._index = 0

    def consume(self, stamp: float, boxes: list[dict]) -> None:
        for box in boxes:
            if box["conf"] < 0.05:
                continue
            if not (4.0 <= box["w"] <= BALL_MAX_WIDTH_FRAC * box["frame_w"]):
                continue
            if point_in_polygon(box["cx"], box["cy"], self.air):
                self.candidates.append((self._index, box["cx"], box["cy"]))
        self.times.append(stamp)
        self._index += 1

    def values(self, fps: float) -> list[float]:
        n_frames = len(self.times)
        kept = suppress_static(self.candidates, n_frames, fps)
        supported = temporal_support([frame for frame, _x, _y in kept], fps)
        return [1.0 if index in supported else 0.0 for index in range(n_frames)]


class _FrameDiff:
    def __init__(self, person_zone: np.ndarray, frame_width: int) -> None:
        self.times: list[float] = []
        self.values: list[float] = []
        self._prev: np.ndarray | None = None
        scale = 480.0 / max(1, frame_width)
        small_h = max(1, int(round(person_zone[:, 1].max() * scale))) if len(person_zone) else 1
        # Mask size is fixed from the first frame's aspect once we see it.
        self._scale = scale
        self._zone = person_zone
        self._mask: np.ndarray | None = None

    def consume(self, stamp: float, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        small_w = 480
        small_h = max(1, int(round(height * (480.0 / width))))
        if self._mask is None:
            self._mask = raster_mask(small_h, small_w, scale_polygon(self._zone, 480.0 / width, small_h / height))
            if int(self._mask.sum()) < 16:
                self._mask[:] = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        value = 0.0
        if self._prev is not None and self._prev.shape == blurred.shape:
            changed = cv2.absdiff(blurred, self._prev) > 25
            inside = changed & self._mask
            denom = int(self._mask.sum())
            value = float(inside.sum()) / denom if denom else 0.0
        self._prev = blurred
        self.times.append(stamp)
        self.values.append(value)


class _PlayerTrack:
    def __init__(self, court: np.ndarray) -> None:
        self.court = court
        self.times: list[float] = []
        self.values: list[float] = []
        self._prev: dict[int, tuple[float, float, float]] = {}

    def consume(self, stamp: float, boxes: list[dict]) -> None:
        total = 0.0
        current: dict[int, tuple[float, float, float]] = {}
        for box in boxes:
            if box["h"] <= 1:
                continue
            foot_x = box["cx"]
            foot_y = box["cy"] + box["h"] / 2
            contour = np.asarray(self.court, dtype=np.float32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(contour, (float(foot_x), float(foot_y)), True) < -15:
                continue
            track_id = box["id"]
            current[track_id] = (stamp, foot_x, foot_y)
            previous = self._prev.get(track_id)
            if previous is None or track_id < 0:
                continue
            dt = stamp - previous[0]
            if dt <= 0:
                continue
            speed = math.hypot(foot_x - previous[1], foot_y - previous[2]) / dt / box["h"]
            total += min(3.0, speed)
        self._prev = current
        self.times.append(stamp)
        self.values.append(total)


def _band_index() -> np.ndarray:
    from analyze import SAMPLE_RATE, WHISTLE_HIGH_HZ, WHISTLE_LOW_HZ, WHISTLE_WIN

    freqs = np.fft.rfftfreq(WHISTLE_WIN, 1 / SAMPLE_RATE)
    return (freqs >= WHISTLE_LOW_HZ) & (freqs <= WHISTLE_HIGH_HZ)


def _flux_blocks(path: str, window_start: float, band_only: bool) -> tuple[list[float], list[float]]:
    """95th percentile of spectral flux in each second, read in blocks."""
    from analyze import SAMPLE_RATE, WHISTLE_HOP, WHISTLE_WIN

    n_fft = WHISTLE_WIN
    hop = WHISTLE_HOP
    rate = SAMPLE_RATE
    window = np.hanning(n_fft)
    band = _band_index() if band_only else None
    times: list[float] = []
    fluxes: list[float] = []
    prev = None
    carry = np.zeros(0, dtype=np.float32)
    sample_origin = 0
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != rate:
            raise RuntimeError("Audio extract was not mono 16 kHz.")
        block = rate * 30
        while True:
            raw = handle.readframes(block)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            data = np.concatenate([carry, chunk]) if carry.size else chunk
            limit = len(data) - n_fft
            cursor = 0
            while cursor <= limit:
                piece = data[cursor : cursor + n_fft] * window
                mag = np.log1p(np.abs(np.fft.rfft(piece)))
                if prev is not None:
                    delta = (mag[band] - prev[band]) if band is not None else (mag - prev)
                    fluxes.append(float(delta[delta > 0].sum()))
                    center = (sample_origin + cursor + n_fft / 2) / rate
                    times.append(window_start + center)
                prev = mag
                cursor += hop
            consumed = cursor
            sample_origin += consumed
            carry = data[consumed:]
    return times, fluxes


def _ratio_blocks(path: str, window_start: float) -> tuple[list[float], list[float]]:
    """Whistle-band energy divided by the rest of the spectrum, per hop."""
    from analyze import SAMPLE_RATE, WHISTLE_HOP, WHISTLE_WIN

    n_fft = WHISTLE_WIN
    hop = WHISTLE_HOP
    rate = SAMPLE_RATE
    window = np.hanning(n_fft)
    band = _band_index()
    times: list[float] = []
    ratios: list[float] = []
    carry = np.zeros(0, dtype=np.float32)
    sample_origin = 0
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != rate:
            raise RuntimeError("Audio extract was not mono 16 kHz.")
        block = rate * 30
        while True:
            raw = handle.readframes(block)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            data = np.concatenate([carry, chunk]) if carry.size else chunk
            limit = len(data) - n_fft
            cursor = 0
            while cursor <= limit:
                mag = np.abs(np.fft.rfft(data[cursor : cursor + n_fft] * window))
                rest = float(np.mean(mag[~band])) + 1e-8
                ratios.append(float(np.mean(mag[band]) / rest))
                times.append(window_start + (sample_origin + cursor + n_fft / 2) / rate)
                cursor += hop
            sample_origin += cursor
            carry = data[cursor:]
    return times, ratios


def _onset_seconds(times: list[float], ratios: list[float], origin: float, n_bins: int) -> np.ndarray:
    from rallycompare.post import onset_indices

    pulses = np.zeros(n_bins, dtype=np.float64)
    for index in onset_indices(np.asarray(ratios, dtype=np.float64)):
        second = int(math.floor(times[index] - origin))
        if 0 <= second < n_bins:
            pulses[second] = 1.0
    return pulses


def _per_second_percentile(times: list[float], values: list[float], origin: float, n_bins: int) -> np.ndarray:
    buckets: list[list[float]] = [[] for _ in range(n_bins)]
    for stamp, value in zip(times, values):
        index = int(math.floor(stamp - origin))
        if 0 <= index < n_bins:
            buckets[index].append(value)
    out = np.zeros(n_bins, dtype=np.float64)
    for index, bucket in enumerate(buckets):
        if bucket:
            out[index] = float(np.percentile(bucket, 95))
    return out


def extract_audio(video: str, start: float, end: float, dest: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH")
    duration = max(0.1, end - start)
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", video, "-vn", "-ac", "1", "-ar", "16000",
        str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 64:
        detail = (result.stderr or "no audio track").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "no audio track")


def draw_court_check(frame: np.ndarray, court: np.ndarray, person: np.ndarray, air: np.ndarray, dest: Path) -> None:
    image = frame.copy()
    def poly(points: np.ndarray, color: tuple[int, int, int]) -> None:
        contour = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [contour], True, color, 2)
    poly(air, (0, 200, 255))
    poly(person, (255, 180, 0))
    poly(court, (0, 220, 0))
    cv2.imwrite(str(dest), image)


def extract_signals(
    video: str,
    start: float,
    end: float,
    court: np.ndarray | None,
    methods: list[str],
    sample_fps: float,
    weights: str,
    ball_imgsz: int,
    person_imgsz: int,
    body_margin: float,
    air_margin: float,
    fuse_inputs: list[str],
    out_dir: Path,
    log,
) -> ExtractResult:
    duration, _fps, width, height = video_bounds(video)
    if end <= 0 or end > duration > 0:
        end = duration if duration > 0 else end
    if end <= start:
        raise RuntimeError("The selected range is empty.")
    origin = float(math.floor(start))
    n_bins = max(1, int(math.ceil(end) - origin))
    result = ExtractResult(origin=origin, start=start, end=end)

    selected = set(methods)
    if "fused" in selected:
        selected.update(name for name in fuse_inputs if name in FUSE_INPUTS)
    need_ball = bool(selected & {"ball_baseline", "ball_filtered"})
    need_person = "player_track" in selected
    need_diff = "frame_diff" in selected
    visual = need_ball or need_person or need_diff

    ball_model = person_model = None
    if need_ball or need_person:
        try:
            ball_model, person_model = _load_models(weights, need_ball, need_person)
        except Exception as exc:
            reason = f"could not load {weights}: {exc}"
            if need_ball:
                result.skipped["ball_baseline"] = reason
                result.skipped["ball_filtered"] = reason
                need_ball = False
            if need_person:
                result.skipped["player_track"] = reason
                need_person = False

    court_poly = court if court is not None else full_frame_polygon(max(width, 2), max(height, 2))
    baseline = _BallBaseline()
    filtered: _BallFiltered | None = None
    differ: _FrameDiff | None = None
    tracker: _PlayerTrack | None = None
    device = _device()
    first = None

    if visual:
        for stamp, frame in iter_frames(video, start, end, sample_fps, log):
            if first is None:
                first = frame
                height, width = frame.shape[:2]
                if court is None:
                    court_poly = full_frame_polygon(width, height)
                person_zone = zone_above(court_poly, body_margin * height)
                air_zone = zone_above(court_poly, air_margin * height)
                result.width, result.height = width, height
                draw_court_check(frame, court_poly, person_zone, air_zone, out_dir / "court_check.jpg")
                if need_ball and "ball_filtered" in selected and "ball_filtered" not in result.skipped:
                    filtered = _BallFiltered(air_zone)
                if need_diff and "frame_diff" not in result.skipped:
                    differ = _FrameDiff(person_zone, width)
                if need_person and "player_track" not in result.skipped:
                    tracker = _PlayerTrack(court_poly.astype(np.float32))
            ball_boxes: list[dict] = []
            if need_ball and ball_model is not None and "ball_baseline" not in result.skipped:
                try:
                    prediction = ball_model.predict(
                        source=frame,
                        classes=[COCO_SPORTS_BALL],
                        conf=0.05 if "ball_filtered" in selected else PREDICT_CONF,
                        imgsz=ball_imgsz,
                        device=device,
                        verbose=False,
                    )[0]
                    ball_boxes = _boxes(prediction)
                except Exception as exc:
                    result.skipped["ball_baseline"] = str(exc)
                    result.skipped["ball_filtered"] = str(exc)
                    need_ball = False
                    ball_boxes = []
            if "ball_baseline" in selected and "ball_baseline" not in result.skipped:
                try:
                    baseline.consume(stamp, ball_boxes)
                except Exception as exc:
                    result.skipped["ball_baseline"] = str(exc)
            if filtered is not None and "ball_filtered" not in result.skipped:
                try:
                    filtered.consume(stamp, ball_boxes)
                except Exception as exc:
                    result.skipped["ball_filtered"] = str(exc)
                    filtered = None
            if differ is not None and "frame_diff" not in result.skipped:
                try:
                    differ.consume(stamp, frame)
                except Exception as exc:
                    result.skipped["frame_diff"] = str(exc)
                    differ = None
            if tracker is not None and person_model is not None and "player_track" not in result.skipped:
                try:
                    prediction = person_model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[0],
                        conf=0.25,
                        imgsz=person_imgsz,
                        device=device,
                        verbose=False,
                    )[0]
                    tracker.consume(stamp, _boxes(prediction))
                except Exception as exc:
                    result.skipped["player_track"] = str(exc)
                    tracker = None
        if first is None:
            for name in ("ball_baseline", "ball_filtered", "frame_diff", "player_track"):
                if name in selected:
                    result.skipped[name] = "no frames decoded"
    else:
        cap = cv2.VideoCapture(video)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            if court is None:
                court_poly = full_frame_polygon(width, height)
            person_zone = zone_above(court_poly, body_margin * height)
            air_zone = zone_above(court_poly, air_margin * height)
            result.width, result.height = width, height
            draw_court_check(frame, court_poly, person_zone, air_zone, out_dir / "court_check.jpg")

    if "ball_baseline" in selected and "ball_baseline" not in result.skipped and baseline.times:
        result.raw["ball_baseline"] = _bin(baseline.times, baseline.values, origin, n_bins)
    if filtered is not None and "ball_filtered" not in result.skipped and filtered.times:
        result.raw["ball_filtered"] = _bin(filtered.times, filtered.values(sample_fps), origin, n_bins)
    if differ is not None and "frame_diff" not in result.skipped and differ.times:
        result.raw["frame_diff"] = _bin(differ.times, differ.values, origin, n_bins)
    if tracker is not None and "player_track" not in result.skipped and tracker.times:
        result.raw["player_track"] = _bin(tracker.times, tracker.values, origin, n_bins)

    audio_methods = [name for name in ("audio", "contact", "ratio", "onset") if name in selected]
    if audio_methods:
        wav = out_dir / "_audio.wav"
        try:
            log("extract audio")
            extract_audio(video, start, end, wav)
            if "audio" in selected:
                flux_t, flux_v = _flux_blocks(str(wav), start, band_only=False)
                result.raw["audio"] = _per_second_percentile(flux_t, flux_v, origin, n_bins)
            if "contact" in selected:
                flux_t, flux_v = _flux_blocks(str(wav), start, band_only=True)
                result.raw["contact"] = _per_second_percentile(flux_t, flux_v, origin, n_bins)
            if "ratio" in selected or "onset" in selected:
                ratio_t, ratio_v = _ratio_blocks(str(wav), start)
                if "ratio" in selected:
                    result.raw["ratio"] = _per_second_percentile(ratio_t, ratio_v, origin, n_bins)
                if "onset" in selected:
                    result.raw["onset"] = _onset_seconds(ratio_t, ratio_v, origin, n_bins)
        except Exception as exc:
            for name in audio_methods:
                result.skipped[name] = str(exc)
        finally:
            if wav.exists():
                wav.unlink()

    if "fused" in methods:
        parts = []
        for name in fuse_inputs:
            if name in result.raw:
                smoothed = normalise_percentile(_smooth3(result.raw[name]))
                parts.append(smoothed)
        if parts:
            stacked = np.vstack(parts)
            result.raw["fused"] = stacked.mean(axis=0)
        else:
            result.skipped["fused"] = "no fuse inputs were available"

    for name in methods:
        if name not in result.raw and name not in result.skipped:
            result.skipped[name] = "not produced"
    return result


def _smooth3(signal: np.ndarray) -> np.ndarray:
    from rallycompare.post import moving_average

    return moving_average(signal, 3)


def save_cache(path: Path, extracted: ExtractResult) -> None:
    payload = {
        "origin": np.array([extracted.origin]),
        "start": np.array([extracted.start]),
        "end": np.array([extracted.end]),
        "methods": np.array(list(extracted.raw.keys())),
    }
    for name, values in extracted.raw.items():
        payload[f"raw_{name}"] = values
    skipped_names = list(extracted.skipped.keys())
    skipped_reasons = [extracted.skipped[name] for name in skipped_names]
    payload["skipped_names"] = np.array(skipped_names)
    payload["skipped_reasons"] = np.array(skipped_reasons)
    np.savez(path, **payload)


def load_cache(path: Path) -> ExtractResult:
    data = np.load(path, allow_pickle=False)
    result = ExtractResult(
        origin=float(data["origin"][0]),
        start=float(data["start"][0]),
        end=float(data["end"][0]),
    )
    for name in data["methods"].tolist():
        result.raw[str(name)] = np.asarray(data[f"raw_{name}"], dtype=np.float64)
    names = [str(item) for item in data["skipped_names"].tolist()]
    reasons = [str(item) for item in data["skipped_reasons"].tolist()]
    result.skipped = dict(zip(names, reasons))
    return result
