"""Pure tests for the comparison harness, plus a tiny frame-diff and audio run."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

from rallycompare.metrics import duration_stats, jaccard, second_scores, segment_scores
from rallycompare.post import (
    hold_pulses,
    moving_average,
    onset_indices,
    otsu_threshold,
    segment_signal,
    suppress_static,
    temporal_support,
)


def test_state_machine_merges_pads_and_drops_short_runs() -> None:
    signal = np.zeros(20)
    signal[2:6] = 1
    signal[8:12] = 1
    signal[18:20] = 1
    segments = segment_signal(signal, 0, 0.5, min_dur=4, merge_gap=4, pad=2, limit_start=0, limit_end=20)
    assert segments == [(0.0, 14.0)]


def test_state_machine_padding_merges_overlaps() -> None:
    signal = np.zeros(30)
    signal[10:14] = 1
    signal[16:20] = 1
    segments = segment_signal(signal, 100, 0.5, min_dur=4, merge_gap=0, pad=2, limit_start=100, limit_end=130)
    assert segments == [(108.0, 122.0)]


def test_otsu_splits_two_levels() -> None:
    signal = np.array([0.0] * 40 + [1.0] * 40)
    threshold = otsu_threshold(signal)
    assert 0.0 < threshold < 1.0
    assert int(np.sum(signal >= threshold)) == 40


def test_moving_average_keeps_length() -> None:
    signal = np.arange(10, dtype=np.float64)
    assert len(moving_average(signal, 5)) == 10
    assert len(moving_average(signal, 3)) == 10


def test_evaluation_metrics() -> None:
    scores = second_scores(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 0, 0, 1]))
    assert scores["tp"] == 2
    assert scores["fp"] == 1
    assert scores["fn"] == 1
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    segs = segment_scores([(0, 10), (20, 24)], [(0, 10), (30, 40)])
    assert segs["segment_recall"] == pytest.approx(0.5)
    assert segs["segment_precision"] == pytest.approx(0.5)
    assert jaccard(np.array([1, 1, 0]), np.array([1, 0, 0])) == pytest.approx(0.5)
    stats = duration_stats([(0, 10), (20, 30)], 100)
    assert stats["n_rallies"] == 2
    assert stats["in_play_pct"] == pytest.approx(20)


def test_onset_hold_ignores_a_flat_roar_and_keeps_a_jump() -> None:
    roar = np.linspace(1.0, 1.2, 20)
    assert onset_indices(roar) == []
    hit = np.array([1.0, 1.0, 1.0, 4.0, 1.1])
    assert 3 in onset_indices(hit, percentile=50, jump=1.4)
    assert onset_indices(np.array([0.0, 5.0, 5.0]), percentile=50, jump=1.4) == [1]
    held = hold_pulses(np.array([0, 1, 0, 0, 0, 0], dtype=np.float64), 3)
    assert held.tolist() == [0, 1, 1, 1, 0, 0]


def test_static_mask_rejects_a_fixed_speck_and_keeps_a_moving_ball() -> None:
    fps = 4
    n_frames = 40
    stationary = [(frame, 100.0, 100.0) for frame in range(n_frames)]
    assert suppress_static(stationary, n_frames, fps) == []
    moving = [(0, 0.0, 0.0), (1, 100.0, 0.0), (2, 200.0, 0.0)]
    kept = suppress_static(moving, n_frames, fps)
    assert [frame for frame, _x, _y in kept] == [0, 1, 2]
    assert temporal_support([frame for frame, _x, _y in kept], fps) == {0, 1, 2}
    assert temporal_support([0], fps) == set()


def _write_click_video(directory: Path) -> Path:
    video = directory / "silent.avi"
    audio = directory / "clicks.wav"
    final = directory / "rally.avi"
    width, height, fps, seconds = 320, 180, 10, 12
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    for frame_no in range(seconds * fps):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        second = frame_no / fps
        if 3.0 <= second < 8.0:
            x = int((second - 3.0) * 40) % (width - 30)
            cv2.rectangle(image, (x, 40), (x + 28, 70), (255, 255, 255), -1)
        writer.write(image)
    writer.release()

    rate = 16000
    samples = np.zeros(rate * seconds, dtype=np.int16)
    for second in np.arange(3.0, 8.0, 0.25):
        begin = int(second * rate)
        burst = np.random.default_rng(0).integers(-8000, 8000, 400, dtype=np.int16)
        samples[begin : begin + 400] = burst
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    if shutil.which("ffmpeg") is None:
        return video
    muxed = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio),
            "-c:v", "copy", "-c:a", "pcm_s16le", "-shortest", str(final),
        ],
        capture_output=True,
        text=True,
    )
    return final if muxed.returncode == 0 else video


def test_frame_diff_and_audio_smoke(tmp_path: Path) -> None:
    from rallycompare.__main__ import main

    source = _write_click_video(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("start,end\n3,8\n", encoding="utf-8")
    out = tmp_path / "out"
    code = main(
        [
            str(source),
            "--methods", "frame_diff,audio",
            "--labels", str(labels),
            "--out", str(out),
            "--sample-fps", "4",
            "--min-dur", "1",
            "--merge-gap", "1",
            "--pad", "0.5",
        ]
    )
    assert code == 0
    assert (out / "report.html").exists()
    assert (out / "summary.csv").exists()
    summary = (out / "summary.csv").read_text(encoding="utf-8")
    assert "frame_diff,ok" in summary
    assert "audio," in summary
    assert "frame_diff" in (out / "segments.json").read_text(encoding="utf-8")
    assert (out / "court_check.jpg").exists()

    code = main(
        [
            str(source),
            "--methods", "frame_diff,audio",
            "--labels", str(labels),
            "--out", str(out),
            "--reuse",
            "--thresh", "frame_diff=0.01",
            "--min-dur", "1",
            "--merge-gap", "1",
            "--pad", "0.5",
        ]
    )
    assert code == 0
    assert "0.0100" in (out / "report.html").read_text(encoding="utf-8")
