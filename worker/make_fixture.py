"""Build a 60s game-shaped clip: still pictures, moving pictures, and whistle tones."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

# Still, move, still, move, still, move, still. Whistles sit on each move.
CLIPS = [
    ("color", 10),
    ("noise", 8),
    ("color", 10),
    ("noise", 8),
    ("color", 12),
    ("noise", 7),
    ("color", 5),
]
WHISTLES = (10.0, 28.0, 48.0)
DURATION = sum(item[1] for item in CLIPS)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-1200:])


def _wav(path: str) -> None:
    rate = 16000
    samples = np.zeros(rate * DURATION, dtype=np.float32)
    for start in WHISTLES:
        count = int(0.35 * rate)
        offset = int(start * rate)
        ticks = np.arange(count, dtype=np.float32) / rate
        samples[offset : offset + count] = 0.9 * np.sin(2 * np.pi * 3500 * ticks)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(path, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def make_fixture(path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "tone.wav")
        _wav(wav_path)
        clip_paths: list[str] = []
        for index, (kind, dur) in enumerate(CLIPS):
            clip = os.path.join(tmp, f"{index}.mp4")
            if kind == "color":
                src = f"color=c=0x163216:s=320x180:r=10:d={dur}"
            else:
                src = f"testsrc2=s=320x180:r=10:d={dur}"
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    src,
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "veryfast",
                    clip,
                ]
            )
            clip_paths.append(clip)
        listing = os.path.join(tmp, "list.txt")
        with open(listing, "w", encoding="utf-8") as handle:
            for clip in clip_paths:
                handle.write(f"file '{clip}'\n")
        video = os.path.join(tmp, "video.mp4")
        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                listing,
                "-c",
                "copy",
                video,
            ]
        )
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video,
                "-i",
                wav_path,
                "-c:v",
                "copy",
                "-c:a",
                "pcm_s16le",
                "-shortest",
                path,
            ]
        )


if __name__ == "__main__":
    destination = sys.argv[1] if len(sys.argv) > 1 else "fixture.mov"
    make_fixture(destination)
    print(destination)
