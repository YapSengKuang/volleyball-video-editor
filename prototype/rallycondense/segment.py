"""DEAD → SERVE_READY → IN_PLAY → DEAD, then pad, drop, and merge."""

from __future__ import annotations

from dataclasses import dataclass

# A rally shorter than this, before padding, is a touch or a false start.
MIN_RALLY_S = 2.5
# Quiet this long and the point is over.
QUIET_S = 2
# Join two rallies when the gap between them is shorter than this.
MERGE_GAP_S = 2.5
PRE_ROLL_S = 1.5
POST_ROLL_S = 2.0


@dataclass
class Second:
    motion: float = 0.0
    ball_count: int = 0
    ball_speed: float = 0.0
    players_left: int = 0
    players_right: int = 0
    server_ready: bool = False
    audio_onset: bool = False
    net_crossing: bool = False


@dataclass
class Segment:
    start: float
    end: float
    source: str = "auto"

    @property
    def duration(self) -> float:
        return self.end - self.start


def _in_play(sample: Second, motion_threshold: float) -> bool:
    ball_moving = sample.ball_count > 0 and sample.ball_speed >= 0.15
    court_active = sample.motion >= motion_threshold
    return ball_moving or sample.net_crossing or (court_active and sample.audio_onset) or (
        court_active and sample.ball_count > 0
    )


def _motion_threshold(samples: list[Second]) -> float:
    if not samples:
        return 1e9
    values = sorted(sample.motion for sample in samples)
    peak = values[min(len(values) - 1, int(0.98 * (len(values) - 1)))]
    low = values[min(len(values) - 1, int(0.20 * (len(values) - 1)))]
    if peak < 4:
        return 1e9
    return low + 0.35 * (peak - low)


def detect_rallies(samples: list[Second]) -> list[tuple[float, float]]:
    """Return raw rally windows in seconds, before padding."""
    threshold = _motion_threshold(samples)
    state = "DEAD"
    start = 0
    quiet = 0
    raw: list[tuple[float, float]] = []

    def close(end: int) -> None:
        if end - start >= MIN_RALLY_S:
            raw.append((float(start), float(end)))

    for index, sample in enumerate(samples):
        playing = _in_play(sample, threshold)
        if state == "DEAD":
            if sample.server_ready and not playing:
                state = "SERVE_READY"
                start = index
                quiet = 0
            elif playing:
                state = "IN_PLAY"
                start = index
                quiet = 0
            continue
        if state == "SERVE_READY":
            if playing:
                state = "IN_PLAY"
                quiet = 0
            elif index - start > 8:
                state = "DEAD"
            continue
        if playing:
            quiet = 0
            continue
        quiet += 1
        if quiet >= QUIET_S:
            close(index - quiet + 1)
            state = "DEAD"
            quiet = 0
    if state == "IN_PLAY":
        close(len(samples) - quiet)
    return raw


def clean_segments(
    raw: list[tuple[float, float]],
    duration: float,
    pre_roll: float = PRE_ROLL_S,
    post_roll: float = POST_ROLL_S,
    merge_gap: float = MERGE_GAP_S,
) -> list[Segment]:
    if not raw:
        return []
    padded = [(max(0.0, start - pre_roll), min(duration, end + post_roll)) for start, end in raw]
    merged: list[tuple[float, float]] = []
    for start, end in padded:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [Segment(start, end) for start, end in merged if end - start >= 0.5]
