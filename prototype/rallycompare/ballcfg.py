"""Ball-gate numbers from the clipper, so the comparison does not invent its own."""

from __future__ import annotations

import sys
from pathlib import Path

_WORKER = Path(__file__).resolve().parents[2] / "worker"
if str(_WORKER) not in sys.path:
    sys.path.insert(0, str(_WORKER))

from ball import (  # noqa: E402
    BALL_MAX_WIDTH_FRAC,
    BALL_MIN_CONF,
    BALL_MIN_WIDTH_PX,
    COCO_SPORTS_BALL,
    MIN_SPEED,
    PREDICT_CONF,
)

__all__ = [
    "BALL_MAX_WIDTH_FRAC",
    "BALL_MIN_CONF",
    "BALL_MIN_WIDTH_PX",
    "COCO_SPORTS_BALL",
    "MIN_SPEED",
    "PREDICT_CONF",
]
