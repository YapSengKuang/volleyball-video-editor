"""Score a prediction file against hand-labeled rallies.

Label file:
  {"duration": 3600, "rallies": [{"start": 12.0, "end": 27.5}]}
"""

from __future__ import annotations

import argparse
import json
import sys

from rallycondense.evaluate import report
from rallycondense.segment import Segment


def _load(path: str) -> tuple[list[Segment], float]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rallies = [Segment(float(item["start"]), float(item["end"]), item.get("source", "user")) for item in payload["rallies"]]
    duration = float(payload.get("duration") or max((item.end for item in rallies), default=0))
    return rallies, duration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score predicted rallies against labels.")
    parser.add_argument("--pred", required=True, help="Predicted segments JSON")
    parser.add_argument("--labels", required=True, help="Hand-labeled rallies JSON")
    args = parser.parse_args(argv)
    predicted, pred_duration = _load(args.pred)
    labeled, label_duration = _load(args.labels)
    result = report(predicted, labeled, max(pred_duration, label_duration))
    print(json.dumps(result, indent=2))
    print(
        "exit bar: recall >= 0.90 and typical boundary error <= 1.5s —",
        "pass" if result["meets_exit_bar"] else "not yet",
    )
    return 0 if result["meets_exit_bar"] else 1


if __name__ == "__main__":
    sys.exit(main())
