"""Leave-one-match-out training for the court-masked rally model.

Court polygons below are estimates for the three labeled files. On the site,
the user clicks the four corners and those clicks are what the cutter uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

WORKER = Path(__file__).resolve().parents[1] / "worker"
sys.path.insert(0, str(WORKER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learned_cut import (  # noqa: E402
    feature_table,
    fit_logit,
    predict_proba,
    rallies_from_probs,
    window_matrix,
)
from rallycondense.evaluate import report  # noqa: E402
from rallycondense.segment import Segment  # noqa: E402


def parse_clock(value: str) -> float:
    parts = [float(piece) for piece in value.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(value)


def load_labels(path: str) -> list[Segment]:
    rallies = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 2:
            continue
        try:
            start, end = parse_clock(cells[0]), parse_clock(cells[1])
        except ValueError:
            continue
        if end > start:
            rallies.append(Segment(start, end, "user"))
    return rallies


def labels_vector(rallies: list[Segment], seconds: int) -> np.ndarray:
    target = np.zeros(seconds, dtype=np.float64)
    for rally in rallies:
        begin = max(0, int(rally.start))
        finish = min(seconds, int(rally.end) + 1)
        target[begin:finish] = 1.0
    return target


def load_match(spec: dict) -> dict:
    rallies = load_labels(spec["labels"])
    duration = rallies[-1].end + 20
    cache = Path(__file__).resolve().parent / "rally_compare" / f"{spec['name']}-features.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        table = np.load(cache)
    else:
        print(f"features {spec['name']}", flush=True)
        table = feature_table(spec["video"], spec["court"], duration)
        np.save(cache, table)
    seconds = min(len(table), int(duration) + 1)
    table = table[:seconds]
    return {
        "name": spec["name"],
        "duration": duration,
        "rallies": rallies,
        "table": table,
        "target": labels_vector(rallies, seconds),
        "windows": window_matrix(table),
    }


def score_fold(held: dict, model: dict) -> dict:
    probs = predict_proba(held["windows"], model)
    segments = [
        Segment(start, end, "learned")
        for start, end in rallies_from_probs(probs, held["duration"])
    ]
    result = report(segments, held["rallies"], held["duration"])
    covered = result["matches"] / len(held["rallies"]) if held["rallies"] else 0.0
    return {**result, "segment_recall": covered, "n_true": len(held["rallies"]), "n_pred": len(segments)}


def main() -> int:
    manifest = json.loads((Path(__file__).resolve().parent / "matches.json").read_text(encoding="utf-8"))
    matches = [load_match(spec) for spec in manifest["matches"]]
    print(f"{'held out':<10} {'recall':>8} {'prec':>8} {'boundary':>10} {'rallies':>8}")
    for index, held in enumerate(matches):
        train = [item for item_index, item in enumerate(matches) if item_index != index]
        features = np.vstack([item["windows"] for item in train])
        target = np.concatenate([item["target"] for item in train])
        model = fit_logit(features, target)
        result = score_fold(held, model)
        boundary = result["typical_boundary_error"]
        boundary_text = "—" if boundary is None else f"{boundary:.1f}s"
        print(
            f"{held['name']:<10} {result['recall']:8.3f} {result['precision']:8.3f} "
            f"{boundary_text:>10} {result['n_pred']:>3}/{result['n_true']:<3}"
        )
    everything = fit_logit(
        np.vstack([item["windows"] for item in matches]),
        np.concatenate([item["target"] for item in matches]),
    )
    destination = WORKER / "models" / "rally_logit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(everything), encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
