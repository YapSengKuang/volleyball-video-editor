"""Compare rally signals on one video segment.

Example:
  python -m rallycompare match.mov --labels labels.csv --out rally_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from rallycompare.extract import (
    ALL_METHODS,
    BALL_METHODS,
    ExtractResult,
    extract_signals,
    load_cache,
    parse_clock,
    parse_labels,
    save_cache,
    video_bounds,
)
from rallycompare.post import hold_pulses, moving_average, otsu_threshold, segment_signal
from rallycompare.report import (
    agreement,
    build_rows,
    print_table,
    render_html,
    write_segments,
    write_signals_csv,
    write_summary_csv,
)


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rally-detection signals on one video range.")
    parser.add_argument("video", help="Path to the match video")
    parser.add_argument("--start", default="0", help="Start time, seconds or mm:ss")
    parser.add_argument("--end", default="", help="End time, seconds or mm:ss (default: end of video)")
    parser.add_argument("--court", default="", help='Court floor polygon in pixels: "x,y x,y x,y x,y"')
    parser.add_argument("--labels", default="", help="CSV of absolute start,end rally times")
    parser.add_argument("--out", default="rally_compare", help="Output directory")
    parser.add_argument("--methods", default=",".join(ALL_METHODS), help="Comma-separated method list")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--ball-imgsz", type=int, default=1280)
    parser.add_argument("--person-imgsz", type=int, default=640)
    parser.add_argument("--thresh", action="append", default=[], help="method=value, repeatable")
    parser.add_argument("--min-dur", type=float, default=4.0)
    parser.add_argument("--merge-gap", type=float, default=4.0)
    parser.add_argument("--pad", type=float, default=2.0)
    parser.add_argument("--body-margin", type=float, default=0.15)
    parser.add_argument("--air-margin", type=float, default=0.5)
    parser.add_argument("--fuse", default="frame_diff,player_track,audio")
    parser.add_argument("--hold", type=int, default=8, help="Seconds an onset keeps the clip open")
    parser.add_argument("--reuse", action="store_true", help="Reuse raw_signals.npz and only re-threshold")
    return parser.parse_args(argv)


def parse_court(text: str) -> np.ndarray | None:
    if not text.strip():
        return None
    points = []
    for token in text.split():
        x_text, y_text = token.split(",")
        points.append((float(x_text), float(y_text)))
    if len(points) < 3:
        raise ValueError("A court polygon needs at least 3 points.")
    return np.asarray(points, dtype=np.float64)


def parse_thresholds(items: list[str]) -> dict[str, float]:
    found = {}
    for item in items:
        name, value = item.split("=", 1)
        found[name.strip()] = float(value)
    return found


def finish(extracted: ExtractResult, methods: list[str], args: argparse.Namespace, labels, out_dir: Path) -> int:
    overrides = parse_thresholds(args.thresh)
    n_bins = 0
    for values in extracted.raw.values():
        n_bins = max(n_bins, len(values))
    smoothed: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    segments: dict[str, list[tuple[float, float]]] = {}
    skipped = dict(extracted.skipped)
    for name in methods:
        if name not in extracted.raw:
            skipped.setdefault(name, "not in cache; rerun without --reuse")
            continue
        window = 5 if name in BALL_METHODS else 3
        series = extracted.raw[name]
        if name == "onset":
            series = hold_pulses(series, args.hold)
        smooth = moving_average(series, window)
        smoothed[name] = smooth
        if name in overrides:
            threshold = overrides[name]
        elif name in BALL_METHODS:
            threshold = 0.10
        elif name == "onset":
            threshold = 0.5
        else:
            threshold = otsu_threshold(smooth)
        thresholds[name] = threshold
        segments[name] = segment_signal(
            smooth,
            extracted.origin,
            threshold,
            args.min_dur,
            args.merge_gap,
            args.pad,
            extracted.start,
            extracted.end,
        )
    window = max(0.0, extracted.end - extracted.start)
    rows, playing = build_rows(
        methods, segments, thresholds, skipped, window, labels, extracted.origin, n_bins
    )
    names = [name for name in methods if name in playing]
    matrix = agreement(names, playing)
    write_summary_csv(out_dir / "summary.csv", rows)
    write_signals_csv(
        out_dir / "signals.csv",
        extracted.origin,
        [name for name in methods if name in extracted.raw],
        extracted.raw,
        smoothed,
        playing,
    )
    write_segments(
        out_dir / "segments.json",
        {
            "start": extracted.start,
            "end": extracted.end,
            "origin": extracted.origin,
            "methods": {
                name: {
                    "threshold": thresholds.get(name),
                    "segments": [{"start": start, "end": end} for start, end in segments.get(name, [])],
                }
                for name in methods
                if name in segments
            },
            "skipped": skipped,
        },
    )
    render_html(
        out_dir / "report.html",
        extracted.origin,
        extracted.start,
        extracted.end,
        methods,
        smoothed,
        thresholds,
        segments,
        rows,
        matrix,
        names,
        skipped,
        labels,
    )
    print_table(rows)
    print(f"wrote {out_dir / 'report.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    start = parse_clock(args.start)
    end = parse_clock(args.end) if args.end else 0.0
    labels = parse_labels(args.labels) if args.labels else None
    cache = out_dir / "raw_signals.npz"
    if args.reuse and cache.exists():
        _log(f"reuse {cache}")
        extracted = load_cache(cache)
    else:
        if end <= 0:
            duration, _fps, _w, _h = video_bounds(args.video)
            end = duration
        extracted = extract_signals(
            args.video,
            start,
            end,
            parse_court(args.court),
            methods,
            args.sample_fps,
            args.weights,
            args.ball_imgsz,
            args.person_imgsz,
            args.body_margin,
            args.air_margin,
            [name.strip() for name in args.fuse.split(",") if name.strip()],
            out_dir,
            _log,
        )
        save_cache(cache, extracted)
    return finish(extracted, methods, args, labels, out_dir)


if __name__ == "__main__":
    sys.exit(main())
