"""Console table, CSV, JSON, and a single self-contained HTML report."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from rallycompare.metrics import duration_stats, in_play_mask, jaccard, second_scores, segment_scores


def _clock(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    return f"{whole // 60}:{whole % 60:02d}"


def _lane_svg(
    name: str,
    signal: np.ndarray | None,
    threshold: float | None,
    segments: list[tuple[float, float]],
    origin: float,
    start: float,
    end: float,
    y: int,
    width: int,
) -> str:
    span = max(1e-6, end - start)
    height = 52
    top = y

    def x_of(second: float) -> float:
        return 120 + (second - start) / span * (width - 140)

    parts = [f'<g class="lane">']
    parts.append(
        f'<text class="label" x="8" y="{top + 30}">{html.escape(name)}</text>'
    )
    parts.append(
        f'<rect class="lane-bg" x="120" y="{top + 4}" width="{width - 140}" height="{height - 10}" rx="4"/>'
    )
    for seg_start, seg_end in segments:
        x0 = x_of(max(start, seg_start))
        x1 = x_of(min(end, seg_end))
        if x1 > x0:
            parts.append(
                f'<rect class="rally" x="{x0:.1f}" y="{top + 4}" width="{x1 - x0:.1f}" height="{height - 10}"/>'
            )
    if signal is not None and signal.size:
        peak = float(max(np.max(signal), threshold or 0.0, 1e-6))
        points = []
        step = max(1, len(signal) // 800)
        for index in range(0, len(signal), step):
            second = origin + index
            if second < start or second > end:
                continue
            yy = top + 4 + (height - 10) * (1 - float(signal[index]) / peak)
            points.append(f"{x_of(second):.1f},{yy:.1f}")
        if points:
            parts.append(f'<polyline class="signal" points="{" ".join(points)}"/>')
        if threshold is not None:
            yy = top + 4 + (height - 10) * (1 - threshold / peak)
            parts.append(
                f'<line class="threshold" x1="120" x2="{width - 20}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
            )
    parts.append("</g>")
    return "\n".join(parts)


def render_html(
    path: Path,
    origin: float,
    start: float,
    end: float,
    order: list[str],
    smoothed: dict[str, np.ndarray],
    thresholds: dict[str, float],
    segments: dict[str, list[tuple[float, float]]],
    rows: list[dict],
    matrix: list[list[float]],
    matrix_names: list[str],
    skipped: dict[str, str],
    labels: list[tuple[float, float]] | None,
) -> None:
    width = 1100
    lanes = []
    y = 36
    if labels is not None:
        lanes.append(_lane_svg("ground truth", None, None, labels, origin, start, end, y, width))
        y += 58
    for name in order:
        lanes.append(
            _lane_svg(
                name,
                smoothed.get(name),
                thresholds.get(name),
                segments.get(name, []),
                origin,
                start,
                end,
                y,
                width,
            )
        )
        y += 58
    axis = []
    span = max(1e-6, end - start)
    ticks = 6
    for tick in range(ticks + 1):
        second = start + span * tick / ticks
        x = 120 + (width - 140) * tick / ticks
        axis.append(
            f'<text class="tick" x="{x:.1f}" y="{y + 18}" text-anchor="middle">{_clock(second)}</text>'
        )
    height = y + 36

    def cell(value: object) -> str:
        if value is None or value == "":
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return html.escape(str(value))

    best = None
    scored = [row["f1"] for row in rows if isinstance(row.get("f1"), float)]
    if scored:
        best = max(scored)
    header = ["method", "rallies", "in play %", "mean s", "median s", "precision", "recall", "F1", "IoU", "seg precision", "seg recall"]
    body = []
    for row in rows:
        highlight = isinstance(row.get("f1"), float) and best is not None and abs(row["f1"] - best) < 1e-9
        klass = ' class="best"' if highlight else ""
        keys = [
            "method", "n_rallies", "in_play_pct", "mean_dur", "median_dur",
            "precision", "recall", "f1", "iou", "segment_precision", "segment_recall",
        ]
        tds = "".join(f"<td>{cell(row.get(key, ''))}</td>" for key in keys)
        body.append(f"<tr{klass}>{tds}</tr>")

    heat_head = "".join(f"<th>{html.escape(name)}</th>" for name in matrix_names)
    heat_rows = []
    for name, line in zip(matrix_names, matrix):
        cells = "".join(
            f'<td style="--heat:{value:.3f}">{value:.2f}</td>' for value in line
        )
        heat_rows.append(f"<tr><th>{html.escape(name)}</th>{cells}</tr>")

    notes = []
    for name, reason in skipped.items():
        notes.append(f"<li><strong>{html.escape(name)}</strong> skipped: {html.escape(reason)}</li>")
    for name, value in thresholds.items():
        notes.append(f"<li><strong>{html.escape(name)}</strong> threshold {value:.4f}</li>")
    if not notes:
        notes.append("<li>Every requested method produced a signal.</li>")

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Rally signal comparison</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #f6f4ef;
  --ink: #1c1915;
  --muted: #5c564c;
  --card: #fffdf8;
  --line: #d9d3c7;
  --rally: rgba(36, 116, 78, 0.35);
  --signal: #1d4e89;
  --threshold: #9a3b2f;
  --best: #e7f6ee;
  --heat: 0;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #161411;
    --ink: #f3efe6;
    --muted: #b7b0a4;
    --card: #221f1b;
    --line: #3a342c;
    --rally: rgba(126, 196, 154, 0.35);
    --signal: #8eb6e8;
    --threshold: #e39b90;
    --best: #1c3a2a;
  }}
}}
body {{ margin: 0; padding: 24px; background: var(--bg); color: var(--ink);
  font: 14px/1.4 "Iowan Old Style", Palatino, Georgia, serif; }}
h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 4px; }}
p.sub {{ color: var(--muted); margin: 0 0 20px; }}
section {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px; margin: 0 0 16px; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
tr.best td {{ background: var(--best); font-weight: 600; }}
.heat td {{ background: color-mix(in srgb, var(--signal) calc(var(--heat) * 100%), transparent); }}
svg {{ width: 100%; height: auto; }}
.lane-bg {{ fill: transparent; stroke: var(--line); }}
.rally {{ fill: var(--rally); }}
.signal {{ fill: none; stroke: var(--signal); stroke-width: 1.5; }}
.threshold {{ stroke: var(--threshold); stroke-dasharray: 4 3; stroke-width: 1; }}
.label, .tick {{ fill: var(--muted); font-size: 11px; font-family: inherit; }}
</style>
</head>
<body>
<h1>Rally signal comparison</h1>
<p class="sub">{_clock(start)}–{_clock(end)} · court in green, person zone in blue, air zone in amber on court_check.jpg</p>
<section>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Timeline of each signal">
{''.join(lanes)}
{''.join(axis)}
</svg>
</section>
<section>
<table>
<thead><tr>{''.join(f'<th>{html.escape(item)}</th>' for item in header)}</tr></thead>
<tbody>
{''.join(body)}
</tbody>
</table>
</section>
<section>
<table class="heat">
<thead><tr><th></th>{heat_head}</tr></thead>
<tbody>
{''.join(heat_rows)}
</tbody>
</table>
</section>
<section>
<ul>
{''.join(notes)}
</ul>
</section>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "method", "status", "threshold", "n_rallies", "n_true", "in_play_pct",
        "mean_dur", "median_dur", "precision", "recall", "f1", "iou",
        "segment_precision", "segment_recall",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_signals_csv(
    path: Path,
    origin: float,
    names: list[str],
    raw: dict[str, np.ndarray],
    smoothed: dict[str, np.ndarray],
    playing: dict[str, np.ndarray],
) -> None:
    length = 0
    for name in names:
        if name in raw:
            length = max(length, len(raw[name]))
    headers = ["time_s"]
    for name in names:
        headers.extend([f"{name}_raw", f"{name}_smoothed", f"{name}_in_play"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(length):
            row: list[object] = [f"{origin + index:.0f}"]
            for name in names:
                series = raw.get(name)
                smooth = smoothed.get(name)
                flags = playing.get(name)
                row.append("" if series is None or index >= len(series) else f"{series[index]:.6f}")
                row.append("" if smooth is None or index >= len(smooth) else f"{smooth[index]:.6f}")
                row.append("" if flags is None or index >= len(flags) else int(flags[index]))
            writer.writerow(row)


def write_segments(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_rows(
    order: list[str],
    segments: dict[str, list[tuple[float, float]]],
    thresholds: dict[str, float],
    skipped: dict[str, str],
    window: float,
    labels: list[tuple[float, float]] | None,
    origin: float,
    n_bins: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    playing: dict[str, np.ndarray] = {}
    truth = None
    if labels is not None:
        truth = in_play_mask(labels, origin, n_bins)
    rows = []
    for name in order:
        if name in skipped and name not in segments:
            rows.append({"method": name, "status": f"skipped: {skipped[name]}"})
            continue
        stats = duration_stats(segments.get(name, []), window)
        flags = in_play_mask(segments.get(name, []), origin, n_bins)
        playing[name] = flags
        row = {"method": name, "status": "ok", "threshold": thresholds.get(name, ""), **stats}
        if truth is not None:
            seconds = second_scores(flags, truth)
            segs = segment_scores(segments.get(name, []), labels or [])
            row.update(seconds)
            row.update(segs)
        rows.append(row)
    return rows, playing


def agreement(names: list[str], playing: dict[str, np.ndarray]) -> list[list[float]]:
    matrix = []
    for left in names:
        matrix.append([
            jaccard(playing[left], playing[right]) if left in playing and right in playing else 0.0
            for right in names
        ])
    return matrix


def print_table(rows: list[dict]) -> None:
    print(f"{'method':<16} {'rallies':>7} {'in_play%':>8} {'mean':>7} {'F1':>7} {'recall':>7} status")
    for row in rows:
        f1 = row.get("f1")
        recall = row.get("recall")
        print(
            f"{row['method']:<16} "
            f"{row.get('n_rallies', ''):>7} "
            f"{row.get('in_play_pct', 0):8.1f} "
            f"{row.get('mean_dur', 0):7.1f} "
            f"{(f1 if isinstance(f1, float) else 0):7.3f} "
            f"{(recall if isinstance(recall, float) else 0):7.3f} "
            f"{row.get('status', '')}"
        )
