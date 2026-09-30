# Offline rally comparison

`rallycompare` scores several rally signals on the same stretch of video. It does not change how the clipper cuts a match.

```bash
cd prototype
python -m rallycompare /path/to/match.mov \
  --start 0:00 --end 8:00 \
  --court "120,700 1700,700 1500,280 320,280" \
  --labels labels.csv \
  --out rally_compare
```

`--court` is the court floor in original pixel coordinates. The person zone and the air zone are that polygon extended upward. `court_check.jpg` draws all three so the polygon can be checked.

`--labels` is optional. Columns are `start,end` as absolute video times (`90` or `1:30`). A header row is optional. See `labels_template.csv`.

Re-threshold without reading the video again:

```bash
python -m rallycompare /path/to/match.mov --reuse --thresh frame_diff=0.2 --out rally_compare
```

Methods: `ball_baseline`, `ball_filtered`, `frame_diff`, `player_track`, `audio`, `contact`, `ratio`, `onset`, `fused`.

`ratio` is the whistle band (2500–4500 Hz) divided by the rest of the spectrum, so a roar that lifts every frequency cancels out. `onset` marks a sudden jump in that ratio and holds the clip open for `--hold` seconds (default 8). Change the hold without re-reading the video:

```bash
python -m rallycompare /path/to/match.mov --methods ratio,onset --reuse --hold 12 --out rally_compare
``` A method that cannot run is marked `skipped:` in the report instead of stopping the others. Ball thresholds (`0.25` confidence, 8px to 18% of the frame width, minimum speed) come from `worker/ball.py`.
