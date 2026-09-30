"""Court polygon, net line, and which side of the net a point is on."""

from __future__ import annotations

import numpy as np


def as_points(points: list) -> np.ndarray:
    pairs = []
    for point in points:
        if isinstance(point, dict):
            pairs.append((float(point["x"]), float(point["y"])))
        else:
            pairs.append((float(point[0]), float(point[1])))
    return np.asarray(pairs, dtype=np.float64)


def order_corners(points: list) -> np.ndarray:
    pts = as_points(points)
    if pts.shape != (4, 2):
        raise ValueError("A court needs four corners.")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def point_in_polygon(x: float, y: float, corners: list) -> bool:
    poly = order_corners(corners)
    sign = 0.0
    crosses = []
    cx, cy = poly.mean(axis=0)
    for index in range(4):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % 4]
        crosses.append((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))
        sign += (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
    want_positive = sign >= 0
    return all(value >= -1e-6 if want_positive else value <= 1e-6 for value in crosses)


def side_of_net(x: float, y: float, net: list) -> int:
    """Return -1 or 1 for the two sides of the net. 0 if the net is missing."""
    if not net or len(net) < 2:
        return 0
    ends = as_points(net[:2])
    x1, y1 = ends[0]
    x2, y2 = ends[1]
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if abs(cross) < 1e-8:
        return 0
    return 1 if cross > 0 else -1


def homography(corners: list) -> np.ndarray:
    """Map the court quadrilateral onto a unit rectangle. Image x,y to court u,v."""
    source = order_corners(corners)
    # After angle sort the order is around the boundary, not TL-TR-BR-BL.
    # Match each corner to the nearest unit-rectangle corner by angle from center.
    center = source.mean(axis=0)
    target = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    target_center = target.mean(axis=0)
    source_angles = np.arctan2(source[:, 1] - center[1], source[:, 0] - center[0])
    target_angles = np.arctan2(target[:, 1] - target_center[1], target[:, 0] - target_center[0])
    source = source[np.argsort(source_angles)]
    target = target[np.argsort(target_angles)]
    matrix = []
    values = []
    for (x, y), (u, v) in zip(source, target):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.extend((u, v))
    solution, *_ = np.linalg.lstsq(np.asarray(matrix), np.asarray(values), rcond=None)
    return np.array(
        [
            [solution[0], solution[1], solution[2]],
            [solution[3], solution[4], solution[5]],
            [solution[6], solution[7], 1.0],
        ]
    )


def apply_homography(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = matrix @ np.array([x, y, 1.0])
    if abs(point[2]) < 1e-8:
        return x, y
    return float(point[0] / point[2]), float(point[1] / point[2])
