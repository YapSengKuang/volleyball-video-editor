"""Court polygon and the person/air zones above it."""

from __future__ import annotations

import numpy as np


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Even-odd test. Polygon is an (N, 2) array of pixel coordinates."""
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if (y1 > y) == (y2 > y):
            continue
        cross = (x2 - x1) * (y - y1) / (y2 - y1 + 0.0) + x1
        if x < cross:
            inside = not inside
    return inside


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. Returns points in boundary order."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.vstack(lower[:-1] + upper[:-1])
    return hull


def zone_above(polygon: np.ndarray, rise_px: float) -> np.ndarray:
    """Convex hull of the floor polygon and a copy shifted up the image."""
    poly = np.asarray(polygon, dtype=np.float64)
    if rise_px <= 0:
        return convex_hull(poly)
    lifted = poly.copy()
    lifted[:, 1] -= rise_px
    return convex_hull(np.vstack([poly, lifted]))


def full_frame_polygon(width: int, height: int) -> np.ndarray:
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )


def scale_polygon(polygon: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = np.asarray(polygon, dtype=np.float64).copy()
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy
    return scaled


def raster_mask(height: int, width: int, polygon: np.ndarray) -> np.ndarray:
    """Boolean mask, True inside the polygon."""
    poly = np.asarray(polygon, dtype=np.float64)
    yy, xx = np.mgrid[0:height, 0:width]
    mask = np.zeros((height, width), dtype=bool)
    count = len(poly)
    for index in range(count):
        x1, y1 = poly[index]
        x2, y2 = poly[(index + 1) % count]
        cond = (y1 > yy) != (y2 > yy)
        cross = (x2 - x1) * (yy - y1) / (y2 - y1 + 1e-12) + x1
        mask ^= cond & (xx < cross)
    return mask
