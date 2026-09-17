"""
Trajectory analysis.

Turns a raw list of (x, y, timestamp_s) points into higher-level signals:
compass-style direction of travel, and total path length in pixels (used
for basic sanity checks / analytics, not for speed - see app/speed).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

Point = Tuple[float, float, float]

# 8-way compass directions in image coordinates (y grows downward).
_DIRECTIONS = [
    ("East", 0),
    ("South-East", 45),
    ("South", 90),
    ("South-West", 135),
    ("West", 180),
    ("North-West", -135),
    ("North", -90),
    ("North-East", -45),
]


def estimate_direction(
    trajectory: List[Point], min_displacement_px: float = 8.0
) -> Optional[str]:
    """
    Classify overall direction of travel using the start and end points of
    the visible trajectory window (robust to per-frame jitter, unlike using
    only the last two points).
    """
    if len(trajectory) < 2:
        return None

    x1, y1, _ = trajectory[0]
    x2, y2, _ = trajectory[-1]
    dx, dy = x2 - x1, y2 - y1

    if math.hypot(dx, dy) < min_displacement_px:
        return "Stationary"

    angle_deg = math.degrees(math.atan2(dy, dx))
    best_label, best_diff = None, 361.0
    for label, ref_angle in _DIRECTIONS:
        diff = abs(_angle_diff(angle_deg, ref_angle))
        if diff < best_diff:
            best_diff = diff
            best_label = label
    return best_label


def _angle_diff(a: float, b: float) -> float:
    d = (a - b + 180) % 360 - 180
    return d


def trajectory_length_px(trajectory: List[Point]) -> float:
    """Sum of pixel distances between consecutive trajectory points."""
    total = 0.0
    for (x1, y1, _), (x2, y2, _) in zip(trajectory, trajectory[1:]):
        total += math.hypot(x2 - x1, y2 - y1)
    return total
