"""
Speed estimation.

Two modes, and the UI/README must never blur the distinction between them:

  * CALIBRATED  - a real-world pixels-per-meter ratio is known (derived from
    two reference points a known real distance apart). Output is a genuine
    km/h estimate, still subject to camera-angle/perspective error.

  * APPROXIMATE - no calibration was provided. We fall back to a rough
    "pixels moved per second", which is NOT a real-world speed and must be
    labelled as approximate (arbitrary units) wherever it is shown.

Speed is computed purely from trajectory points and their video timestamps
(never wall-clock time), so results are identical whether processing runs
faster or slower than real-time.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple


@dataclass
class SpeedResult:
    value: float
    unit: str  # "km/h" or "px/s"
    is_calibrated: bool


def calibrate_pixels_per_meter(
    point_a: Tuple[float, float], point_b: Tuple[float, float], real_distance_m: float
) -> float:
    """Given two pixel points that correspond to a known real-world
    distance (e.g. lane-marking spacing), return pixels-per-meter."""
    if real_distance_m <= 0:
        raise ValueError("real_distance_m must be > 0")
    pixel_dist = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
    if pixel_dist == 0:
        raise ValueError("Calibration points must not be identical")
    return pixel_dist / real_distance_m


class SpeedEstimator:
    """Stateless-per-call speed estimator operating on a trajectory deque of
    (x, y, timestamp_s) tuples, as produced by VehicleTracker."""

    def __init__(
        self,
        pixels_per_meter: Optional[float] = None,
        smoothing_window: int = 5,
        max_plausible_kmh: float = 220.0,
    ):
        self.pixels_per_meter = pixels_per_meter
        self.smoothing_window = max(2, smoothing_window)
        self.max_plausible_kmh = max_plausible_kmh
        # per-track rolling history of instantaneous speed samples, for smoothing
        self._history: dict[int, Deque[float]] = {}

    @property
    def is_calibrated(self) -> bool:
        return self.pixels_per_meter is not None and self.pixels_per_meter > 0

    def estimate(
        self, track_id: int, trajectory: List[Tuple[float, float, float]]
    ) -> Optional[SpeedResult]:
        """Estimate current speed for a track from its recent trajectory."""
        if len(trajectory) < 2:
            return None

        (x1, y1, t1), (x2, y2, t2) = trajectory[-2], trajectory[-1]
        dt = t2 - t1
        if dt <= 0:
            return None

        pixel_dist = math.hypot(x2 - x1, y2 - y1)
        pixels_per_s = pixel_dist / dt

        if self.is_calibrated:
            meters_per_s = pixels_per_s / self.pixels_per_meter
            raw_kmh = meters_per_s * 3.6
            smoothed = self._smooth(track_id, raw_kmh)
            # Guard against absurd spikes from a single bad detection/track swap.
            smoothed = min(smoothed, self.max_plausible_kmh)
            return SpeedResult(value=round(smoothed, 1), unit="km/h", is_calibrated=True)

        smoothed = self._smooth(track_id, pixels_per_s)
        return SpeedResult(value=round(smoothed, 1), unit="px/s", is_calibrated=False)

    def _smooth(self, track_id: int, sample: float) -> float:
        hist = self._history.setdefault(track_id, deque(maxlen=self.smoothing_window))
        hist.append(sample)
        return sum(hist) / len(hist)

    def reset(self) -> None:
        self._history.clear()
