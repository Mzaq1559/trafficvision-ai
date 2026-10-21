"""
Aggregate per-vehicle state into dashboard-ready traffic analytics.

This module only reads TrackedVehicle objects; it never mutates tracker
state. Track totals count every track retained in the run (including lost
tracks); speed statistics use active tracks only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.tracking.tracker import TrackedVehicle


@dataclass
class AnalyticsSnapshot:
    total_vehicles: int
    active_vehicles: int
    average_speed: Optional[float]
    speed_unit: str
    class_distribution: Dict[str, int]
    direction_distribution: Dict[str, int]
    flow_per_minute: float
    processing_fps: float


class TrafficAnalytics:
    """Aggregates one run's vehicle list. The only state it keeps is the
    run start time (`_start_time_s`) and processing-rate bookkeeping, so use
    a new instance per run. It never modifies tracker state."""

    def __init__(self) -> None:
        self._start_time_s: Optional[float] = None
        self._processed_frames = 0
        self._processing_wall_start: Optional[float] = None

    def note_frame_processed(self, wall_clock_now: float) -> None:
        if self._processing_wall_start is None:
            self._processing_wall_start = wall_clock_now
        self._processed_frames += 1

    def processing_fps(self, wall_clock_now: float) -> float:
        if self._processing_wall_start is None or self._processed_frames == 0:
            return 0.0
        elapsed = max(1e-6, wall_clock_now - self._processing_wall_start)
        return self._processed_frames / elapsed

    def reset(self) -> None:
        self._start_time_s = None
        self._processed_frames = 0
        self._processing_wall_start = None

    def snapshot(
        self,
        all_vehicles: List[TrackedVehicle],
        current_video_time_s: float,
        wall_clock_now: float,
    ) -> AnalyticsSnapshot:
        active = [v for v in all_vehicles if v.status == "active"]

        class_dist = Counter(v.class_name for v in all_vehicles)
        direction_dist = Counter(v.direction for v in all_vehicles if v.direction)

        # Speed statistics use active tracks only: lost tracks carry the speed
        # from their last observation, which is stale.
        speeds = [v.speed_kmh for v in active if v.speed_kmh is not None]
        calibrated = any(v.speed_is_calibrated for v in active if v.speed_kmh is not None)
        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else None

        if self._start_time_s is None and all_vehicles:
            self._start_time_s = min(v.first_seen_s for v in all_vehicles)

        elapsed_minutes = 0.0
        if self._start_time_s is not None:
            elapsed_minutes = max(1e-6, (current_video_time_s - self._start_time_s) / 60.0)
        flow_per_minute = round(len(all_vehicles) / elapsed_minutes, 1) if elapsed_minutes else 0.0

        return AnalyticsSnapshot(
            total_vehicles=len(all_vehicles),
            active_vehicles=len(active),
            average_speed=avg_speed,
            speed_unit="km/h" if calibrated else "px/s (approx.)",
            class_distribution=dict(class_dist),
            direction_distribution=dict(direction_dist),
            flow_per_minute=flow_per_minute,
            processing_fps=round(self.processing_fps(wall_clock_now), 1),
        )
