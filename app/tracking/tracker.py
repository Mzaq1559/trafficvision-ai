"""
Multi-object tracking with persistent vehicle IDs.

Uses the `supervision` library's ByteTrack implementation (a maintained,
CPU-friendly Python port of ByteTrack) for the actual association logic,
and layers persistent per-track state on top of it:

  - trajectory history (for drawing paths and for speed estimation)
  - time-in-scene (based on video timestamps, not wall-clock time)
  - a majority-vote class label (detections can flicker between classes)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from app.detection.detector import Detection


@dataclass
class TrackedVehicle:
    """Everything the dashboard needs to know about one persistent vehicle."""

    track_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]
    trajectory: Deque[Tuple[float, float, float]]  # (x, y, timestamp_s)
    first_seen_s: float
    last_seen_s: float
    frames_seen: int = 0
    speed_kmh: Optional[float] = None
    speed_is_calibrated: bool = False
    direction: Optional[str] = None
    status: str = "active"  # active | lost

    @property
    def duration_s(self) -> float:
        return max(0.0, self.last_seen_s - self.first_seen_s)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class VehicleTracker:
    """
    Wraps supervision.ByteTrack and maintains persistent TrackedVehicle state
    across frames so the UI can show a stable vehicle table.
    """

    def __init__(
        self,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        trajectory_length: int = 40,
    ):
        self.frame_rate = frame_rate
        self.lost_track_buffer = lost_track_buffer
        self.trajectory_length = trajectory_length

        import supervision as sv

        self._sv = sv
        self._tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

        self._vehicles: Dict[int, TrackedVehicle] = {}
        self._class_votes: Dict[int, Dict[str, int]] = {}

    def reset(self) -> None:
        """Fully reset tracker state - call this whenever a new processing
        run starts so stale IDs/trajectories never leak between videos."""
        self._tracker.reset()
        self._vehicles.clear()
        self._class_votes.clear()

    def update(
        self, detections: List[Detection], timestamp_s: float
    ) -> List[TrackedVehicle]:
        """Feed one frame's detections in, get back the current set of
        actively-tracked vehicles (with persistent IDs and trajectories)."""
        sv = self._sv

        # Map class_id -> class_name (not position -> name: tracked results
        # are re-ordered/filtered by ByteTrack, so index alignment with the
        # input detections list cannot be relied on).
        id_to_name = {d.class_id: d.class_name for d in detections}

        if detections:
            xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.array([d.class_id for d in detections], dtype=int)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            confidence = np.zeros((0,), dtype=np.float32)
            class_id = np.zeros((0,), dtype=int)

        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._tracker.update_with_detections(sv_detections)

        seen_ids = set()
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            box = tuple(float(v) for v in tracked.xyxy[i])
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            class_name = id_to_name.get(cls_id, "vehicle")

            self._register_class_vote(track_id, class_name)
            seen_ids.add(track_id)

            if track_id not in self._vehicles:
                self._vehicles[track_id] = TrackedVehicle(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=conf,
                    xyxy=box,
                    trajectory=deque(maxlen=self.trajectory_length),
                    first_seen_s=timestamp_s,
                    last_seen_s=timestamp_s,
                )

            veh = self._vehicles[track_id]
            veh.xyxy = box
            veh.confidence = conf
            veh.class_name = self._majority_class(track_id)
            veh.last_seen_s = timestamp_s
            veh.frames_seen += 1
            veh.status = "active"
            cx, cy = veh.center
            veh.trajectory.append((cx, cy, timestamp_s))

        # Anything not seen this frame is (still) lost; ByteTrack already
        # drops IDs past lost_track_buffer, so we mirror that by marking
        # status rather than deleting immediately (keeps the table stable).
        for track_id, veh in self._vehicles.items():
            if track_id not in seen_ids:
                veh.status = "lost"

        return list(self._vehicles.values())

    def active_vehicles(self) -> List[TrackedVehicle]:
        return [v for v in self._vehicles.values() if v.status == "active"]

    def get(self, track_id: int) -> Optional[TrackedVehicle]:
        return self._vehicles.get(track_id)

    def _register_class_vote(self, track_id: int, class_name: str) -> None:
        votes = self._class_votes.setdefault(track_id, {})
        votes[class_name] = votes.get(class_name, 0) + 1

    def _majority_class(self, track_id: int) -> str:
        votes = self._class_votes.get(track_id, {})
        if not votes:
            return "vehicle"
        return max(votes.items(), key=lambda kv: kv[1])[0]
