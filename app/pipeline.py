"""
FrameProcessor: the single place that wires detection -> tracking ->
trajectory/direction -> speed together for one frame. Streamlit only calls
`process_frame` in a loop; all CV logic lives here so it stays testable
without a running Streamlit session.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from app.config.settings import AppConfig
from app.detection.detector import VehicleDetector
from app.tracking.tracker import VehicleTracker, TrackedVehicle
from app.speed.estimator import SpeedEstimator
from app.utils.trajectory import estimate_direction


@dataclass
class ProcessedFrame:
    frame: np.ndarray
    vehicles: List[TrackedVehicle]
    frame_index: int
    timestamp_s: float
    ran_inference: bool


class FrameProcessor:
    """Stateful per-run processor. Call `reset()` before a new video/run."""

    def __init__(self, config: AppConfig, detector: VehicleDetector):
        self.config = config
        self.detector = detector
        self.tracker = VehicleTracker(
            frame_rate=config.tracking.frame_rate,
            track_activation_threshold=config.tracking.track_activation_threshold,
            lost_track_buffer=config.tracking.lost_track_buffer,
            minimum_matching_threshold=config.tracking.minimum_matching_threshold,
            minimum_consecutive_frames=config.tracking.minimum_consecutive_frames,
            trajectory_length=config.tracking.trajectory_length,
        )
        self.speed_estimator = SpeedEstimator(
            pixels_per_meter=config.speed.pixels_per_meter,
            smoothing_window=config.speed.smoothing_window,
            max_plausible_kmh=config.speed.max_plausible_kmh,
        )
        self._last_detections = []
        self._frames_since_inference = 0

    def reset(self) -> None:
        """Completely reset processing state - required whenever a new
        processing run starts, so IDs/trajectories/speeds never leak
        between separate videos or repeated runs of the same video."""
        self.tracker.reset()
        self.speed_estimator.reset()
        self._last_detections = []
        self._frames_since_inference = 0

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp_s: float) -> ProcessedFrame:
        interval = max(1, self.config.detection.inference_interval)
        ran_inference = (frame_index % interval == 0) or not self._last_detections

        if ran_inference:
            proc_frame = _resize_for_inference(frame, self.config.detection.processing_width)
            scale_x = frame.shape[1] / proc_frame.shape[1]
            scale_y = frame.shape[0] / proc_frame.shape[0]
            detections = self.detector.detect(proc_frame)
            for d in detections:
                x1, y1, x2, y2 = d.xyxy
                d.xyxy = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            self._last_detections = detections
        else:
            detections = self._last_detections

        vehicles = self.tracker.update(detections, timestamp_s)

        for veh in vehicles:
            if veh.status != "active":
                continue
            veh.direction = estimate_direction(list(veh.trajectory))
            speed = self.speed_estimator.estimate(veh.track_id, list(veh.trajectory))
            if speed is not None:
                veh.speed_kmh = speed.value
                veh.speed_is_calibrated = speed.is_calibrated

        return ProcessedFrame(
            frame=frame,
            vehicles=vehicles,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            ran_inference=ran_inference,
        )


def _resize_for_inference(frame: np.ndarray, target_width: int) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    new_size = (target_width, max(1, int(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
