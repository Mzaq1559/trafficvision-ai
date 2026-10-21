"""Pipeline/tracker tests using a scripted fake detector (no YOLO weights, no torch)."""
from collections import deque

import numpy as np
import pytest

pytest.importorskip("supervision")

from app.analytics.analytics import TrafficAnalytics
from app.config import AppConfig
from app.detection.detector import Detection
from app.pipeline import FrameProcessor
from app.tracking.tracker import TrackedVehicle
from app.utils.drawing import draw_detections, draw_trajectory


class FakeDetector:
    """Returns one scripted box list per call; records how often it was called."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def detect(self, frame):
        boxes = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return [Detection(xyxy=b, confidence=0.9, class_id=2, class_name="car") for b in boxes]


def _blank_frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_skipped_frames_do_not_call_detector_or_tracker():
    # inference_interval defaults to 2: frames 0, 2, 4 run inference.
    # The box moves 2 px per inference frame (IoU stays above the 0.8 match threshold).
    det = FakeDetector([[(10, 10, 50, 50)], [(12, 10, 52, 50)], [(14, 10, 54, 50)]])
    proc = FrameProcessor(AppConfig(), det)
    for i in range(6):
        proc.process_frame(_blank_frame(), i, i / 30)

    assert det.calls == 3
    vehicles = proc.tracker.all_vehicles()
    assert len(vehicles) == 1
    veh = vehicles[0]
    assert veh.frames_seen == 3  # not 6: skipped frames do not reach the tracker
    assert [round(t, 4) for _, _, t in veh.trajectory] == [0.0, round(2 / 30, 4), round(4 / 30, 4)]
    centres_x = [x for x, _, _ in veh.trajectory]
    assert len(set(centres_x)) == 3  # no duplicated centres from reused boxes


def test_speed_uses_inference_timestamps_and_is_uncalibrated_without_scale():
    det = FakeDetector([[(10, 10, 50, 50)], [(12, 10, 52, 50)]])
    proc = FrameProcessor(AppConfig(), det)
    result = None
    for i in range(3):
        result = proc.process_frame(_blank_frame(), i, i / 30)
    veh = result.vehicles[0]
    # 2 px displacement over 2 frames (2/30 s) = 30 px/s, no calibration -> px/s
    assert veh.speed_kmh == pytest.approx(30.0, abs=0.1)
    assert veh.speed_is_calibrated is False


def test_speed_disabled_by_config():
    cfg = AppConfig()
    cfg.speed.enabled = False
    det = FakeDetector([[(10, 10, 50, 50)], [(12, 10, 52, 50)]])
    proc = FrameProcessor(cfg, det)
    for i in range(3):
        result = proc.process_frame(_blank_frame(), i, i / 30)
    assert result.vehicles[0].speed_kmh is None


def test_lost_track_is_retained_but_not_drawn():
    det = FakeDetector([[(10, 10, 50, 50)], []])  # object disappears at the second inference
    proc = FrameProcessor(AppConfig(), det)
    first = proc.process_frame(_blank_frame(), 0, 0.0)
    canvas = _blank_frame()
    draw_detections(canvas, first.vehicles, show_labels=True)
    draw_trajectory(canvas, first.vehicles)
    assert canvas.any()  # active track is drawn

    second = proc.process_frame(_blank_frame(), 2, 2 / 30)
    assert second.vehicles[0].status == "lost"
    assert len(second.vehicles) == 1  # still retained in the run's track list

    canvas = _blank_frame()
    draw_detections(canvas, second.vehicles, show_labels=True)
    draw_trajectory(canvas, second.vehicles)
    assert not canvas.any()  # lost track produces no stale boxes or paths


def test_analytics_speed_ignores_lost_tracks():
    def make(track_id, status, speed):
        return TrackedVehicle(
            track_id=track_id, class_name="car", confidence=0.9, xyxy=(0, 0, 1, 1),
            trajectory=deque(), first_seen_s=0.0, last_seen_s=1.0,
            speed_kmh=speed, speed_is_calibrated=True, status=status,
        )

    snap = TrafficAnalytics().snapshot(
        [make(1, "active", 30.0), make(2, "lost", 90.0)],
        current_video_time_s=60.0,
        wall_clock_now=1.0,
    )
    assert snap.total_vehicles == 2
    assert snap.active_vehicles == 1
    assert snap.average_speed == pytest.approx(30.0)
