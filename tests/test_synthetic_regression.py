"""Reproducible end-to-end check on a synthetic video (no YOLO weights).

A white box moves 4 px/frame at 25 fps, so its true speed is 100 px/s.
With 10 px = 1 m that is 36.0 km/h. The video is written to a temporary
directory with OpenCV and decoded with OpenCV, as the app does.
"""
import cv2
import numpy as np
import pytest

from app.config import AppConfig
from app.pipeline import FrameProcessor
from tests.synthetic import ThresholdDetector, speed_samples

FPS = 25.0
W, H = 320, 240


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "box.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert writer.isOpened(), "OpenCV mp4v encoder is required for this test"
    for i in range(100):
        frame = np.zeros((H, W, 3), np.uint8)
        x = 20 + 4 * i
        cv2.rectangle(frame, (x, 100), (x + 40, 130), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def run(video_path):
    cfg = AppConfig()
    cfg.speed.pixels_per_meter = 10.0
    proc = FrameProcessor(cfg, ThresholdDetector())
    cap = cv2.VideoCapture(str(video_path))
    results = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results.append(proc.process_frame(frame, i, i / FPS))
        i += 1
    cap.release()
    return proc, results


def test_one_stable_track_identity(run):
    proc, _ = run
    assert len(proc.tracker.all_vehicles()) == 1


def test_speed_matches_ground_truth(run):
    _, results = run
    samples = [s for _, _, s, _ in speed_samples(results)]
    assert len(samples) >= 40
    median = sorted(samples)[len(samples) // 2]
    assert abs(median - 36.0) < 0.5
    assert all(abs(s - 36.0) < 0.5 for s in samples), "a clipped or duplicated sample leaked in"


def test_no_duplicate_trajectory_points(run):
    proc, _ = run
    vehicle = proc.tracker.all_vehicles()[0]
    centres = [(round(x, 2), round(y, 2)) for x, y, _ in vehicle.trajectory]
    assert len(centres) == len(set(centres))


def test_trajectory_timestamps_follow_inference_cadence(run):
    proc, _ = run
    vehicle = proc.tracker.all_vehicles()[0]
    times = [t for _, _, t in vehicle.trajectory]
    steps = {round(b - a, 4) for a, b in zip(times, times[1:])}
    assert steps == {round(2 / FPS, 4)}  # inference_interval 2 at 25 fps
    assert vehicle.frames_seen == len(times)
