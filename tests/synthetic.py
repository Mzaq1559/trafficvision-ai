"""Synthetic scenes with known ground truth. No YOLO weights, no torch.

A white box on a black background is "detected" by thresholding. This is a
stand-in for the YOLO detector that exercises the pipeline's geometry, timing,
tracking and speed logic against values that are known exactly.
"""
from __future__ import annotations

import numpy as np
import cv2

from app.config import AppConfig
from app.detection.detector import Detection
from app.pipeline import FrameProcessor

W, H = 320, 240
BOX_W, BOX_H = 40, 30


def box_frame(x: float, y: float, shape=(H, W)) -> np.ndarray:
    """Black frame with a white box whose top-left corner is (x, y).
    Parts outside the frame are clipped by OpenCV, as a real camera would show."""
    frame = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    cv2.rectangle(frame, (int(round(x)), int(round(y))),
                  (int(round(x)) + BOX_W, int(round(y)) + BOX_H), (255, 255, 255), -1)
    return frame


class ThresholdDetector:
    """Returns the bounding box of all bright pixels as one 'car'."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame: np.ndarray):
        self.calls += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray > 200)
        if xs.size == 0:
            return []
        return [Detection(xyxy=(float(xs.min()), float(ys.min()),
                                float(xs.max() + 1), float(ys.max() + 1)),
                          confidence=0.9, class_id=2, class_name="car")]


def run_pipeline(frames, fps: float, pixels_per_meter=None, config: AppConfig | None = None):
    """Process frames in order. Returns (processor, list of ProcessedFrame)."""
    cfg = config or AppConfig()
    if config is None:
        cfg.speed.pixels_per_meter = pixels_per_meter
    proc = FrameProcessor(cfg, ThresholdDetector())
    results = [proc.process_frame(f, i, i / fps) for i, f in enumerate(frames)]
    return proc, results


def speed_samples(results):
    """(frame_index, track_id, speed_kmh, is_calibrated) for every active speed value."""
    out = []
    for r in results:
        for v in r.vehicles:
            if v.status == "active" and v.speed_kmh is not None:
                out.append((r.frame_index, v.track_id, v.speed_kmh, v.speed_is_calibrated))
    return out
