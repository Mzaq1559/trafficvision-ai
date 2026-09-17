"""
Vehicle detection using Ultralytics YOLO.

This module is intentionally framework-agnostic (no Streamlit imports) so it
can be unit tested and reused outside the dashboard. Model loading is the
expensive part; callers (the Streamlit app) are expected to cache the
VehicleDetector instance themselves (st.cache_resource).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# COCO class ids we care about for road traffic.
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    """A single raw detection for one frame, before tracking is applied."""

    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


class VehicleDetector:
    """Thin, testable wrapper around an Ultralytics YOLO model."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: Optional[Sequence[int]] = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = list(classes) if classes else list(COCO_VEHICLE_CLASSES.keys())
        self.device = device
        self._model = None  # lazy-loaded so importing this module is cheap

    def load(self) -> "VehicleDetector":
        """Load (and, on first use, download) the YOLO weights."""
        if self._model is None:
            from ultralytics import YOLO  # local import keeps module import light

            self._model = YOLO(self.model_name)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame (as returned by OpenCV)."""
        if self._model is None:
            self.load()

        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        return parse_yolo_result(results[0])


def parse_yolo_result(result) -> List[Detection]:
    """
    Convert one Ultralytics `Results` object into a list of Detection.

    Split out as a standalone function so detection parsing can be unit
    tested against a lightweight fake object instead of a real model.
    """
    detections: List[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    names = getattr(result, "names", COCO_VEHICLE_CLASSES)

    xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
    cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

    for xyxy, conf, cls in zip(xyxy_arr, conf_arr, cls_arr):
        class_id = int(cls)
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        detections.append(
            Detection(
                xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                confidence=float(conf),
                class_id=class_id,
                class_name=class_name,
            )
        )
    return detections
