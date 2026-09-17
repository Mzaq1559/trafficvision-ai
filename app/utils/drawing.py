"""
Rendering helpers. Kept separate from the tracking/analytics logic so the
"draw the frame" concern never gets tangled with "update the dashboard"
concern - the two are updated at different intervals on purpose (see
streamlit_app.py: rendering happens every frame, analytics/UI widgets
refresh every `ui_update_interval` frames).
"""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

_PALETTE = [
    (56, 189, 248),   # sky blue
    (52, 211, 153),   # emerald
    (251, 191, 36),   # amber
    (244, 114, 182),  # pink
    (167, 139, 250),  # violet
    (248, 113, 113),  # red
]


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    vehicles: Iterable,
    show_labels: bool = True,
    selected_id: int | None = None,
) -> np.ndarray:
    """Draw bounding boxes + optional labels for a set of TrackedVehicle."""
    for veh in vehicles:
        x1, y1, x2, y2 = (int(v) for v in veh.xyxy)
        color = _color_for_id(veh.track_id)
        thickness = 3 if veh.track_id == selected_id else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if show_labels:
            speed_txt = ""
            if veh.speed_kmh is not None:
                unit = "km/h" if veh.speed_is_calibrated else "px/s~"
                speed_txt = f" {veh.speed_kmh:.0f}{unit}"
            label = f"#{veh.track_id} {veh.class_name}{speed_txt}"
            _draw_label(frame, label, (x1, max(0, y1 - 8)), color)
    return frame


def _draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw + 4, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 2, y - 2), font, scale, (17, 24, 39), thickness, cv2.LINE_AA)


def draw_trajectory(frame: np.ndarray, vehicles: Iterable) -> np.ndarray:
    """Draw the recent path of each tracked vehicle as a polyline."""
    for veh in vehicles:
        pts = [(int(x), int(y)) for x, y, _ in veh.trajectory]
        if len(pts) < 2:
            continue
        color = _color_for_id(veh.track_id)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
    return frame
