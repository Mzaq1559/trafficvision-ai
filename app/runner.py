"""
Process one uploaded video and return a plain result dictionary.

No Streamlit imports: the Streamlit entry point supplies the detector factory
and an optional per-frame callback for progress and preview. Errors are
returned in the result, never raised, so the UI can show them.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np

from app.analytics.analytics import TrafficAnalytics
from app.config import AppConfig, DEFAULT_CONFIG_PATH, read_config
from app.pipeline import FrameProcessor
from app.ui_state import is_explicit_config
from app.utils.drawing import draw_detections, draw_trajectory

DetectorFactory = Callable[[AppConfig], Any]
FrameCallback = Callable[[int, int, np.ndarray], None]


def process_video(
    video_bytes: bytes,
    filename: str,
    config_path: str,
    max_frames: int,
    upload_key: str,
    make_detector: DetectorFactory,
    on_frame: Optional[FrameCallback] = None,
) -> dict[str, Any]:
    path = (config_path or "").strip() or DEFAULT_CONFIG_PATH
    loaded = read_config(path, explicit=is_explicit_config(path))
    if loaded.error is not None or loaded.config is None:
        return {"upload_key": upload_key, "error": loaded.error or "configuration could not be loaded",
                "warnings": loaded.warnings}
    config = loaded.config

    video_path = None
    cap = None
    try:
        suffix = os.path.splitext(filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(video_bytes)
            video_path = handle.name
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not decode this file. Try a video using a supported codec.")

        processor = FrameProcessor(config, make_detector(config))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit = min(max_frames, total) if total > 0 else max_frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0

        processed = 0
        latest_vehicles: list = []
        last_rgb = None
        started = time.perf_counter()
        while processed < limit:
            ok, frame = cap.read()
            if not ok:
                break
            result = processor.process_frame(frame, processed, processed / fps)
            latest_vehicles = result.vehicles
            annotated = frame.copy()
            if config.ui.show_labels:
                annotated = draw_detections(annotated, latest_vehicles, show_labels=True)
            if config.ui.show_trajectories:
                annotated = draw_trajectory(annotated, latest_vehicles)
            last_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            processed += 1
            if on_frame is not None:
                on_frame(processed, limit, last_rgb)
        elapsed = max(time.perf_counter() - started, 1e-9)

        snap = TrafficAnalytics().snapshot(
            latest_vehicles,
            current_video_time_s=processed / fps,
            wall_clock_now=time.perf_counter(),
        )
        return {
            "upload_key": upload_key,
            "error": None,
            "warnings": loaded.warnings,
            "config_source": str(loaded.source) if loaded.source else "built-in defaults",
            "frames": processed,
            "fps_observed": processed / elapsed,
            "annotated_rgb": last_rgb,
            "summary": {
                "total_vehicles": snap.total_vehicles,
                "active_vehicles": snap.active_vehicles,
                "average_speed": snap.average_speed,
                "speed_unit": snap.speed_unit,
                "class_distribution": snap.class_distribution,
                "direction_distribution": snap.direction_distribution,
            },
            "table": [
                {"track_id": v.track_id, "class": v.class_name, "direction": v.direction,
                 "speed": v.speed_kmh,
                 "speed_unit": "km/h" if v.speed_is_calibrated else "px/s (approx.)",
                 "tracker_updates": v.frames_seen}
                for v in latest_vehicles if v.status == "active"
            ],
        }
    except Exception as exc:
        return {"upload_key": upload_key, "error": f"Video processing failed: {exc}",
                "warnings": loaded.warnings}
    finally:
        if cap is not None:
            cap.release()
        if video_path:
            try:
                os.remove(video_path)
            except OSError:
                pass
