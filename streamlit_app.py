"""Streamlit entry point for local-video TrafficVision AI demos."""
from __future__ import annotations
import tempfile
from pathlib import Path
import time
import cv2
import streamlit as st

from app.analytics.analytics import TrafficAnalytics
from app.config import load_config
from app.detection.detector import VehicleDetector
from app.pipeline import FrameProcessor
from app.utils.drawing import draw_detections, draw_trajectory


@st.cache_resource(show_spinner="Loading detector weights...")
def _get_detector(model_name: str, confidence: float, iou: float, classes: tuple, device: str) -> VehicleDetector:
    """Cache the YOLO model across Streamlit reruns; the detector holds no per-run state."""
    return VehicleDetector(
        model_name=model_name,
        confidence_threshold=confidence,
        iou_threshold=iou,
        classes=list(classes),
        device=device,
    )


st.set_page_config(page_title="TrafficVision AI", page_icon="🚦", layout="wide")
st.title("TrafficVision AI")
st.caption("Vehicle detection and tracking prototype — results are not ground-truth validated.")
uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
max_frames = st.sidebar.slider("Maximum frames to process", min_value=30, max_value=600, value=180, step=30)
config_path = st.sidebar.text_input("Configuration file", "configs/default.yaml")
if uploaded is None:
    st.info("Upload a local video to begin. YOLO weights may be downloaded on first inference.")
    st.stop()

if st.button("Process video", type="primary"):
    video_path = None
    cap = None
    try:
        config = load_config(config_path)
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded.getbuffer())
            video_path = handle.name
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not decode this file. Try a video using a supported codec.")
        detector = _get_detector(
            config.detection.model_name,
            config.detection.confidence_threshold,
            config.detection.iou_threshold,
            tuple(config.detection.classes),
            config.detection.device,
        )
        processor = FrameProcessor(config, detector)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit = min(max_frames, total) if total > 0 else max_frames
        progress = st.progress(0, text="Processing video...")
        preview = st.empty()
        processed = 0
        latest_vehicles = []
        started = time.perf_counter()
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
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
            preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            processed += 1
            progress.progress(min(1.0, processed / max(1, limit)), text=f"Processed {processed}/{limit} frames")
        elapsed = max(time.perf_counter() - started, 1e-9)
        st.metric("Frames processed", processed)
        st.metric("Observed processing throughput (including preview updates)", f"{processed / elapsed:.2f} FPS")
        st.caption("Throughput depends on hardware, model, input and UI overhead. Track counts and speed values are estimates.")

        snap = TrafficAnalytics().snapshot(
            latest_vehicles,
            current_video_time_s=processed / fps,
            wall_clock_now=time.perf_counter(),
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Tracks retained (incl. lost)", snap.total_vehicles)
        col2.metric("Active at last processed frame", snap.active_vehicles)
        avg = "n/a" if snap.average_speed is None else f"{snap.average_speed} {snap.speed_unit}"
        col3.metric("Mean speed of active tracks", avg)
        st.caption(f"Class counts: {snap.class_distribution or '—'} · Direction counts: {snap.direction_distribution or '—'}")

        active = [v for v in latest_vehicles if v.status == "active"]
        if active:
            st.dataframe([{"track_id": v.track_id, "class": v.class_name,
                           "direction": v.direction,
                           "speed": v.speed_kmh,
                           "speed_unit": "km/h" if v.speed_is_calibrated else "px/s (approx.)",
                           "frames_seen": v.frames_seen}
                          for v in active], use_container_width=True)
    except Exception as exc:
        st.error(f"Video processing failed: {exc}")
        st.exception(exc)
    finally:
        if cap is not None:
            cap.release()
        if video_path:
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                pass
