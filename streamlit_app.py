"""Streamlit entry point for local-video TrafficVision AI demos."""
from __future__ import annotations
import tempfile
from pathlib import Path
import cv2
import streamlit as st

from app.config import load_config
from app.detection.detector import VehicleDetector
from app.pipeline import FrameProcessor
from app.utils.drawing import draw_detections, draw_trajectory

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
    try:
        config = load_config(config_path)
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded.getbuffer())
            video_path = handle.name
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not decode this file. Try an MP4 with a supported codec.")
        detector = VehicleDetector(config.detection)
        processor = FrameProcessor(config, detector)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit = min(max_frames, total) if total > 0 else max_frames
        progress = st.progress(0, text="Processing video...")
        preview = st.empty()
        count_metric = st.empty()
        processed = 0
        latest_vehicles = []
        while processed < limit:
            ok, frame = cap.read()
            if not ok:
                break
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 30.0
            result = processor.process_frame(frame, processed, processed / fps)
            latest_vehicles = result.vehicles
            annotated = frame.copy()
            if config.ui.show_labels:
                annotated = draw_detections(annotated, result.vehicles)
            if config.ui.show_trajectories:
                for vehicle in result.vehicles:
                    annotated = draw_trajectory(annotated, vehicle)
            preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            processed += 1
            progress.progress(min(1.0, processed / max(1, limit)), text=f"Processed {processed}/{limit} frames")
        cap.release()
        count_metric.metric("Frames processed", processed)
        st.caption("Track counts and speed values are model-derived estimates. Validate against labelled data before drawing research conclusions.")
        if latest_vehicles:
            st.dataframe([{"track_id": v.track_id, "class": getattr(v, "class_name", getattr(v, "class_id", "")),
                           "direction": getattr(v, "direction", None),
                           "speed": getattr(v, "speed_kmh", None),
                           "speed_calibrated": getattr(v, "speed_is_calibrated", False),
                           "status": getattr(v, "status", None)} for v in latest_vehicles], use_container_width=True)
        try:
            Path(video_path).unlink(missing_ok=True)
        except OSError:
            pass
    except Exception as exc:
        st.error(f"Video processing failed: {exc}")
        st.exception(exc)
