"""Streamlit entry point for local-video TrafficVision AI demos.

Run results are kept in session state. They survive widget interactions and
reruns, are replaced by each new processing run, and are discarded when the
uploaded file changes or is removed.
"""
from __future__ import annotations
from typing import Any

import streamlit as st

from app.config import DEFAULT_CONFIG_PATH, read_config
from app.detection.detector import VehicleDetector
from app.runner import process_video
from app.ui_state import RESULTS_KEY, is_explicit_config, is_stale, upload_key


@st.cache_resource(show_spinner="Loading detector weights...")
def _get_detector(model_name: str, confidence: float, iou: float, classes: tuple, device: str) -> VehicleDetector:
    """Cache the YOLO model across reruns. The detector holds no per-run tracking state.
    Concurrent predict calls from several browser sessions are not verified to be safe."""
    return VehicleDetector(
        model_name=model_name,
        confidence_threshold=confidence,
        iou_threshold=iou,
        classes=list(classes),
        device=device,
    )


def _run(uploaded, config_path: str, max_frames: int, key: str) -> dict[str, Any]:
    """Process one upload, driving the live progress and preview widgets."""
    progress = st.progress(0, text="Processing video...")
    preview = st.empty()

    def on_frame(done: int, limit: int, rgb) -> None:
        preview.image(rgb, channels="RGB", use_container_width=True)
        progress.progress(min(1.0, done / max(1, limit)), text=f"Processed {done}/{limit} frames")

    def make_detector(config):
        return _get_detector(
            config.detection.model_name,
            config.detection.confidence_threshold,
            config.detection.iou_threshold,
            tuple(config.detection.classes),
            config.detection.device,
        )

    return process_video(uploaded.getvalue(), uploaded.name, config_path, max_frames, key,
                         make_detector=make_detector, on_frame=on_frame)


def _render(stored: dict[str, Any]) -> None:
    for message in stored.get("warnings", []):
        st.warning(message)
    if stored.get("error"):
        st.error(stored["error"])
        return
    st.caption(f"Configuration in use: {stored['config_source']}")
    st.metric("Frames processed", stored["frames"])
    st.metric("Observed processing throughput (including preview updates)", f"{stored['fps_observed']:.2f} FPS")
    st.caption("Throughput depends on hardware, model, input and UI overhead. Track counts and speed values are estimates.")
    if stored["annotated_rgb"] is not None:
        st.image(stored["annotated_rgb"], channels="RGB", caption="Last processed frame", use_container_width=True)
    s = stored["summary"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Tracks retained (incl. lost)", s["total_vehicles"])
    col2.metric("Active at last inference frame", s["active_vehicles"])
    avg = "n/a" if s["average_speed"] is None else f"{s['average_speed']} {s['speed_unit']}"
    col3.metric("Mean speed of active tracks", avg)
    st.caption(f"Class counts (all retained tracks): {s['class_distribution'] or '—'} · "
               f"Direction counts (all retained tracks): {s['direction_distribution'] or '—'}")
    if stored["table"]:
        st.dataframe(stored["table"], use_container_width=True)


st.set_page_config(page_title="TrafficVision AI", page_icon="🚦", layout="wide")
st.title("TrafficVision AI")
st.caption("Vehicle detection and tracking prototype — results are not ground-truth validated.")
uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
max_frames = st.sidebar.slider("Maximum frames to process", min_value=30, max_value=600, value=180, step=30)
config_path = st.sidebar.text_input("Configuration file", DEFAULT_CONFIG_PATH)

# Show the state of the selected configuration before any processing.
_path = config_path.strip() or DEFAULT_CONFIG_PATH
_status = read_config(_path, explicit=is_explicit_config(_path))
if _status.error is not None:
    st.sidebar.error(_status.error)
else:
    for _message in _status.warnings:
        st.sidebar.warning(_message)
    st.sidebar.caption(f"Configuration loaded: {_status.source or 'built-in defaults'}")

current_key = upload_key(uploaded.name, uploaded.size) if uploaded is not None else None
if is_stale(st.session_state.get(RESULTS_KEY), current_key):
    st.session_state.pop(RESULTS_KEY, None)

if uploaded is None:
    st.info("Upload a local video to begin. YOLO weights may be downloaded on first inference.")
    st.stop()

if st.button("Process video", type="primary"):
    st.session_state.pop(RESULTS_KEY, None)
    st.session_state[RESULTS_KEY] = _run(uploaded, config_path, max_frames, current_key)
    st.rerun()

stored = st.session_state.get(RESULTS_KEY)
if stored is not None:
    _render(stored)
