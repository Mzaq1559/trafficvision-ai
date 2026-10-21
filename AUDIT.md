# Audit and development notes

Status of the items in the original audit, checked against the code on branch `docs/verify-and-license`.

## Resolved in this branch
- **Duplicate tracker updates on reused detections.** Skipped frames no longer call the detector or tracker (`app/pipeline.py`). Covered by `tests/test_pipeline_tracking.py`.
- **Lost tracks drawn as stale boxes and trajectories.** Drawing functions skip non-active tracks unless `include_lost=True` (`app/utils/drawing.py`). Covered by a test.
- **Speed ignored `speed.enabled`.** Now honoured by `FrameProcessor`. Covered by a test.
- **Speed statistics included lost tracks.** Analytics now averages active tracks only (`app/analytics/analytics.py`). Covered by a test.
- **Traffic summary not shown in the UI.** The Streamlit app now shows the summary from `TrafficAnalytics`.
- **Detector recreated on every run.** The Streamlit app caches it with `st.cache_resource`.
- **Failing test: escaped newlines in YAML.** Fixed in `tests/test_trajectory_speed.py`.
- **`supervision` lower bound too low.** `requirements.txt` now requires `>=0.23,<0.24`. On 0.20.0 the tracker constructor rejects `minimum_consecutive_frames`.
- **Config keys documented as active but unused.** Unused keys are marked RESERVED in `configs/default.yaml` and `app/config/settings.py`.
- **Non-deterministic download cache name.** `URLVideoSource` uses a SHA-256 digest instead of `hash()`.
- **Lint and type errors.** `ruff check` and `mypy app` report no issues.

## Remaining (not fixed)
- `processing_mode` presets are defined but not applied. Reserved.
- `video/source.py` (webcam and URL input) is not used by the UI and has no tests.
- Lost tracks stay in memory for the whole run. Long videos have not been tested.
- `tracking.frame_rate` is a constant passed to ByteTrack, not derived from video FPS.
- Speed has no perspective correction; calibration is a single global scale.
- Tracks are not counted across a line; the summary counts retained tracks.
- The Streamlit app has not been run on a real video with YOLO weights in this environment.

## Could not be validated here
- Installing the full requirements (PyTorch and Ultralytics were not installed; the test suite does not import them).
- Detection and tracking quality on real traffic video. No labelled data is included.
- Model weights: `yolov8n.pt` is downloaded by Ultralytics at first use and is not committed.
