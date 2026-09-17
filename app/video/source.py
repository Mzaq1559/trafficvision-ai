"""
Video input handling.

Supports three input modes:
  1. Recorded video upload (a local file path, e.g. from Streamlit's file_uploader)
  2. Camera / webcam input (device index) - only works where OpenCV can see a camera,
     which is NOT the case on Streamlit Community Cloud's hosted containers.
  3. Public demo video URLs (downloaded once, then read as a normal file)

All sources expose the same minimal interface so the processing pipeline
never needs to know which one it is dealing with.
"""
from __future__ import annotations

import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameResult:
    ok: bool
    frame: Optional[np.ndarray]
    frame_index: int
    timestamp_s: float  # video-relative timestamp, NOT wall-clock time


class VideoSource(ABC):
    """Common interface for every input mode."""

    def __init__(self) -> None:
        self._frame_index = -1

    @abstractmethod
    def read(self) -> FrameResult:
        ...

    @abstractmethod
    def fps(self) -> float:
        ...

    @abstractmethod
    def total_frames(self) -> Optional[int]:
        """None when unknown (e.g. a live camera)."""
        ...

    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    def is_live(self) -> bool:
        """True for sources with no fixed length (webcam)."""
        return self.total_frames() is None


class _CVBackedSource(VideoSource):
    """Shared implementation for anything OpenCV's VideoCapture can open."""

    def __init__(self, cap: cv2.VideoCapture, live: bool = False):
        super().__init__()
        self._cap = cap
        self._live = live
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self._fps <= 0 or self._fps != self._fps:  # NaN guard
            self._fps = 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._total_frames = int(frame_count) if frame_count and frame_count > 0 else None
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    def read(self) -> FrameResult:
        ok, frame = self._cap.read()
        self._frame_index += 1
        if not ok:
            return FrameResult(False, None, self._frame_index, 0.0)

        if self._live:
            timestamp_s = time.monotonic()
        else:
            timestamp_s = self._frame_index / self._fps
        return FrameResult(True, frame, self._frame_index, timestamp_s)

    def fps(self) -> float:
        return self._fps

    def total_frames(self) -> Optional[int]:
        return None if self._live else self._total_frames

    def frame_size(self) -> Tuple[int, int]:
        return self._width, self._height

    def release(self) -> None:
        self._cap.release()


class FileVideoSource(_CVBackedSource):
    """A locally-available video file (e.g. an uploaded file saved to disk)."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"OpenCV could not open video file: {path}")
        super().__init__(cap, live=False)
        self.path = path


class WebcamVideoSource(_CVBackedSource):
    """
    Live camera input.

    Only works when a camera device is actually present and accessible to the
    process (i.e. running locally). On Streamlit Community Cloud there is no
    camera device, so this will fail to open - the UI should catch that and
    fall back to file/URL/demo mode.
    """

    def __init__(self, device_index: int = 0):
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise IOError(
                f"Could not open camera device {device_index}. "
                "Webcam input is unavailable in hosted/cloud environments."
            )
        super().__init__(cap, live=True)
        self.device_index = device_index


class URLVideoSource(FileVideoSource):
    """
    Downloads a public video URL to a temp file, then behaves like a
    FileVideoSource. Downloading (rather than streaming) keeps frame-count
    and seeking well-defined and avoids depending on network stability
    during processing.
    """

    def __init__(self, url: str, timeout_s: int = 60):
        import requests  # local import: only needed for this mode

        tmp_dir = tempfile.gettempdir()
        local_path = os.path.join(tmp_dir, f"trafficvision_dl_{abs(hash(url))}.mp4")

        if not os.path.exists(local_path):
            resp = requests.get(url, stream=True, timeout=timeout_s)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

        super().__init__(local_path)
        self.url = url


def open_source(mode: str, **kwargs) -> VideoSource:
    """
    Factory used by the Streamlit app.

    mode: one of "file", "webcam", "url"
    kwargs:
        file  -> path: str
        webcam -> device_index: int = 0
        url   -> url: str
    """
    if mode == "file":
        return FileVideoSource(kwargs["path"])
    if mode == "webcam":
        return WebcamVideoSource(kwargs.get("device_index", 0))
    if mode == "url":
        return URLVideoSource(kwargs["url"])
    raise ValueError(f"Unknown video source mode: {mode}")
