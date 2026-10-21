"""
Video input handling.

Supports three input modes:
  1. Recorded video upload (a local file path, e.g. from Streamlit's file_uploader)
  2. Camera / webcam input (device index) - only works where OpenCV can see a camera,
     which is NOT the case on Streamlit Community Cloud's hosted containers.
  3. Public demo video URLs (downloaded once into a cache file, then read as a normal file)

All sources expose the same minimal interface so the processing pipeline
never needs to know which one it is dealing with.

Note: the Streamlit entry point currently calls cv2.VideoCapture directly and
does not use these classes. They are tested in isolation only.
"""
from __future__ import annotations

import hashlib
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
    camera device, so this will fail to open. The capture is released before
    the error is raised.
    """

    def __init__(self, device_index: int = 0):
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            cap.release()
            raise IOError(
                f"Could not open camera device {device_index}. "
                "Webcam input is unavailable in hosted/cloud environments."
            )
        super().__init__(cap, live=True)
        self.device_index = device_index


def is_readable_video(path: str) -> bool:
    """True if OpenCV can open the file and reports at least one frame."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    cap = cv2.VideoCapture(path)
    try:
        return bool(cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0)
    finally:
        cap.release()


class URLVideoSource(FileVideoSource):
    """
    Downloads a public video URL into a cache file, then behaves like a
    FileVideoSource.

    The download goes to a temporary file first. It replaces the cache file
    only after the transfer completed (Content-Length matched when given) and
    the result opens as a video. A failed download removes its temporary file
    and leaves any previously valid cache file untouched.
    """

    def __init__(self, url: str, timeout_s: int = 60, cache_dir: Optional[str] = None,
                 refresh: bool = False):
        cache_dir = cache_dir or tempfile.gettempdir()
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        local_path = os.path.join(cache_dir, f"trafficvision_dl_{digest}.mp4")

        if refresh or not is_readable_video(local_path):
            download_atomically(url, local_path, timeout_s=timeout_s)

        super().__init__(local_path)
        self.url = url


def download_atomically(url: str, destination: str, timeout_s: int = 60) -> None:
    """Fetch `url` into `destination` or raise, never leaving a partial file behind."""
    import requests  # local import: only needed for this mode

    tmp_path = f"{destination}.{os.getpid()}.part"
    try:
        with requests.get(url, stream=True, timeout=timeout_s) as resp:
            resp.raise_for_status()
            expected = resp.headers.get("Content-Length")
            written = 0
            with open(tmp_path, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
        if expected is not None and written != int(expected):
            raise IOError(f"incomplete download: {written} of {expected} bytes")
        if not is_readable_video(tmp_path):
            raise IOError("downloaded file is not a readable video")
        os.replace(tmp_path, destination)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def open_source(mode: str, **kwargs) -> VideoSource:
    """
    Factory for the three source types.

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
