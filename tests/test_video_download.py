"""URL video cache: interrupted downloads, cache preservation, and webcam cleanup.
Uses a local HTTP server; no network access is needed."""
import hashlib
import http.server
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.video import source as source_mod
from app.video.source import URLVideoSource, is_readable_video


def _make_video_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "valid.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def server(tmp_path):
    state = {"body": b"", "mode": "ok"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = state["body"]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "video/mp4")
            self.end_headers()
            if state["mode"] == "truncated":
                self.wfile.write(body[: len(body) // 2])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{httpd.server_address[1]}/clip.mp4"
    state["valid"] = _make_video_bytes(tmp_path)
    state["body"] = state["valid"]
    yield state
    httpd.shutdown()
    httpd.server_close()


def _cache_file(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"trafficvision_dl_{digest}.mp4"


def test_successful_download_is_cached_and_readable(server, tmp_path):
    src = URLVideoSource(server["url"], cache_dir=str(tmp_path))
    assert src.total_frames() == 10
    cache = _cache_file(tmp_path, server["url"])
    assert cache.is_file() and is_readable_video(str(cache))
    assert not list(tmp_path.glob("*.part"))


def test_interrupted_download_leaves_no_file_behind(server, tmp_path):
    server["body"] = server["valid"]
    server["mode"] = "truncated"
    with pytest.raises(Exception):
        URLVideoSource(server["url"], cache_dir=str(tmp_path))
    assert not _cache_file(tmp_path, server["url"]).exists()
    assert not list(tmp_path.glob("*.part"))


def test_failed_refresh_preserves_valid_cache(server, tmp_path):
    URLVideoSource(server["url"], cache_dir=str(tmp_path))  # valid cache
    cache = _cache_file(tmp_path, server["url"])
    before = cache.read_bytes()
    server["mode"] = "truncated"
    with pytest.raises(Exception):
        URLVideoSource(server["url"], cache_dir=str(tmp_path), refresh=True)
    assert cache.read_bytes() == before
    assert is_readable_video(str(cache))
    assert not list(tmp_path.glob("*.part"))


def test_corrupt_cache_is_replaced(server, tmp_path):
    cache = _cache_file(tmp_path, server["url"])
    cache.write_bytes(b"not a video")
    src = URLVideoSource(server["url"], cache_dir=str(tmp_path))
    assert src.total_frames() == 10


class _ClosedCapture:
    released = 0

    def __init__(self, *args, **kwargs):
        pass

    def isOpened(self):
        return False

    def release(self):
        _ClosedCapture.released += 1


def test_failed_webcam_open_releases_capture(monkeypatch):
    _ClosedCapture.released = 0
    monkeypatch.setattr(source_mod.cv2, "VideoCapture", _ClosedCapture)
    with pytest.raises(IOError):
        source_mod.WebcamVideoSource(0)
    assert _ClosedCapture.released == 1
