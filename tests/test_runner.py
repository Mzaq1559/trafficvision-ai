"""Processing path (no browser): real video bytes, injected threshold detector.
Covers valid, missing and malformed configuration, bad video data, and
isolation between runs."""
from pathlib import Path

import cv2
import numpy as np

from app.runner import process_video
from tests.synthetic import ThresholdDetector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = str(ROOT / "configs" / "default.yaml")


def _video_bytes(tmp_path: Path, start_x: int, name: str) -> bytes:
    path = tmp_path / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    assert writer.isOpened()
    for i in range(60):
        frame = np.zeros((240, 320, 3), np.uint8)
        x = start_x + 4 * i
        cv2.rectangle(frame, (x, 100), (x + 40, 130), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def _run(data, config_path=DEFAULT_CFG, key="k", max_frames=60):
    return process_video(data, "clip.mp4", config_path, max_frames, key,
                         make_detector=lambda cfg: ThresholdDetector())


def test_valid_run_reports_config_source_and_results(tmp_path):
    result = _run(_video_bytes(tmp_path, 20, "a.mp4"))
    assert result["error"] is None
    assert result["config_source"] == DEFAULT_CFG
    assert result["frames"] == 60
    assert result["annotated_rgb"].shape == (240, 320, 3)
    assert len(result["table"]) >= 1


def test_missing_explicit_config_is_reported_not_defaulted(tmp_path):
    result = _run(_video_bytes(tmp_path, 20, "a.mp4"), config_path=str(tmp_path / "nope.yaml"))
    assert "not found" in result["error"]
    assert "frames" not in result


def test_malformed_config_is_reported_not_defaulted(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("detection: [oops\n", encoding="utf-8")
    result = _run(_video_bytes(tmp_path, 20, "a.mp4"), config_path=str(bad))
    assert "Malformed YAML" in result["error"]
    assert "frames" not in result


def test_invalid_value_config_is_reported(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("tracking:\n  lost_track_buffer: lots\n", encoding="utf-8")
    result = _run(_video_bytes(tmp_path, 20, "a.mp4"), config_path=str(bad))
    assert result["error"] is not None and "frames" not in result


def test_corrupt_video_bytes_are_reported(tmp_path):
    result = _run(b"this is not a video")
    assert "OpenCV could not decode" in result["error"]


def test_unknown_keys_surface_as_warnings(tmp_path):
    cfg = tmp_path / "typo.yaml"
    cfg.write_text("detection:\n  inference_intervall: 5\n", encoding="utf-8")
    result = _run(_video_bytes(tmp_path, 20, "a.mp4"), config_path=str(cfg))
    assert result["error"] is None
    assert any("inference_intervall" in w for w in result["warnings"])


def test_runs_do_not_leak_tracks_between_videos(tmp_path):
    first = _run(_video_bytes(tmp_path, 20, "a.mp4"), key="a")
    second = _run(_video_bytes(tmp_path, 20, "b.mp4"), key="b")
    assert first["summary"]["total_vehicles"] == 1
    assert second["summary"]["total_vehicles"] == 1
    # Track IDs are not guaranteed to restart at 1: supervision's ByteTrack uses a
    # process-wide counter. Only the set of tracks is required to be isolated.
    assert len(second["table"]) == 1
    assert {r["track_id"] for r in first["table"]}.isdisjoint(
        {r["track_id"] for r in second["table"]})


def test_temporary_video_file_is_removed(tmp_path, monkeypatch):
    import app.runner as runner
    created = []
    real = runner.tempfile.NamedTemporaryFile

    def spy(*args, **kwargs):
        handle = real(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr(runner.tempfile, "NamedTemporaryFile", spy)
    _run(_video_bytes(tmp_path, 20, "a.mp4"))
    assert created and not any(Path(p).exists() for p in created)
