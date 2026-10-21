"""Speed must not be taken from boxes clipped by the frame border.

Ground truth: 4 px/frame at 25 fps = 100 px/s. With 10 px = 1 m that is
10 m/s = 36.0 km/h. A vehicle moving at constant speed must read 36.0 km/h
on every sample where it is fully visible, and no sample may deviate from
that value (the last reliable value is kept while the box is clipped).
"""
import pytest

from tests.synthetic import box_frame, run_pipeline, speed_samples

FPS = 25.0
TRUE_KMH = 36.0  # 100 px/s -> 10 m/s -> 36 km/h at 10 px/m


def _path(start, step, n):
    x0, y0 = start
    dx, dy = step
    return [(x0 + dx * i, y0 + dy * i) for i in range(n)]


@pytest.mark.parametrize("name, start, step", [
    ("right edge", (20, 100), (4, 0)),
    ("left edge", (280, 100), (-4, 0)),
    ("bottom edge", (100, 20), (0, 4)),
    ("top edge", (100, 200), (0, -4)),
])
def test_clipped_samples_do_not_change_speed(name, start, step):
    frames = [box_frame(x, y) for x, y in _path(start, step, 90)]
    _, results = run_pipeline(frames, FPS, pixels_per_meter=10.0)
    samples = speed_samples(results)
    assert len(samples) >= 10, f"{name}: too few speed samples to be meaningful"
    worst = max(abs(s - TRUE_KMH) for _, _, s, _ in samples)
    assert worst < 0.5, f"{name}: speed deviated by {worst:.1f} km/h (clipped sample used?)"


def test_fully_visible_motion_reads_ground_truth():
    frames = [box_frame(x, y) for x, y in _path((60, 100), (4, 0), 40)]
    _, results = run_pipeline(frames, FPS, pixels_per_meter=10.0)
    samples = speed_samples(results)
    assert samples
    assert all(abs(s - TRUE_KMH) < 0.5 for _, _, s, _ in samples)
    assert all(calibrated for _, _, _, calibrated in samples)


def test_missing_calibration_reports_px_per_s_without_error():
    frames = [box_frame(x, y) for x, y in _path((60, 100), (4, 0), 40)]
    _, results = run_pipeline(frames, FPS, pixels_per_meter=None)
    samples = speed_samples(results)
    assert samples
    assert all(abs(s - 100.0) < 0.5 for _, _, s, _ in samples)  # px/s, not km/h
    assert all(not calibrated for _, _, _, calibrated in samples)
