"""Trajectory, direction, calibration and speed-estimator unit tests."""
from pathlib import Path

import pytest

from app.config import ConfigError, ProcessingMode, load_config
from app.speed.estimator import SpeedEstimator, calibrate_pixels_per_meter
from app.utils.trajectory import estimate_direction, trajectory_length_px

ROOT = Path(__file__).resolve().parents[1]


def test_direction_east():
    assert estimate_direction([(0, 0, 0), (20, 0, 1)]) == "East"


def test_stationary_and_insufficient_points():
    assert estimate_direction([(0, 0, 0), (3, 2, 1)]) == "Stationary"
    assert estimate_direction([]) is None
    assert estimate_direction([(1, 2, 0)]) is None


def test_trajectory_length():
    assert trajectory_length_px([(0,0,0), (3,4,1), (6,8,2)]) == pytest.approx(10)


def test_calibration():
    assert calibrate_pixels_per_meter((0,0), (30,40), 10) == pytest.approx(5)
    with pytest.raises(ValueError):
        calibrate_pixels_per_meter((0,0), (1,1), 0)
    with pytest.raises(ValueError):
        calibrate_pixels_per_meter((1,1), (1,1), 2)


def test_speed_approximate_and_calibrated():
    result = SpeedEstimator(smoothing_window=2).estimate(1, [(0,0,0), (3,4,1)])
    assert result.value == pytest.approx(5)
    assert result.unit == "px/s" and not result.is_calibrated
    result = SpeedEstimator(pixels_per_meter=2).estimate(1, [(0,0,0), (10,0,1)])
    assert result.value == pytest.approx(18)
    assert result.unit == "km/h" and result.is_calibrated


def test_nonincreasing_timestamps_are_ignored():
    assert SpeedEstimator().estimate(1, [(0,0,1), (10,0,1)]) is None


def test_config_defaults_and_yaml():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.processing_mode is ProcessingMode.BALANCED
    assert cfg.detection.classes == [2,3,5,7]
    assert cfg.speed.pixels_per_meter is None


def test_missing_explicit_config_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_invalid_config_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("detection:\n  confidence_threshold: 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_config_derives_speed_calibration_from_reference_points(tmp_path):
    p = tmp_path / "calibrated.yaml"
    p.write_text(
        "speed:\n  calibration_points: [[0, 0], [30, 40]]\n  calibration_real_distance_m: 10\n",
        encoding="utf-8",
    )
    assert load_config(p).speed.pixels_per_meter == pytest.approx(5.0)
