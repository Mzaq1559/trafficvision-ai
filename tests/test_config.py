"""Configuration loading: valid, missing, malformed, and unknown-key cases."""
from pathlib import Path

import pytest

from app.config import ConfigError, load_config, read_config
from app.config.settings import DEFAULT_CONFIG_PATH
from app.detection.detector import VehicleDetector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / "configs" / "default.yaml"


def test_valid_default_file_reports_its_source():
    result = read_config(DEFAULT_FILE)
    assert result.error is None
    assert result.source == DEFAULT_FILE
    assert result.warnings == []


def test_missing_explicit_file_is_an_error(tmp_path):
    result = read_config(tmp_path / "nope.yaml")
    assert result.config is None
    assert "not found" in result.error


def test_missing_explicit_file_raises_in_load_config(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_missing_default_falls_back_to_builtin_with_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no configs/default.yaml here
    result = read_config()  # path=None -> default, not explicit
    assert result.error is None
    assert result.source is None
    assert any("built-in defaults" in w for w in result.warnings)
    assert result.config.detection.model_name == "yolov8n.pt"


def test_malformed_yaml_is_an_error_not_defaults(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("detection: [unclosed\n", encoding="utf-8")
    result = read_config(bad)
    assert result.config is None
    assert "Malformed YAML" in result.error


def test_non_mapping_root_is_an_error(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    assert "must be a YAML mapping" in read_config(p).error


def test_section_that_is_not_a_mapping_is_an_error(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("detection: 5\n", encoding="utf-8")
    assert "'detection' must be a mapping" in read_config(p).error


def test_wrong_type_is_an_error(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("detection:\n  confidence_threshold: high\n", encoding="utf-8")
    result = read_config(p)
    assert result.config is None
    assert "Invalid configuration" in result.error


def test_out_of_range_value_is_an_error(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("detection:\n  confidence_threshold: 2\n", encoding="utf-8")
    assert "between 0 and 1" in read_config(p).error


def test_unknown_keys_are_reported_and_ignored(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text("detection:\n  inference_intervall: 5\nspeed:\n  enabled: false\nwhatever: 1\n",
                 encoding="utf-8")
    result = read_config(p)
    assert result.error is None
    assert result.config.speed.enabled is False
    assert result.config.detection.inference_interval == 2  # typo did not take effect
    text = " ".join(result.warnings)
    assert "detection.inference_intervall" in text and "whatever" in text


def test_empty_classes_list_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("detection:\n  classes: []\n", encoding="utf-8")
    assert "must not be empty" in read_config(p).error


def test_empty_file_means_all_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    result = read_config(p)
    assert result.error is None and result.config.detection.model_name == "yolov8n.pt"


def test_detector_rejects_empty_classes_but_accepts_none():
    with pytest.raises(ValueError, match="must not be empty"):
        VehicleDetector(classes=[])
    assert VehicleDetector(classes=None).classes == [2, 3, 5, 7]


def test_default_path_constant_matches_shipped_file():
    assert DEFAULT_CONFIG_PATH == "configs/default.yaml"
    assert DEFAULT_FILE.is_file()
