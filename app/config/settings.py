"""Typed application configuration loaded from configs/default.yaml.

Keys marked RESERVED below are parsed and validated but not read by the
current pipeline or Streamlit app. They have no effect yet.

Loading rules (see `read_config`):
  * An explicitly requested file that is missing or malformed is an error.
    Defaults are never substituted silently for it.
  * The default file being absent is not an error: built-in defaults are used
    and a warning says so.
  * Unknown keys are reported as warnings and otherwise ignored.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import yaml

DEFAULT_CONFIG_PATH = "configs/default.yaml"


class ConfigError(ValueError):
    """Raised by load_config when a configuration cannot be used."""


class ProcessingMode(str, Enum):
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    ACCURACY = "accuracy"


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    processing_width: int = 640
    inference_interval: int = 2
    frame_skip: int = 0  # RESERVED: not used; scheduling uses inference_interval
    classes: list[int] = field(default_factory=lambda: [2, 3, 5, 7])
    device: str = "cpu"


@dataclass
class TrackingConfig:
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30  # counted in inference updates, not video frames
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 30  # passed to ByteTrack; not derived from the video FPS
    minimum_consecutive_frames: int = 1
    trajectory_length: int = 40


@dataclass
class SpeedConfig:
    enabled: bool = True  # honoured by FrameProcessor
    pixels_per_meter: float | None = None
    calibration_points: list[list[float]] | None = None
    calibration_real_distance_m: float | None = None
    smoothing_window: int = 5  # values below 2 are treated as 2
    max_plausible_kmh: float = 220.0


@dataclass
class UIConfig:
    ui_update_interval: int = 5  # RESERVED
    show_trajectories: bool = True
    show_labels: bool = True
    max_table_rows: int = 200  # RESERVED
    theme_accent: str = "#2dd4bf"  # RESERVED


@dataclass
class DemoConfig:  # RESERVED: whole section is unused
    enabled: bool = False
    demo_resolution_width: int = 480
    demo_inference_every_n_frames: int = 2
    reduce_ui_overhead: bool = True


@dataclass
class AppConfig:
    processing_mode: ProcessingMode = ProcessingMode.BALANCED  # RESERVED: presets not applied
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)


@dataclass
class ConfigResult:
    """Outcome of reading a configuration file.

    config:   the parsed configuration, or None if `error` is set.
    source:   the file that was loaded, or None when built-in defaults were used.
    warnings: non-fatal problems (unknown keys, default file absent).
    error:    fatal problem; the configuration must not be used.
    """
    config: Optional[AppConfig]
    source: Optional[Path]
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None


_SECTIONS = {
    "detection": DetectionConfig,
    "tracking": TrackingConfig,
    "speed": SpeedConfig,
    "ui": UIConfig,
    "demo": DemoConfig,
}
_TOP_LEVEL = set(_SECTIONS) | {"processing_mode"}


def _section(name: str, cls: Any, values: Any, warnings: list[str]):
    if values is None:
        return cls()
    if not isinstance(values, dict):
        raise ValueError(f"'{name}' must be a mapping")
    allowed = {f.name for f in fields(cls)}
    for key in values:
        if key not in allowed:
            warnings.append(f"Unknown key '{name}.{key}' was ignored.")
    return cls(**{k: v for k, v in values.items() if k in allowed})


def _build(data: dict, warnings: list[str]) -> AppConfig:
    for key in data:
        if key not in _TOP_LEVEL:
            warnings.append(f"Unknown top-level key '{key}' was ignored.")
    try:
        mode = ProcessingMode(data.get("processing_mode", "balanced"))
    except ValueError as exc:
        raise ValueError("processing_mode must be performance, balanced, or accuracy") from exc
    cfg = AppConfig(
        processing_mode=mode,
        **{name: _section(name, cls, data.get(name), warnings) for name, cls in _SECTIONS.items()},
    )
    if not 0 <= cfg.detection.confidence_threshold <= 1:
        raise ValueError("detection.confidence_threshold must be between 0 and 1")
    if not 0 <= cfg.detection.iou_threshold <= 1:
        raise ValueError("detection.iou_threshold must be between 0 and 1")
    if cfg.detection.processing_width < 1 or cfg.detection.inference_interval < 1:
        raise ValueError("processing_width and inference_interval must be positive")
    if not cfg.detection.classes:
        raise ValueError("detection.classes must not be empty")
    if cfg.speed.pixels_per_meter is not None and cfg.speed.pixels_per_meter <= 0:
        raise ValueError("speed.pixels_per_meter must be positive when provided")
    # Two image points plus their known real-world separation. An explicit
    # pixels_per_meter value takes priority.
    if cfg.speed.pixels_per_meter is None and cfg.speed.calibration_points is not None:
        points = cfg.speed.calibration_points
        distance = cfg.speed.calibration_real_distance_m
        if len(points) != 2 or any(len(point) != 2 for point in points):
            raise ValueError("speed.calibration_points must contain exactly two [x, y] points")
        if distance is None or distance <= 0:
            raise ValueError("speed.calibration_real_distance_m must be positive when calibration_points are set")
        pixel_distance = math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
        if pixel_distance <= 0:
            raise ValueError("speed.calibration_points must not be identical")
        cfg.speed.pixels_per_meter = pixel_distance / distance
    return cfg


def read_config(path: str | Path | None = None, *, explicit: bool | None = None) -> ConfigResult:
    """Read and validate a configuration file without raising.

    path=None means the default file. `explicit` says whether the caller asked
    for this exact file; it defaults to True when a path is given. An explicit
    file that is missing is an error. A missing default file falls back to
    built-in defaults with a warning.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
        explicit = False if explicit is None else explicit
    elif explicit is None:
        explicit = True
    config_path = Path(path)
    if not config_path.is_file():
        if explicit:
            return ConfigResult(None, None, error=f"Configuration file not found: {config_path}")
        return ConfigResult(
            AppConfig(), None,
            warnings=[f"Default configuration '{config_path}' not found; built-in defaults are used."],
        )
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        return ConfigResult(None, config_path, error=f"Malformed YAML in {config_path}: {exc}")
    except (OSError, UnicodeDecodeError) as exc:
        return ConfigResult(None, config_path, error=f"Cannot read {config_path}: {exc}")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return ConfigResult(None, config_path, error=f"Configuration must be a YAML mapping: {config_path}")
    warnings: list[str] = []
    try:
        cfg = _build(data, warnings)
    except (ValueError, TypeError) as exc:
        return ConfigResult(None, config_path, error=f"Invalid configuration in {config_path}: {exc}")
    return ConfigResult(cfg, config_path, warnings=warnings)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Return the configuration or raise ConfigError.

    With no path, the default file is used if present, otherwise built-in defaults.
    With an explicit path, a missing or invalid file raises ConfigError.
    """
    result = read_config(path)
    if result.error is not None or result.config is None:
        raise ConfigError(result.error or "configuration could not be loaded")
    return result.config


def get_processing_mode_presets() -> dict[ProcessingMode, dict[str, Any]]:
    """Return UI-friendly overrides. Not applied anywhere yet (RESERVED)."""
    return {
        ProcessingMode.PERFORMANCE: {"processing_width": 480, "inference_interval": 3},
        ProcessingMode.BALANCED: {"processing_width": 640, "inference_interval": 2},
        ProcessingMode.ACCURACY: {"processing_width": 960, "inference_interval": 1},
    }
