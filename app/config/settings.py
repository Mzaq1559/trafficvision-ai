"""Typed application configuration loaded from configs/default.yaml."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any
import yaml

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
    frame_skip: int = 0
    classes: list[int] = field(default_factory=lambda: [2, 3, 5, 7])
    device: str = "cpu"

@dataclass
class TrackingConfig:
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 30
    minimum_consecutive_frames: int = 1
    trajectory_length: int = 40

@dataclass
class SpeedConfig:
    enabled: bool = True
    pixels_per_meter: float | None = None
    calibration_points: list[list[float]] | None = None
    calibration_real_distance_m: float | None = None
    smoothing_window: int = 5
    max_plausible_kmh: float = 220.0

@dataclass
class UIConfig:
    ui_update_interval: int = 5
    show_trajectories: bool = True
    show_labels: bool = True
    max_table_rows: int = 200
    theme_accent: str = "#2dd4bf"

@dataclass
class DemoConfig:
    enabled: bool = False
    demo_resolution_width: int = 480
    demo_inference_every_n_frames: int = 2
    reduce_ui_overhead: bool = True

@dataclass
class AppConfig:
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

def _section(cls: type, values: dict[str, Any] | None):
    values = values or {}
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{k: v for k, v in values.items() if k in allowed})

def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load YAML configuration, falling back to documented defaults if absent."""
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    try:
        mode = ProcessingMode(data.get("processing_mode", "balanced"))
    except ValueError as exc:
        raise ValueError("processing_mode must be performance, balanced, or accuracy") from exc
    cfg = AppConfig(
        processing_mode=mode,
        detection=_section(DetectionConfig, data.get("detection")),
        tracking=_section(TrackingConfig, data.get("tracking")),
        speed=_section(SpeedConfig, data.get("speed")),
        ui=_section(UIConfig, data.get("ui")),
        demo=_section(DemoConfig, data.get("demo")),
    )
    if not 0 <= cfg.detection.confidence_threshold <= 1:
        raise ValueError("detection.confidence_threshold must be between 0 and 1")
    if not 0 <= cfg.detection.iou_threshold <= 1:
        raise ValueError("detection.iou_threshold must be between 0 and 1")
    if cfg.detection.processing_width < 1 or cfg.detection.inference_interval < 1:
        raise ValueError("processing_width and inference_interval must be positive")
    if cfg.speed.pixels_per_meter is not None and cfg.speed.pixels_per_meter <= 0:
        raise ValueError("speed.pixels_per_meter must be positive when provided")
    # Permit calibration to be specified as two image points plus their known
    # real-world separation. An explicit pixels_per_meter value takes priority.
    if cfg.speed.pixels_per_meter is None and cfg.speed.calibration_points is not None:
        points = cfg.speed.calibration_points
        distance = cfg.speed.calibration_real_distance_m
        if len(points) != 2 or any(len(point) != 2 for point in points):
            raise ValueError("speed.calibration_points must contain exactly two [x, y] points")
        if distance is None or distance <= 0:
            raise ValueError("speed.calibration_real_distance_m must be positive when calibration_points are set")
        import math
        pixel_distance = math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
        if pixel_distance <= 0:
            raise ValueError("speed.calibration_points must not be identical")
        cfg.speed.pixels_per_meter = pixel_distance / distance
    return cfg

def get_processing_mode_presets() -> dict[ProcessingMode, dict[str, Any]]:
    """Return UI-friendly overrides; callers decide whether/how to apply them."""
    return {
        ProcessingMode.PERFORMANCE: {"processing_width": 480, "inference_interval": 3},
        ProcessingMode.BALANCED: {"processing_width": 640, "inference_interval": 2},
        ProcessingMode.ACCURACY: {"processing_width": 960, "inference_interval": 1},
    }
