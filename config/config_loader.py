"""Configuration loader module with typed data structure and fallback defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class SystemConfig:
    device: str = "auto"
    log_level: str = "INFO"
    log_file: str = "logs/system.log"


@dataclass
class ModelsConfig:
    object_model: str = "yolo11l.pt"
    pose_model: str = "yolo11s-pose.pt"


@dataclass
class DetectionConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    wrist_min_conf: float = 0.30


@dataclass
class TrackingConfig:
    persist: bool = True
    tracker_type: str = "bytetrack.yaml"


@dataclass
class ROIConfig:
    margin: int = 40
    padding: int = 25
    proximity_limit: int = 80
    proximity_factor: float = 0.6


@dataclass
class AssociationConfig:
    max_distance: float = 250.0
    overlap_weight: float = 0.4
    distance_weight: float = 0.6
    temporal_smooth_frames: int = 5
    max_lost_frames: int = 30


@dataclass
class UnattendedBagConfig:
    timeout_seconds: float = 10.0
    movement_threshold: float = 15.0
    min_separation_distance: float = 180.0


@dataclass
class RestrictedZoneConfig:
    dwell_time_seconds: float = 3.0


@dataclass
class AlertsConfig:
    cooldown_seconds: float = 5.0


@dataclass
class SafetyConfig:
    unattended_bag: UnattendedBagConfig = field(default_factory=UnattendedBagConfig)
    restricted_zone: RestrictedZoneConfig = field(default_factory=RestrictedZoneConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)


@dataclass
class PPEConfig:
    enabled: bool = False
    model_path: Optional[str] = None


@dataclass
class SnapshotsConfig:
    enabled: bool = True
    output_dir: str = "events"
    save_severities: List[str] = field(default_factory=lambda: ["WARNING", "CRITICAL"])


@dataclass
class ReportingConfig:
    enabled: bool = True
    output_dir: str = "reports"
    export_json: bool = True
    export_csv: bool = True


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ppe: PPEConfig = field(default_factory=PPEConfig)
    snapshots: SnapshotsConfig = field(default_factory=SnapshotsConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def _dict_to_dataclass(cls: Any, data: Dict[str, Any]) -> Any:
    """Helper to populate dataclass fields safely from a dict."""
    if not isinstance(data, dict):
        return cls()
    init_kwargs = {}
    for f_name, f_def in cls.__dataclass_fields__.items():
        if f_name in data:
            val = data[f_name]
            # Recursively instantiate nested dataclasses
            if hasattr(f_def.type, "__dataclass_fields__"):
                init_kwargs[f_name] = _dict_to_dataclass(f_def.type, val)
            else:
                init_kwargs[f_name] = val
    return cls(**init_kwargs)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file or return defaults if absent."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(base_dir, "config", "config.yaml")
        config_path = default_path if os.path.exists(default_path) else "config.yaml"

    if not os.path.exists(config_path):
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    app_cfg = AppConfig()

    if "system" in raw_data:
        app_cfg.system = _dict_to_dataclass(SystemConfig, raw_data["system"])
    if "models" in raw_data:
        app_cfg.models = _dict_to_dataclass(ModelsConfig, raw_data["models"])
    if "detection" in raw_data:
        app_cfg.detection = _dict_to_dataclass(DetectionConfig, raw_data["detection"])
    if "tracking" in raw_data:
        app_cfg.tracking = _dict_to_dataclass(TrackingConfig, raw_data["tracking"])
    if "roi" in raw_data:
        app_cfg.roi = _dict_to_dataclass(ROIConfig, raw_data["roi"])
    if "association" in raw_data:
        app_cfg.association = _dict_to_dataclass(AssociationConfig, raw_data["association"])
    if "safety" in raw_data:
        s_data = raw_data["safety"]
        app_cfg.safety.unattended_bag = _dict_to_dataclass(
            UnattendedBagConfig, s_data.get("unattended_bag", {})
        )
        app_cfg.safety.restricted_zone = _dict_to_dataclass(
            RestrictedZoneConfig, s_data.get("restricted_zone", {})
        )
        app_cfg.safety.alerts = _dict_to_dataclass(
            AlertsConfig, s_data.get("alerts", {})
        )
    if "ppe" in raw_data:
        app_cfg.ppe = _dict_to_dataclass(PPEConfig, raw_data["ppe"])
    if "snapshots" in raw_data:
        app_cfg.snapshots = _dict_to_dataclass(SnapshotsConfig, raw_data["snapshots"])
    if "reporting" in raw_data:
        app_cfg.reporting = _dict_to_dataclass(ReportingConfig, raw_data["reporting"])

    return app_cfg
