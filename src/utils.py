"""
Workplace Safety AI - Utility Functions and Benchmark Profiler
"""

import os
import sys
import time
import math
import logging
from typing import Dict, List, Tuple, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

try:
    import numpy as np
except ImportError:
    np = None


def setup_logger(log_file: str = "logs/application.log", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a dual-handler logger (stdout + rotating file).
    """
    logger = logging.getLogger("WorkplaceSafetyAI")
    logger.setLevel(level)

    # Prevent duplicate handlers if re-initialized
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as exc:
        print(f"Warning: Could not create log file handler at {log_file}: {exc}")

    return logger


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file safely.
    """
    if yaml is None:
        raise ImportError("PyYAML package is not installed. Run 'pip install pyyaml'.")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError(f"Invalid YAML content in {config_path}: expected dictionary root.")
            return config
        except yaml.YAMLError as err:
            raise ValueError(f"Error parsing YAML file {config_path}: {err}") from err


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert floating point seconds to MM:SS.s display timestamp format.
    Example: 42.34 -> '00:42.3'
    """
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    remaining_secs = seconds % 60
    return f"{minutes:02d}:{remaining_secs:04.1f}"


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0.0, x2 - x1)
    inter_height = max(0.0, y2 - y1)
    intersection = inter_width * inter_height

    if intersection <= 0:
        return 0.0

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def calculate_overlap_ratio(inner_box: List[float], outer_box: List[float]) -> float:
    """
    Calculate fraction of inner_box area contained within outer_box.
    Formula: intersection_area / inner_box_area.
    Useful for checking if a helmet/vest detection is enclosed inside a body region.
    """
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    inner_area = max(0.0, (inner_box[2] - inner_box[0])) * max(0.0, (inner_box[3] - inner_box[1]))
    if inner_area <= 0:
        return 0.0

    return inter_area / inner_area


def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two 2D points.
    """
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def scale_polygon_points(
    points: List[List[float]],
    frame_width: int,
    frame_height: int
) -> Any:
    """
    Scale polygon vertices to pixel dimensions.
    If points are normalized in [0.0, 1.0], scale by width/height.
    Otherwise, clamp to image dimensions.
    """
    scaled = []
    is_normalized = all(0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0 for pt in points)

    for pt in points:
        if is_normalized:
            px = int(round(pt[0] * frame_width))
            py = int(round(pt[1] * frame_height))
        else:
            px = int(round(pt[0]))
            py = int(round(pt[1]))
        # Clamp to frame boundary
        px = max(0, min(frame_width - 1, px))
        py = max(0, min(frame_height - 1, py))
        scaled.append([px, py])

    if np is not None:
        return np.array(scaled, dtype=np.int32)
    return scaled


class LatencyBenchmark:
    """
    Lightweight benchmark and FPS tracking helper.
    """

    def __init__(self, smoothing_window: int = 30):
        self.smoothing_window = smoothing_window
        self.stage_durations: Dict[str, float] = {}
        self.fps_history: List[float] = []
        self._last_frame_time: Optional[float] = None
        self.current_fps: float = 0.0

    def start_frame(self) -> None:
        """Call at beginning of frame processing."""
        now = time.perf_counter()
        if self._last_frame_time is not None:
            delta = now - self._last_frame_time
            if delta > 0:
                instant_fps = 1.0 / delta
                self.fps_history.append(instant_fps)
                if len(self.fps_history) > self.smoothing_window:
                    self.fps_history.pop(0)
                self.current_fps = sum(self.fps_history) / len(self.fps_history)
        self._last_frame_time = now

    def record_stage(self, stage_name: str, duration_sec: float) -> None:
        """Record duration of a specific pipeline stage in milliseconds."""
        self.stage_durations[stage_name] = duration_sec * 1000.0

    def get_stage_ms(self, stage_name: str) -> float:
        """Get latest recorded duration in milliseconds."""
        return self.stage_durations.get(stage_name, 0.0)

    def get_fps(self) -> float:
        """Get smoothed FPS."""
        return self.current_fps
