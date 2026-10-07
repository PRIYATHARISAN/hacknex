"""
Action Kinematics Engine.
Calculates pure physical motion facts (velocities, angles, aspect ratios)
and classifies the current action state (Walking, Stationary, Running, Fallen).
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EntityKinematics:
    entity_id: str
    current_action: str
    speed_px_per_sec: float
    torso_angle_deg: float
    aspect_ratio: float
    stationary_duration_sec: float
    is_fallen: bool
    is_running: bool
    is_stationary: bool
    bbox: List[int]
    centroid: List[int]


class ActionEngine:
    """
    Tracks trajectory history and keypoint postures per entity ID over a sliding window.
    """

    def __init__(
        self,
        window_size: int = 30,
        stationary_threshold: float = 30.0,
        running_threshold: float = 160.0,
        fall_angle_max: float = 35.0,
        fall_aspect_ratio_min: float = 0.95,
    ):
        self.window_size = window_size
        self.stationary_threshold = stationary_threshold
        self.running_threshold = running_threshold
        self.fall_angle_max = fall_angle_max
        self.fall_aspect_ratio_min = fall_aspect_ratio_min

        # State storage per track_id
        # track_id -> deque of (timestamp, centroid_x, centroid_y, bbox, keypoints)
        self.history: Dict[str, deque] = {}
        # track_id -> timestamp when stationary state began
        self.stationary_start_time: Dict[str, float] = {}

    def update_entity(
        self,
        entity_id: str,
        bbox: List[int],  # [x1, y1, x2, y2]
        keypoints: Optional[np.ndarray],  # shape (17, 2 or 3)
        timestamp_sec: float,
    ) -> EntityKinematics:
        """
        Updates tracking history for an entity and computes current kinematics and action.
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_ratio = width / height

        if entity_id not in self.history:
            self.history[entity_id] = deque(maxlen=self.window_size)
            self.stationary_start_time[entity_id] = timestamp_sec

        hist = self.history[entity_id]
        hist.append((timestamp_sec, cx, cy, bbox, keypoints))

        # 1. Compute Centroid Velocity (pixels per second)
        speed = 0.0
        if len(hist) >= 2:
            t_old, ox, oy, _, _ = hist[0]
            dt = timestamp_sec - t_old
            if dt > 0.05:
                dist = math.hypot(cx - ox, cy - oy)
                speed = dist / dt

        # 2. Compute Torso Angle from Pose Keypoints
        torso_angle = self._compute_torso_angle(keypoints)

        # 3. Determine Fallen Condition
        # Condition A: Torso angle <= fall_angle_max (horizontal)
        # Condition B: High aspect ratio (W / H >= 0.95) as fallback if keypoints occluded
        is_fallen = False
        if torso_angle is not None and torso_angle <= self.fall_angle_max:
            is_fallen = True
        elif torso_angle is None and aspect_ratio >= self.fall_aspect_ratio_min:
            is_fallen = True

        # 4. Stationary Duration Dwell Calculation
        is_stationary = speed < self.stationary_threshold and not is_fallen
        if is_stationary:
            if entity_id not in self.stationary_start_time or self.stationary_start_time[entity_id] is None:
                self.stationary_start_time[entity_id] = timestamp_sec
            stationary_duration = timestamp_sec - self.stationary_start_time[entity_id]
        else:
            self.stationary_start_time[entity_id] = None
            stationary_duration = 0.0

        # 5. Running Condition
        is_running = speed >= self.running_threshold and not is_fallen

        # 6. Action Label
        if is_fallen:
            current_action = "Fallen / Prone"
        elif is_running:
            current_action = "Running"
        elif is_stationary:
            current_action = "Stationary / Idle"
        else:
            current_action = "Walking"

        return EntityKinematics(
            entity_id=entity_id,
            current_action=current_action,
            speed_px_per_sec=round(speed, 2),
            torso_angle_deg=round(torso_angle if torso_angle is not None else 90.0, 1),
            aspect_ratio=round(aspect_ratio, 2),
            stationary_duration_sec=round(stationary_duration, 1),
            is_fallen=is_fallen,
            is_running=is_running,
            is_stationary=is_stationary,
            bbox=[int(x1), int(y1), int(x2), int(y2)],
            centroid=[cx, cy],
        )

    def _compute_torso_angle(self, keypoints: Optional[np.ndarray]) -> Optional[float]:
        """
        Calculates torso angle relative to horizontal ground (0 deg = lying flat, 90 deg = standing).
        Uses shoulders (idx 5, 6) and hips (idx 11, 12).
        """
        if keypoints is None or len(keypoints) < 13:
            return None

        # Extract (x, y) coordinates
        # COCO keypoints: 5=L shoulder, 6=R shoulder, 11=L hip, 12=R hip
        try:
            ls, rs = keypoints[5][:2], keypoints[6][:2]
            lh, rh = keypoints[11][:2], keypoints[12][:2]

            # If confidence values exist and are too low, skip
            if len(keypoints[5]) > 2 and (keypoints[5][2] < 0.25 and keypoints[6][2] < 0.25):
                return None
            if len(keypoints[11]) > 2 and (keypoints[11][2] < 0.25 and keypoints[12][2] < 0.25):
                return None

            mid_shoulder = np.array([(ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0])
            mid_hip = np.array([(lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0])

            dx = abs(mid_shoulder[0] - mid_hip[0])
            dy = abs(mid_shoulder[1] - mid_hip[1])

            # Angle from ground horizontal (degrees)
            angle_rad = math.atan2(dy, dx)
            return math.degrees(angle_rad)
        except Exception:
            return None

    def purge_lost_entities(self, active_ids: List[str]):
        """Clean up state for entities that have exited the scene."""
        current_tracked = set(active_ids)
        for eid in list(self.history.keys()):
            if eid not in current_tracked:
                del self.history[eid]
                if eid in self.stationary_start_time:
                    del self.stationary_start_time[eid]
