"""
Anomaly Evaluator Module.
Applies deterministic threshold rules from config.py to determine
whether an entity's behavior is Normal or Abnormal.
"""

from typing import Dict, List, Tuple
from core.action_engine import EntityKinematics
import config


class AnomalyEvaluator:
    """
    Evaluates kinematic states against operational safety thresholds.
    Filters noise and verifies temporal persistence.
    """

    def __init__(
        self,
        loitering_threshold_sec: float = config.LOITERING_DURATION_SECONDS,
        fall_confirm_sec: float = config.FALL_CONFIRMATION_SECONDS,
        running_confirm_sec: float = config.RUNNING_CONFIRMATION_SECONDS,
    ):
        self.loitering_threshold_sec = loitering_threshold_sec
        self.fall_confirm_sec = fall_confirm_sec
        self.running_confirm_sec = running_confirm_sec

        # State tracking for duration confirmation
        # track_id -> timestamp when fallen state first started
        self.fall_start_times: Dict[str, float] = {}
        # track_id -> timestamp when running state first started
        self.running_start_times: Dict[str, float] = {}

    def evaluate(
        self,
        kinematics: EntityKinematics,
        timestamp_sec: float,
    ) -> Tuple[bool, List[str], List[str], Tuple[int, int, int]]:
        """
        Evaluates an entity's kinematics.
        Returns:
            is_abnormal (bool)
            anomaly_types (List[str])
            reasons (List[str])
            box_color_bgr (Tuple[int, int, int]): (0, 255, 0) for Normal, (0, 0, 255) for Abnormal
        """
        eid = kinematics.entity_id
        anomaly_types: List[str] = []
        reasons: List[str] = []

        # 1. Evaluate Loitering (Stationary > threshold)
        if kinematics.stationary_duration_sec >= self.loitering_threshold_sec:
            anomaly_types.append("Loitering / Prolonged Inactivity")
            reasons.append(
                f"Stationary for {kinematics.stationary_duration_sec:.1f}s "
                f"(exceeds {self.loitering_threshold_sec:.1f}s threshold)"
            )

        # 2. Evaluate Fall / Man Down (Prone state > fall_confirm_sec)
        if kinematics.is_fallen:
            if eid not in self.fall_start_times or self.fall_start_times[eid] is None:
                self.fall_start_times[eid] = timestamp_sec
            fall_duration = timestamp_sec - self.fall_start_times[eid]
            if fall_duration >= self.fall_confirm_sec:
                anomaly_types.append("Slip & Fall / Man Down")
                reasons.append(
                    f"Entity prone on floor (torso angle: {kinematics.torso_angle_deg}°) "
                    f"for {fall_duration:.1f}s"
                )
        else:
            self.fall_start_times[eid] = None

        # 3. Evaluate Running / Panic (Running state > running_confirm_sec)
        if kinematics.is_running:
            if eid not in self.running_start_times or self.running_start_times[eid] is None:
                self.running_start_times[eid] = timestamp_sec
            run_duration = timestamp_sec - self.running_start_times[eid]
            if run_duration >= self.running_confirm_sec:
                anomaly_types.append("Running / Panic")
                reasons.append(
                    f"Abnormal velocity spike ({kinematics.speed_px_per_sec:.1f} px/s) "
                    f"for {run_duration:.1f}s"
                )
        else:
            self.running_start_times[eid] = None

        is_abnormal = len(anomaly_types) > 0
        # Color coding: Green for Normal, Red for Abnormal (BGR format for OpenCV)
        box_color = (0, 0, 255) if is_abnormal else (0, 255, 0)

        return is_abnormal, anomaly_types, reasons, box_color

    def purge_lost_entities(self, active_ids: List[str]):
        """Clean up state for lost entities."""
        current_tracked = set(active_ids)
        for eid in list(self.fall_start_times.keys()):
            if eid not in current_tracked:
                del self.fall_start_times[eid]
        for eid in list(self.running_start_times.keys()):
            if eid not in current_tracked:
                del self.running_start_times[eid]
