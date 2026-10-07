"""
Workplace Safety AI - Pose Estimation & Posture/Fall Analysis using YOLO11s-pose
"""

from __future__ import annotations
import os
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.utils import calculate_iou


class PoseEstimator:
    """
    Performs human pose estimation and biomechanical posture analysis.
    Extracts key anatomical landmarks (shoulders, hips, knees, ankles)
    to detect abnormal postures, slouching, or suspected falls.
    """

    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(
        self,
        model_path: str = "models/yolo11s-pose.pt",
        confidence: float = 0.35,
        pose_every_n_frames: int = 2,
        device: str = "auto",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Pose Estimator.

        Args:
            model_path: Path to YOLO pose weights.
            confidence: Keypoint confidence threshold.
            pose_every_n_frames: Interval between full pose inferences.
            device: 'cuda', 'cpu', or 'auto'.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("WorkplaceSafetyAI.PoseEstimator")
        self.confidence = float(confidence)
        self.pose_every_n_frames = max(1, int(pose_every_n_frames))
        self.device = device
        self.model_path = model_path
        self._frame_counter = 0

        self.model = self._load_pose_model(model_path)
        self.cached_poses: Dict[int, Dict[str, Any]] = {}

    def _load_pose_model(self, path: str) -> Any:
        """Load YOLO pose model weights safely."""
        if YOLO is None:
            self.logger.warning("Ultralytics library unavailable. Pose estimator disabled.")
            return None

        candidates = [
            path,
            os.path.join("models", os.path.basename(path)),
            "yolo11s-pose.pt",
            "yolov8s-pose.pt"
        ]

        for cand in candidates:
            if os.path.isfile(cand):
                self.logger.info("Loading pose model from: %s", cand)
                try:
                    return YOLO(cand)
                except Exception as err:
                    self.logger.error("Failed loading pose model from %s: %s", cand, err)

        self.logger.warning("Pose model weights not found at %s. Attempting online fetch.", path)
        try:
            return YOLO(os.path.basename(path))
        except Exception as exc:
            self.logger.warning("Could not instantiate pose model: %s. Pose analysis will be skipped.", exc)
            return None

    def estimate(
        self,
        frame: np.ndarray,
        workers: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Estimate pose keypoints for workers in the current frame.
        Applies frame interval skipping to maximize runtime throughput.

        Args:
            frame: Current BGR image.
            workers: Current tracked workers dictionary.

        Returns:
            Dictionary mapping person_id -> pose_info:
            {
                person_id: {
                    "keypoints": Dict[str, Tuple[float, float, float]],
                    "torso_angle_deg": float,
                    "fall_suspected": bool,
                    "confidence": float
                }
            }
        """
        self._frame_counter += 1

        # Skip inference on skipped frames and reuse cached keypoints if available
        if (self._frame_counter % self.pose_every_n_frames != 0) and self.cached_poses:
            # Refresh pose reference for active workers
            for pid, worker in workers.items():
                if pid in self.cached_poses:
                    worker["pose_data"] = self.cached_poses[pid]
            return self.cached_poses

        if self.model is None or frame is None or not workers:
            return self.cached_poses

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                verbose=False
            )
        except Exception as exc:
            self.logger.error("Pose inference failure: %s", exc)
            return self.cached_poses

        if not results or len(results) == 0 or results[0].keypoints is None:
            return self.cached_poses

        res = results[0]
        kpts_data = res.keypoints.data.cpu().numpy()  # Shape: (N, 17, 3)
        boxes_data = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

        new_cached: Dict[int, Dict[str, Any]] = {}

        # Associate each detected pose instance with tracked workers via IoU
        for idx, (pose_kpts, p_box) in enumerate(zip(kpts_data, boxes_data)):
            best_pid = None
            best_iou = 0.0

            for pid, worker in workers.items():
                iou = calculate_iou(list(p_box), worker["bbox"])
                if iou > best_iou and iou >= 0.3:
                    best_iou = iou
                    best_pid = pid

            if best_pid is None:
                continue

            # Parse named keypoints
            keypoints_dict = {}
            for k_idx, name in enumerate(self.KEYPOINT_NAMES):
                if k_idx < len(pose_kpts):
                    kx, ky, kconf = pose_kpts[k_idx]
                    keypoints_dict[name] = (float(kx), float(ky), float(kconf))

            # Biomechanical posture assessment
            fall_suspected, torso_angle, pose_conf = self.analyze_posture(
                keypoints_dict, workers[best_pid]["bbox"]
            )

            pose_info = {
                "keypoints": keypoints_dict,
                "torso_angle_deg": round(torso_angle, 1),
                "fall_suspected": fall_suspected,
                "confidence": round(pose_conf, 3)
            }

            new_cached[best_pid] = pose_info
            workers[best_pid]["pose_data"] = pose_info

        self.cached_poses = new_cached
        return self.cached_poses

    def analyze_posture(
        self,
        keypoints: Dict[str, Tuple[float, float, float]],
        bbox: List[float]
    ) -> Tuple[bool, float, float]:
        """
        Evaluate body posture geometry:
        1. Torso tilt angle relative to horizontal ground.
        2. Aspect ratio of bounding box (width vs height).
        3. Vertical position of head relative to hips and ground.

        Returns:
            Tuple of (fall_suspected: bool, torso_angle_deg: float, confidence: float).
        """
        l_sh = keypoints.get("left_shoulder", (0, 0, 0))
        r_sh = keypoints.get("right_shoulder", (0, 0, 0))
        l_hip = keypoints.get("left_hip", (0, 0, 0))
        r_hip = keypoints.get("right_hip", (0, 0, 0))

        # Check keypoint reliability
        sh_conf = max(l_sh[2], r_sh[2])
        hip_conf = max(l_hip[2], r_hip[2])
        if sh_conf < 0.25 or hip_conf < 0.25:
            # Fallback to bbox aspect ratio
            w = max(1.0, bbox[2] - bbox[0])
            h = max(1.0, bbox[3] - bbox[1])
            aspect_ratio = w / h  # In standing humans, width/height is ~0.3 - 0.5. In falls, ratio > 1.2
            if aspect_ratio >= 1.4:
                return True, 15.0, 0.65
            return False, 85.0, 0.40

        # Mid-shoulder and mid-hip coordinates
        sh_mid_x = (l_sh[0] + r_sh[0]) / 2.0
        sh_mid_y = (l_sh[1] + r_sh[1]) / 2.0
        hip_mid_x = (l_hip[0] + r_hip[0]) / 2.0
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2.0

        dx = hip_mid_x - sh_mid_x
        dy = hip_mid_y - sh_mid_y

        # Angle of torso vector with horizontal (in degrees)
        # 90 degrees = strictly upright; 0 degrees = completely horizontal
        torso_angle_rad = abs(math.atan2(dy, dx))
        torso_angle_deg = math.degrees(torso_angle_rad)
        if torso_angle_deg > 90:
            torso_angle_deg = 180 - torso_angle_deg

        # Worker bbox aspect ratio
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        aspect_ratio = w / h

        # A fall or prone condition occurs when torso is nearly horizontal
        # AND aspect ratio collapses (person is wide rather than tall)
        fall_suspected = False
        confidence = float(min(sh_conf, hip_conf))

        if torso_angle_deg < 35.0 and aspect_ratio > 0.95:
            fall_suspected = True
            confidence = min(0.95, confidence + 0.2)
        elif torso_angle_deg < 25.0:
            fall_suspected = True

        return fall_suspected, torso_angle_deg, confidence
