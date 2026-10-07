"""
Tracker Module.
Wraps YOLO26m-pose model with ByteTrack to perform unified detection,
persistent ID assignment, and skeletal keypoint extraction in a single pass.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from ultralytics import YOLO
import config



class TrackedEntity:
    def __init__(
        self,
        track_id: str,
        bbox: List[int],
        confidence: float,
        keypoints: Optional[np.ndarray],
    ):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.keypoints = keypoints  # shape (17, 2 or 3)


class PoseTracker:
    """
    Unified detector, tracker, and pose estimator using yolo26m-pose and ByteTrack.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
        conf_thresh: float = config.CONFIDENCE_THRESHOLD,
        iou_thresh: float = config.IOU_THRESHOLD,
        tracker_config: str = config.TRACKER_CONFIG,
    ):
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.tracker_config = tracker_config

        # Check if local model file exists in models/ directory, else use model_name directly
        local_model = config.MODELS_DIR / model_name
        target_path = str(local_model) if local_model.exists() else model_name

        print(f"[Tracker] Loading pose model '{target_path}' on device '{self.device}'...")
        self.model = YOLO(target_path)
        print("[Tracker] Model loaded successfully.")

    def track_frame(self, frame: np.ndarray) -> List[TrackedEntity]:
        """
        Processes a single BGR video frame.
        Returns a list of TrackedEntity objects with persistent IDs and keypoints.
        """
        # Run tracking (classes=[0] for person class in COCO)
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=[0],
            device=self.device,
            verbose=False,
        )

        tracked_entities: List[TrackedEntity] = []

        if not results or len(results) == 0:
            return tracked_entities

        result = results[0]
        boxes = result.boxes
        keypoints_obj = result.keypoints

        if boxes is None or len(boxes) == 0:
            return tracked_entities

        # Extract box coordinates, track IDs, confidences
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None

        # Extract keypoints coordinates (N, 17, 2 or 3)
        kpts_data = None
        if keypoints_obj is not None and keypoints_obj.data is not None:
            kpts_data = keypoints_obj.data.cpu().numpy()

        for i, box in enumerate(xyxy):
            # If tracker hasn't assigned an ID yet, skip or generate a temporary ID
            raw_id = int(ids[i]) if ids is not None else (i + 1)
            formatted_id = f"Person #{raw_id:02d}"

            kpts = kpts_data[i] if kpts_data is not None and i < len(kpts_data) else None

            tracked_entities.append(
                TrackedEntity(
                    track_id=formatted_id,
                    bbox=[int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    confidence=float(confs[i]),
                    keypoints=kpts,
                )
            )

        return tracked_entities
