"""
Workplace Safety AI - Multi-Object Person Tracking with Persistent IDs
"""

from __future__ import annotations
import logging
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.utils import calculate_iou


class WorkerTracker:
    """
    Tracks workers across video frames using ByteTrack algorithm.
    Maintains persistent worker state dictionaries and spatial coordinates.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tracker_config: str = "bytetrack.yaml",
        confidence: float = 0.35,
        iou_threshold: float = 0.5,
        max_missing_frames: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the tracker.

        Args:
            model: Instantiated Ultralytics YOLO model for tracked inference.
            tracker_config: Tracking configuration (default: 'bytetrack.yaml').
            confidence: Confidence threshold for tracking.
            iou_threshold: IoU threshold for matching.
            max_missing_frames: Frames before pruning an unseen track.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("WorkplaceSafetyAI.Tracker")
        self.model = model
        self.tracker_config = tracker_config
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.max_missing_frames = int(max_missing_frames)

        # Worker state registry: person_id -> worker_dict
        self.workers: Dict[int, Dict[str, Any]] = {}

        # Tracking auxiliary state
        self._frame_count = 0
        self._next_fallback_id = 1
        self._missing_counts: Dict[int, int] = {}

    def update(
        self,
        frame: np.ndarray,
        current_time_sec: float,
        pre_detections: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Update tracking state for the current video frame.

        Args:
            frame: Current BGR video frame.
            current_time_sec: Current video timestamp in seconds.
            pre_detections: Optional detections list (used for offline or fallback tracking).

        Returns:
            Dictionary of active workers:
            {
                person_id: {
                    "person_id": int,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "foot_point": [fx, fy],
                    "last_seen": float,
                    "first_seen": float,
                    "inside_restricted_zone": bool,
                    "helmet_present": Optional[bool],
                    "vest_present": Optional[bool],
                    "behavior_state": str,
                    "trajectory": deque
                }
            }
        """
        self._frame_count += 1
        current_frame_tracked: List[Tuple[int, List[float], float]] = []

        # Strategy 1: Ultralytics built-in ByteTrack if model is present and frame provided
        if self.model is not None and frame is not None:
            try:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    classes=[0],  # Person class
                    verbose=False
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    res_boxes = results[0].boxes
                    if res_boxes.id is not None:
                        track_ids = res_boxes.id.int().cpu().tolist()
                        boxes = res_boxes.xyxy.cpu().tolist()
                        confs = res_boxes.conf.cpu().tolist()

                        for tid, box, conf in zip(track_ids, boxes, confs):
                            current_frame_tracked.append((int(tid), [float(c) for c in box], float(conf)))
            except Exception as err:
                self.logger.warning("ByteTrack inference exception: %s. Switching to fallback matching.", err)

        # Strategy 2: If model track returned nothing or in unit test mode with pre_detections
        if not current_frame_tracked and pre_detections is not None:
            current_frame_tracked = self._fallback_match(pre_detections)

        # Update active worker states
        active_ids = set()
        for person_id, bbox, conf in current_frame_tracked:
            active_ids.add(person_id)
            self._missing_counts[person_id] = 0

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            fx = cx
            fy = y2  # Bottom-center represents ground foot point

            if person_id not in self.workers:
                # Initialize new persistent worker record
                self.workers[person_id] = {
                    "person_id": person_id,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "center": [round(cx, 1), round(cy, 1)],
                    "foot_point": [round(fx, 1), round(fy, 1)],
                    "first_seen": current_time_sec,
                    "last_seen": current_time_sec,
                    "confidence": round(conf, 3),
                    "inside_restricted_zone": False,
                    "current_zone_id": None,
                    "zone_state": "OUTSIDE",
                    "helmet_present": None,
                    "vest_present": None,
                    "behavior_state": "NORMAL",
                    "active_violations": [],
                    "trajectory": deque(maxlen=40)
                }
                self.logger.info(
                    "New track registered: Person #%d at [%.1f, %.1f] (t=%.2fs)",
                    person_id, cx, cy, current_time_sec
                )
            else:
                worker = self.workers[person_id]
                worker["bbox"] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                worker["center"] = [round(cx, 1), round(cy, 1)]
                worker["foot_point"] = [round(fx, 1), round(fy, 1)]
                worker["last_seen"] = current_time_sec
                worker["confidence"] = round(conf, 3)

            self.workers[person_id]["trajectory"].append((round(fx, 1), round(fy, 1)))

        # Age and purge tracks lost for too many frames
        for pid in list(self.workers.keys()):
            if pid not in active_ids:
                self._missing_counts[pid] = self._missing_counts.get(pid, 0) + 1
                if self._missing_counts[pid] > self.max_missing_frames:
                    self.logger.debug("Pruning stale track: Person #%d", pid)
                    del self.workers[pid]
                    if pid in self._missing_counts:
                        del self._missing_counts[pid]

        return self.workers

    def _fallback_match(self, detections: List[Dict[str, Any]]) -> List[Tuple[int, List[float], float]]:
        """
        Lightweight greedy IoU matching for testing or fallback when ByteTrack is uninitialized.
        """
        matched: List[Tuple[int, List[float], float]] = []
        unmatched_dets = []

        existing_ids = list(self.workers.keys())
        used_ids = set()

        for det in detections:
            bbox = det["bbox"]
            conf = det.get("confidence", 1.0)
            best_id = None
            best_iou = 0.0

            for pid in existing_ids:
                if pid in used_ids:
                    continue
                prev_bbox = self.workers[pid]["bbox"]
                iou = calculate_iou(bbox, prev_bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_id = pid

            if best_id is not None:
                used_ids.add(best_id)
                matched.append((best_id, bbox, conf))
            else:
                unmatched_dets.append((bbox, conf))

        # Assign new persistent IDs to unmatched detections
        for bbox, conf in unmatched_dets:
            new_id = self._next_fallback_id
            self._next_fallback_id += 1
            matched.append((new_id, bbox, conf))

        return matched

    def get_worker(self, person_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve individual worker state dict."""
        return self.workers.get(person_id)

    def get_all_workers(self) -> Dict[int, Dict[str, Any]]:
        """Return reference to all active workers."""
        return self.workers
