"""
Workplace Safety AI - Personal Protective Equipment (PPE) Detection & Worker Association
"""

from __future__ import annotations
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.utils import calculate_overlap_ratio, calculate_iou


class PPEManager:
    """
    Manages detection and worker association for safety helmets and high-visibility vests.
    Associates PPE detections with individual worker bounding boxes.
    """

    PPE_HELMET_CLASSES = {"helmet", "hard_hat", "hard-hat", "safety_helmet", "head_protection"}
    PPE_VEST_CLASSES = {"vest", "safety_vest", "high_viz_vest", "reflective_vest", "vest_protection"}

    def __init__(
        self,
        model_path: str = "models/workplace_ppe.pt",
        confidence: float = 0.40,
        head_region_ratio: float = 0.25,
        torso_region_ratio: Tuple[float, float] = (0.20, 0.65),
        association_iou_thresh: float = 0.20,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PPE Manager.

        Args:
            model_path: Path to workplace PPE custom weights.
            confidence: Detection confidence for PPE items.
            head_region_ratio: Fraction of person box from top representing head.
            torso_region_ratio: Tuple (start_ratio, end_ratio) for upper body torso.
            association_iou_thresh: Threshold for spatial overlap association.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("WorkplaceSafetyAI.PPEManager")
        self.model_path = model_path
        self.confidence = float(confidence)
        self.head_region_ratio = float(head_region_ratio)
        self.torso_start_ratio, self.torso_end_ratio = torso_region_ratio
        self.association_iou_thresh = float(association_iou_thresh)

        self.is_available = False
        self.model = self._load_ppe_model(model_path)

    def _load_ppe_model(self, path: str) -> Any:
        """
        Load custom PPE model weights.
        If file does not exist, clearly output the required warning and continue safely.
        """
        if not os.path.isfile(path):
            print("\n" + "=" * 50)
            print("WARNING:")
            print("Custom PPE model not found.")
            print(f"File '{path}' does not exist.")
            print("PPE detection disabled.")
            print("=" * 50 + "\n")
            self.logger.warning("Custom PPE model not found at '%s'. PPE detection disabled.", path)
            self.is_available = False
            return None

        if YOLO is None:
            self.logger.warning("Ultralytics library unavailable. PPE detection disabled.")
            self.is_available = False
            return None

        try:
            self.logger.info("Loading custom workplace PPE model from: %s", path)
            model = YOLO(path)
            self.is_available = True
            return model
        except Exception as exc:
            self.logger.error("Failed to initialize PPE model at '%s': %s", path, exc)
            self.is_available = False
            return None

    def detect_ppe(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference with PPE model on input frame.

        Returns:
            List of detected PPE objects:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_name": "helmet" | "vest",
                    "confidence": float
                }
            ]
        """
        if not self.is_available or self.model is None or frame is None:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                verbose=False
            )
        except Exception as exc:
            self.logger.error("Error executing PPE inference: %s", exc)
            return []

        detections = []
        if not results or len(results) == 0 or results[0].boxes is None:
            return detections

        first_res = results[0]
        boxes = first_res.boxes.xyxy.cpu().numpy()
        confs = first_res.boxes.conf.cpu().numpy()
        cls_ids = first_res.boxes.cls.cpu().numpy().astype(int)
        names = first_res.names if hasattr(first_res, "names") else {}

        for box, conf, cid in zip(boxes, confs, cls_ids):
            cname = str(names.get(cid, cid)).lower()
            detections.append({
                "bbox": [float(c) for c in box],
                "class_name": cname,
                "confidence": float(conf)
            })

        return detections

    def associate_ppe(
        self,
        workers: Dict[int, Dict[str, Any]],
        ppe_detections: List[Dict[str, Any]]
    ) -> None:
        """
        Associate detected PPE equipment with specific workers based on spatial anatomy.
        Modifies worker dictionary in-place:
            worker["helmet_present"] = True / False / None
            worker["vest_present"] = True / False / None
        """
        if not self.is_available and not ppe_detections:
            # When PPE model is not present, mark PPE status as None (unmonitored)
            for worker in workers.values():
                worker["helmet_present"] = None
                worker["vest_present"] = None
            return

        # Segregate detected PPE items
        helmets = []
        vests = []
        for item in ppe_detections:
            cname = item["class_name"].lower()
            if any(term in cname for term in ("helmet", "hat", "hard")):
                helmets.append(item)
            elif any(term in cname for term in ("vest", "jacket", "high_viz")):
                vests.append(item)

        # For each worker, evaluate head and torso spatial overlap
        for worker in workers.values():
            wx1, wy1, wx2, wy2 = worker["bbox"]
            w_h = wy2 - wy1

            # Anatomical head region: top 25% of worker bounding box
            head_box = [wx1, wy1, wx2, wy1 + self.head_region_ratio * w_h]

            # Anatomical torso region: upper-middle 20% - 65% of worker bounding box
            torso_box = [wx1, wy1 + self.torso_start_ratio * w_h, wx2, wy1 + self.torso_end_ratio * w_h]

            # Test helmet association
            has_helmet = False
            for h in helmets:
                h_box = h["bbox"]
                # Check center of helmet box inside head box
                h_cx = (h_box[0] + h_box[2]) / 2.0
                h_cy = (h_box[1] + h_box[3]) / 2.0
                center_inside = (head_box[0] <= h_cx <= head_box[2]) and (head_box[1] <= h_cy <= head_box[3])
                overlap = calculate_overlap_ratio(h_box, head_box)

                if center_inside or overlap >= self.association_iou_thresh:
                    has_helmet = True
                    break

            # Test vest association
            has_vest = False
            for v in vests:
                v_box = v["bbox"]
                v_cx = (v_box[0] + v_box[2]) / 2.0
                v_cy = (v_box[1] + v_box[3]) / 2.0
                center_inside = (torso_box[0] <= v_cx <= torso_box[2]) and (torso_box[1] <= v_cy <= torso_box[3])
                overlap = calculate_overlap_ratio(v_box, torso_box)

                if center_inside or overlap >= self.association_iou_thresh:
                    has_vest = True
                    break

            worker["helmet_present"] = has_helmet
            worker["vest_present"] = has_vest
