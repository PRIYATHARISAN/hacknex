"""
Workplace Safety AI - Person & Object Detection Module
"""

from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    torch = None
    YOLO = None


class ObjectDetector:
    """
    Object detector utilizing Ultralytics YOLO models (e.g. YOLO11m)
    for high-precision detection of persons and workplace assets.
    """

    def __init__(
        self,
        model_path: str = "models/yolo11m.pt",
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "auto",
        target_classes: Optional[List[int]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO weights (.pt) or standard model identifier.
            confidence: Detection confidence threshold [0.0 - 1.0].
            image_size: Input inference resolution (e.g. 640).
            device: 'cuda', 'cpu', or 'auto'.
            target_classes: List of class IDs to filter (e.g. [0] for persons). None = all classes.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("WorkplaceSafetyAI.Detector")
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.target_classes = target_classes if target_classes is not None else [0]  # default: person (0)
        self.device = self._resolve_device(device)
        self.model_path = model_path
        self.model = self._load_model(model_path)

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve execution device based on hardware availability."""
        if requested_device == "cpu":
            return "cpu"

        if torch is not None and torch.cuda.is_available():
            if requested_device in ("auto", "cuda", "0"):
                self.logger.info("CUDA GPU detected. Using acceleration device: %s", torch.cuda.get_device_name(0))
                return "0"

        if requested_device not in ("auto", "cpu"):
            self.logger.warning("Requested device '%s' unavailable. Defaulting to CPU.", requested_device)

        self.logger.info("Running detector inference on CPU.")
        return "cpu"

    def _load_model(self, path: str) -> Any:
        """Load YOLO model weights with multi-stage fallback resolution."""
        if YOLO is None:
            self.logger.warning("Ultralytics YOLO package is not installed. Detector initialized in dry-run mode.")
            return None

        # Check explicit path
        if os.path.isfile(path):
            self.logger.info("Loading object detector model from: %s", path)
            return YOLO(path)

        # Fallback checks
        fallbacks = [
            os.path.join("models", os.path.basename(path)),
            os.path.join("models", "yolo11m.pt"),
            os.path.join("models", "yolo11n.pt"),
            "yolo11m.pt",
            "yolo11n.pt"
        ]

        for candidate in fallbacks:
            if os.path.isfile(candidate):
                self.logger.warning(
                    "Model not found at '%s'. Falling back to local candidate: %s", path, candidate
                )
                return YOLO(candidate)

        # If still not found locally, pass the name to YOLO to allow automated download
        base_name = os.path.basename(path)
        self.logger.info("Attempting online resolution/download for model: %s", base_name)
        try:
            return YOLO(base_name)
        except Exception as exc:
            self.logger.error("Failed to load YOLO model: %s", exc)
            raise FileNotFoundError(f"Could not load detector model from {path} or fallbacks: {exc}") from exc

    def detect(self, frame: np.ndarray, person_only: bool = True) -> List[Dict[str, Any]]:
        """
        Run inference on an input BGR image.

        Args:
            frame: BGR numpy image from OpenCV.
            person_only: If True, filters strictly for person class (ID 0).

        Returns:
            List of structured detections:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int,
                    "class_name": str,
                    "confidence": float
                },
                ...
            ]
        """
        if self.model is None or frame is None:
            return []

        classes_filter = [0] if person_only else self.target_classes

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                classes=classes_filter,
                verbose=False
            )
        except Exception as err:
            self.logger.error("Inference failure in ObjectDetector: %s", err)
            return []

        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes = first_res.boxes.xyxy.cpu().numpy()
        confs = first_res.boxes.conf.cpu().numpy()
        cls_ids = first_res.boxes.cls.cpu().numpy().astype(int)
        names = first_res.names if hasattr(first_res, "names") else {}

        for box, conf, cid in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = [float(coord) for coord in box]
            cname = names.get(cid, str(cid))
            detections.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "class_id": int(cid),
                "class_name": cname,
                "confidence": round(float(conf), 3)
            })

        return detections
