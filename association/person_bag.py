"""Person-Bag Association module.

Associates bags with individuals based on spatial proximity, bounding box overlap,
wrist interaction, and temporal persistence.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PersonBagAssociation:
    """Represents a computed association between a bag and a person."""
    person_id: Optional[int]
    bag_id: int
    distance: float
    confidence: float
    association_duration: float  # In seconds or frame count
    frame_count: int = 0
    is_stable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class PersonBagAssociator:
    """
    Associates detected bags with corresponding persons.
    Uses spatial proximity, box overlap, optional wrist keypoints, and temporal smoothing
    to prevent flickering and spurious association switches.
    """

    def __init__(
        self,
        max_distance: float = 250.0,
        overlap_weight: float = 0.4,
        distance_weight: float = 0.6,
        temporal_smooth_frames: int = 5,
        max_lost_frames: int = 30,
        fps: float = 30.0,
    ) -> None:
        self.max_distance = max_distance
        self.overlap_weight = overlap_weight
        self.distance_weight = distance_weight
        self.temporal_smooth_frames = temporal_smooth_frames
        self.max_lost_frames = max_lost_frames
        self.fps = max(fps, 1.0)

        # Internal state tracking: bag_id -> association tracking dict
        # {
        #   "person_id": int | None,
        #   "candidate_person_id": int | None,
        #   "candidate_frames": int,
        #   "frames_associated": int,
        #   "start_time": float,
        #   "last_distance": float,
        #   "last_conf": float,
        #   "lost_frames": int,
        # }
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def _compute_box_overlap(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute intersection over smaller box area (IoU-like overlap)."""
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))

        if x2 < x1 or y2 < y1:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)
        area1 = max(1.0, (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1])))
        area2 = max(1.0, (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1])))

        # Overlap relative to the bag (smaller object)
        min_area = min(area1, area2)
        return min(1.0, inter_area / min_area)

    def _compute_wrist_closeness(
        self,
        bag_center: Tuple[float, float],
        wrists: Optional[List[Dict[str, np.ndarray]]]
    ) -> float:
        """Return a bonus score [0.0, 0.3] if any wrist is physically close to bag."""
        if not wrists:
            return 0.0

        min_wrist_dist = float("inf")
        for hand in wrists:
            for side in ("left", "right"):
                pt = hand.get(side)
                if pt is not None and len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:
                    d = math.dist(bag_center, (float(pt[0]), float(pt[1])))
                    if d < min_wrist_dist:
                        min_wrist_dist = d

        if min_wrist_dist < 80.0:
            return 0.3
        elif min_wrist_dist < 150.0:
            return 0.15
        return 0.0

    def update(
        self,
        persons: List[Dict[str, Any]],
        bags: List[Dict[str, Any]],
        wrists: Optional[List[Dict[str, np.ndarray]]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[int, PersonBagAssociation]:
        """
        Compute associations between detected persons and bags.

        Args:
            persons: List of dicts with 'id' and 'box' [x1, y1, x2, y2]
            bags: List of dicts with 'id' and 'box' [x1, y1, x2, y2]
            wrists: Optional list of wrist coordinates
            timestamp: Optional monotonic timestamp

        Returns:
            Dict mapping bag_id -> PersonBagAssociation
        """
        curr_time = timestamp if timestamp is not None else time.time()
        results: Dict[int, PersonBagAssociation] = {}

        # Mark unseen bags
        active_bag_ids = {int(b["id"]) for b in bags if "id" in b}
        for b_id in list(self._tracks.keys()):
            if b_id not in active_bag_ids:
                self._tracks[b_id]["lost_frames"] += 1
                if self._tracks[b_id]["lost_frames"] > self.max_lost_frames:
                    del self._tracks[b_id]

        for bag in bags:
            b_id = int(bag["id"])
            b_box = np.array(bag["box"], dtype=float)
            bcx = (b_box[0] + b_box[2]) / 2.0
            bcy = (b_box[1] + b_box[3]) / 2.0
            bag_center = (bcx, bcy)

            # Evaluate each candidate person
            best_person_id: Optional[int] = None
            best_score = -1.0
            best_dist = float("inf")
            best_overlap = 0.0

            for person in persons:
                p_id = int(person["id"])
                p_box = np.array(person["box"], dtype=float)
                pcx = (p_box[0] + p_box[2]) / 2.0
                pcy = (p_box[1] + p_box[3]) / 2.0

                dist = math.dist(bag_center, (pcx, pcy))

                if dist > self.max_distance:
                    continue

                # Normalized proximity [0.0, 1.0]
                norm_dist_score = max(0.0, 1.0 - (dist / self.max_distance))

                # Spatial overlap score [0.0, 1.0]
                overlap = self._compute_box_overlap(b_box, p_box)

                # Wrist proximity bonus [0.0, 0.3]
                wrist_bonus = self._compute_wrist_closeness(bag_center, wrists)

                # Combined affinity score
                score = (
                    norm_dist_score * self.distance_weight
                    + overlap * self.overlap_weight
                    + wrist_bonus
                )
                score = min(1.0, score)

                if score > best_score:
                    best_score = score
                    best_person_id = p_id
                    best_dist = dist
                    best_overlap = overlap

            # Temporal smoothing logic
            if b_id not in self._tracks:
                self._tracks[b_id] = {
                    "person_id": best_person_id,
                    "candidate_person_id": best_person_id,
                    "candidate_frames": 1 if best_person_id is not None else 0,
                    "frames_associated": 1 if best_person_id is not None else 0,
                    "start_time": curr_time,
                    "last_distance": best_dist if best_person_id is not None else 0.0,
                    "last_conf": best_score if best_person_id is not None else 0.0,
                    "lost_frames": 0,
                }
            else:
                track = self._tracks[b_id]
                track["lost_frames"] = 0

                curr_p_id = track["person_id"]

                if best_person_id == curr_p_id and curr_p_id is not None:
                    # Same person continues association
                    track["frames_associated"] += 1
                    track["candidate_frames"] = 0
                    track["candidate_person_id"] = None
                    track["last_distance"] = best_dist
                    track["last_conf"] = best_score
                elif best_person_id != curr_p_id:
                    # New candidate detected; debounce before switching
                    if best_person_id == track["candidate_person_id"]:
                        track["candidate_frames"] += 1
                    else:
                        track["candidate_person_id"] = best_person_id
                        track["candidate_frames"] = 1

                    # Switch only after debouncing window satisfied
                    if track["candidate_frames"] >= self.temporal_smooth_frames:
                        track["person_id"] = best_person_id
                        track["frames_associated"] = track["candidate_frames"]
                        track["start_time"] = curr_time
                        track["candidate_person_id"] = None
                        track["candidate_frames"] = 0
                        track["last_distance"] = best_dist
                        track["last_conf"] = best_score
                    else:
                        # Retain current person temporarily
                        pass

            track = self._tracks[b_id]
            assigned_pid = track["person_id"]
            frames_assoc = track["frames_associated"]
            duration_sec = frames_assoc / self.fps
            is_stable = frames_assoc >= self.temporal_smooth_frames

            # Fallback distance & confidence
            display_dist = track["last_distance"] if assigned_pid is not None else 0.0
            display_conf = track["last_conf"] if assigned_pid is not None else 0.0

            results[b_id] = PersonBagAssociation(
                person_id=assigned_pid,
                bag_id=b_id,
                distance=round(display_dist, 1),
                confidence=round(display_conf, 2),
                association_duration=round(duration_sec, 2),
                frame_count=frames_assoc,
                is_stable=is_stable,
                details={"overlap": round(best_overlap, 2)},
            )

        return results
