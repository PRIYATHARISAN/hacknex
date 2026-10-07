"""
Attribution & Audit Logging Module.
Enforces the hackathon rule:
"Every flag for unusual behavior must point to which entity and when.
'Something weird happened' doesn't count — say who and when."
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional


def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    """Converts a frame index and FPS into a formatted timestamp HH:MM:SS.mmm."""
    if fps <= 0:
        fps = 30.0
    total_seconds = frame_idx / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


@dataclass
class AnomalyEvent:
    event_id: str
    entity_id: str
    action_type: str
    status: str
    start_time: str
    end_time: str
    start_frame: int
    end_frame: int
    duration_seconds: float
    reason: str
    latest_bbox: List[int]
    centroid: List[int]

    def to_dict(self) -> Dict:
        return asdict(self)


class EventLogger:
    """
    Manages active and finalized anomaly incidents.
    Binds every alert to Entity ID, Start Time, and End Time.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.active_events: Dict[str, Dict] = {}  # key: (entity_id, action_type)
        self.finalized_events: List[AnomalyEvent] = []
        self.all_seen_entities = set()
        self._event_counter = 0

    def register_seen_entity(self, entity_id: str):
        """Records an entity ID observed anywhere in the footage."""
        self.all_seen_entities.add(entity_id)

    def trigger_or_update(
        self,
        entity_id: str,
        action_type: str,
        frame_idx: int,
        fps: float,
        reason: str,
        bbox: List[int],
        centroid: List[int],
    ):
        """Called each frame an entity is in an abnormal state."""
        key = f"{entity_id}_{action_type}"
        now_ts = frame_to_timestamp(frame_idx, fps)

        if key not in self.active_events:
            self._event_counter += 1
            event_id = f"EVT_{self._event_counter:04d}"
            self.active_events[key] = {
                "event_id": event_id,
                "entity_id": entity_id,
                "action_type": action_type,
                "status": "ABNORMAL",
                "start_time": now_ts,
                "end_time": now_ts,
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "duration_seconds": 0.0,
                "reason": reason,
                "latest_bbox": bbox,
                "centroid": centroid,
            }
        else:
            evt = self.active_events[key]
            evt["end_time"] = now_ts
            evt["end_frame"] = frame_idx
            evt["duration_seconds"] = round((frame_idx - evt["start_frame"]) / max(fps, 1.0), 2)
            evt["latest_bbox"] = bbox
            evt["centroid"] = centroid
            evt["reason"] = reason

    def finalize_if_resolved(self, entity_id: str, active_actions: List[str], current_frame: int, fps: float):
        """Closes events that are no longer active for the given entity."""
        keys_to_close = []
        for key, evt in self.active_events.items():
            if evt["entity_id"] == entity_id and evt["action_type"] not in active_actions:
                keys_to_close.append(key)

        for key in keys_to_close:
            raw = self.active_events.pop(key)
            finalized = AnomalyEvent(**raw)
            self.finalized_events.append(finalized)

    def close_all(self):
        """Finalizes any remaining active events (e.g. at the end of video processing)."""
        for key in list(self.active_events.keys()):
            raw = self.active_events.pop(key)
            self.finalized_events.append(AnomalyEvent(**raw))

    def get_all_events(self) -> List[Dict]:
        """Returns all completed and currently active events."""
        all_evts = [e.to_dict() for e in self.finalized_events]
        for evt in self.active_events.values():
            all_evts.append(dict(evt))
        return all_evts

    def save_to_json(self):
        """Writes all events to the configured JSON file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_incidents": len(self.finalized_events) + len(self.active_events),
            "total_tracked_entities": len(self.all_seen_entities),
            "events": self.get_all_events(),
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
