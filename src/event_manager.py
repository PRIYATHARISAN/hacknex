"""
Workplace Safety AI - Safety Event Management, Deduplication & Export
"""

import os
import csv
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from src.utils import seconds_to_timestamp


class EventManager:
    """
    Manages safety event lifecycles: creation, deduplication, duration updating,
    cooldown hysteresis, and persistence to JSON/CSV formats.
    """

    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        output_dir: str = "output/events",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Event Manager.

        Args:
            cooldown_seconds: Minimum seconds before a resolved violation can trigger a new event.
            output_dir: Directory where JSON and CSV reports are written.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("WorkplaceSafetyAI.EventManager")
        self.cooldown_seconds = float(cooldown_seconds)
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Active events indexed by (person_id, event_type)
        self.active_events: Dict[Tuple[int, str], Dict[str, Any]] = {}
        # Historical list of all closed events
        self.closed_events: List[Dict[str, Any]] = []
        # Cooldown registry: (person_id, event_type) -> last_closed_timestamp
        self._cooldown_registry: Dict[Tuple[int, str], float] = {}
        self._next_event_id = 1

    def process_violations(
        self,
        violations: List[Dict[str, Any]],
        current_time_sec: float
    ) -> None:
        """
        Ingest frame violations, update or create active events, and close resolved ones.

        Args:
            violations: List of violation dicts from BehaviorEngine.
            current_time_sec: Current video time in seconds.
        """
        active_keys_this_frame = set()

        for v in violations:
            pid = v["person_id"]
            etype = v["violation_type"]
            key = (pid, etype)
            active_keys_this_frame.add(key)

            if key in self.active_events:
                # Event continues -> update duration and end time
                self.update_event(key, current_time_sec, v)
            else:
                # Check cooldown to avoid duplicate flapping
                last_closed = self._cooldown_registry.get(key, -999.0)
                if (current_time_sec - last_closed) >= self.cooldown_seconds:
                    self.create_event(key, current_time_sec, v)

        # Detect active events that have now ceased/resolved
        active_keys = list(self.active_events.keys())
        for key in active_keys:
            if key not in active_keys_this_frame:
                self.close_event(key, current_time_sec)

    def create_event(
        self,
        key: Tuple[int, str],
        current_time_sec: float,
        violation_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create and register a new persistent safety event."""
        person_id, event_type = key
        event_id = f"EVT-{self._next_event_id:04d}"
        self._next_event_id += 1

        formatted_time = seconds_to_timestamp(current_time_sec)
        severity = violation_meta.get("severity", "MEDIUM")
        confidence = float(violation_meta.get("confidence", 0.90))
        zone = violation_meta.get("zone_id") or "restricted_zone"
        description = violation_meta.get("description") or f"Person #{person_id} detected with {event_type} in {zone}"

        event = {
            "event_id": event_id,
            "event_type": event_type,
            "person_id": person_id,
            "timestamp": formatted_time,
            "duration": 0.0,
            "confidence": round(confidence, 2),
            "severity": severity,
            "zone": zone,
            "description": description,
            "start_time": round(current_time_sec, 2),
            "end_time": round(current_time_sec, 2),
            "status": "ACTIVE"
        }

        self.active_events[key] = event
        self.logger.warning(
            "🚨 SAFETY EVENT CREATED: [%s] Person #%d - %s (Severity: %s, Time: %s) | %s",
            event_id, person_id, event_type, severity, formatted_time, description
        )
        return event

    def update_event(
        self,
        key: Tuple[int, str],
        current_time_sec: float,
        violation_meta: Dict[str, Any]
    ) -> None:
        """Update an ongoing event with latest timestamp and cumulative duration."""
        event = self.active_events[key]
        event["end_time"] = round(current_time_sec, 2)
        event["duration"] = round(current_time_sec - event["start_time"], 1)
        if "confidence" in violation_meta:
            event["confidence"] = max(event["confidence"], round(float(violation_meta["confidence"]), 2))
        if violation_meta.get("description"):
            event["description"] = violation_meta["description"]

    def close_event(
        self,
        key: Tuple[int, str],
        current_time_sec: float
    ) -> Optional[Dict[str, Any]]:
        """Close an active event upon cessation and register cooldown."""
        if key not in self.active_events:
            return None

        event = self.active_events.pop(key)
        event["status"] = "CLOSED"
        event["end_time"] = round(current_time_sec, 2)
        event["duration"] = round(current_time_sec - event["start_time"], 1)

        self._cooldown_registry[key] = current_time_sec
        self.closed_events.append(event)

        self.logger.info(
            "RESOLVED EVENT: [%s] Person #%d - %s ended at %s (Duration: %.1fs)",
            event["event_id"], event["person_id"], event["event_type"],
            seconds_to_timestamp(current_time_sec), event["duration"]
        )

        # Register dedicated RESTRICTED_ZONE_EXIT event upon zone exit
        if event["event_type"] in ("RESTRICTED_ZONE_ENTRY", "RESTRICTED_ZONE_VIOLATION"):
            exit_id = f"EVT-{self._next_event_id:04d}"
            self._next_event_id += 1
            exit_event = {
                "event_id": exit_id,
                "event_type": "RESTRICTED_ZONE_EXIT",
                "person_id": event["person_id"],
                "timestamp": seconds_to_timestamp(current_time_sec),
                "duration": event["duration"],
                "confidence": event["confidence"],
                "severity": "LOW",
                "zone": event["zone"],
                "description": f"Person #{event['person_id']} exited {event['zone']} after {event['duration']}s",
                "start_time": round(current_time_sec, 2),
                "end_time": round(current_time_sec, 2),
                "status": "CLOSED"
            }
            self.closed_events.append(exit_event)

        return event

    def close_all(self, final_time_sec: float) -> None:
        """Close any remaining active events when video stream terminates."""
        for key in list(self.active_events.keys()):
            self.close_event(key, final_time_sec)

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Return combined list of active and closed events."""
        return self.closed_events + list(self.active_events.values())

    def get_recent_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return the most recent active and closed events for dashboard display."""
        all_events = self.get_all_events()
        sorted_events = sorted(all_events, key=lambda e: e.get("end_time", 0.0), reverse=True)
        return sorted_events[:limit]

    def save_events(
        self,
        json_filename: str = "events.json",
        csv_filename: str = "events.csv"
    ) -> Tuple[str, str]:
        """
        Export all recorded events to JSON and CSV logs in output_dir.

        Returns:
            Tuple of (json_path, csv_path).
        """
        all_events = self.get_all_events()

        # 1. JSON Export
        json_path = os.path.join(self.output_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(all_events, jf, indent=2)

        # Also write safety_events.json for backward compatibility
        compat_json = os.path.join(self.output_dir, "safety_events.json")
        if compat_json != json_path:
            with open(compat_json, "w", encoding="utf-8") as cjf:
                json.dump(all_events, cjf, indent=2)

        # 2. CSV Export
        csv_path = os.path.join(self.output_dir, csv_filename)
        fieldnames = [
            "event_id", "event_type", "person_id", "timestamp",
            "duration", "confidence", "severity", "zone", "description",
            "start_time", "end_time", "status"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for evt in all_events:
                writer.writerow(evt)

        # Also write safety_events.csv for backward compatibility
        compat_csv = os.path.join(self.output_dir, "safety_events.csv")
        if compat_csv != csv_path:
            with open(compat_csv, "w", newline="", encoding="utf-8") as ccf:
                cwriter = csv.DictWriter(ccf, fieldnames=fieldnames)
                cwriter.writeheader()
                for evt in all_events:
                    cwriter.writerow(evt)

        self.logger.info("Safety events persisted: %s and %s", json_path, csv_path)
        return json_path, csv_path
