"""
Unit tests for EventManager (event lifecycle, deduplication, cooldown, and storage)
"""

import os
import shutil
import tempfile
import unittest
import json
import csv

from src.event_manager import EventManager


class TestEventManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mgr = EventManager(cooldown_seconds=3.0, output_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_event_creation_and_fields(self):
        """Creating an event generates properly structured metadata."""
        violation = [{
            "person_id": 7,
            "violation_type": "RESTRICTED_ZONE_VIOLATION",
            "severity": "HIGH",
            "confidence": 0.94,
            "zone_id": "zone_dock",
            "timestamp": 42.3
        }]

        self.mgr.process_violations(violation, current_time_sec=42.3)
        self.assertEqual(len(self.mgr.active_events), 1)

        key = (7, "RESTRICTED_ZONE_VIOLATION")
        event = self.mgr.active_events[key]
        self.assertEqual(event["person_id"], 7)
        self.assertEqual(event["event_type"], "RESTRICTED_ZONE_VIOLATION")
        self.assertEqual(event["severity"], "HIGH")
        self.assertEqual(event["status"], "ACTIVE")
        self.assertEqual(event["timestamp"], "00:42.3")

    def test_deduplication_updates_duration_without_new_events(self):
        """Subsequent frames of the same violation must update duration, NOT create new events."""
        violation = [{
            "person_id": 7,
            "violation_type": "RESTRICTED_ZONE_VIOLATION",
            "severity": "HIGH",
            "confidence": 0.94,
            "zone_id": "zone_dock"
        }]

        # Frame 1: t=10.0s
        self.mgr.process_violations(violation, current_time_sec=10.0)
        self.assertEqual(len(self.mgr.active_events), 1)
        self.assertEqual(len(self.mgr.get_all_events()), 1)

        # Frame 2: t=11.0s
        self.mgr.process_violations(violation, current_time_sec=11.0)
        self.assertEqual(len(self.mgr.active_events), 1)
        self.assertEqual(len(self.mgr.get_all_events()), 1)

        # Frame 3: t=13.5s
        self.mgr.process_violations(violation, current_time_sec=13.5)
        self.assertEqual(len(self.mgr.active_events), 1)
        self.assertEqual(len(self.mgr.get_all_events()), 1)

        key = (7, "RESTRICTED_ZONE_VIOLATION")
        self.assertEqual(self.mgr.active_events[key]["duration"], 3.5)

    def test_event_closure_and_cooldown(self):
        """When violation ceases, event moves to closed with cooldown active."""
        violation = [{
            "person_id": 7,
            "violation_type": "NO_HELMET",
            "severity": "MEDIUM",
            "confidence": 0.91,
            "zone_id": "floor"
        }]

        # t=20.0s start
        self.mgr.process_violations(violation, current_time_sec=20.0)
        # t=24.0s update
        self.mgr.process_violations(violation, current_time_sec=24.0)
        # t=25.0s violation ceases (empty list)
        self.mgr.process_violations([], current_time_sec=25.0)

        self.assertEqual(len(self.mgr.active_events), 0)
        self.assertEqual(len(self.mgr.closed_events), 1)
        closed_evt = self.mgr.closed_events[0]
        self.assertEqual(closed_evt["status"], "CLOSED")
        self.assertEqual(closed_evt["duration"], 5.0)

        # Re-triggering during cooldown (at t=26.0s, delta 1s < 3s cooldown) should be suppressed
        self.mgr.process_violations(violation, current_time_sec=26.0)
        self.assertEqual(len(self.mgr.active_events), 0, "Event should be suppressed during cooldown")

        # After cooldown passes (at t=29.0s, delta 4s > 3s cooldown) a new event is permitted
        self.mgr.process_violations(violation, current_time_sec=29.0)
        self.assertEqual(len(self.mgr.active_events), 1)

    def test_save_events_json_and_csv(self):
        """Events are properly serialized into valid JSON and CSV format."""
        violation = [{
            "person_id": 4,
            "violation_type": "HIGH_RISK_SAFETY_VIOLATION",
            "severity": "CRITICAL",
            "confidence": 0.95,
            "zone_id": "zone_1"
        }]
        self.mgr.process_violations(violation, current_time_sec=15.0)
        self.mgr.close_all(final_time_sec=18.5)

        json_path, csv_path = self.mgr.save_events("test_events.json", "test_events.csv")
        self.assertTrue(os.path.isfile(json_path))
        self.assertTrue(os.path.isfile(csv_path))

        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["person_id"], 4)
            self.assertEqual(data[0]["severity"], "CRITICAL")

        with open(csv_path, "r", encoding="utf-8") as cf:
            reader = list(csv.DictReader(cf))
            self.assertEqual(len(reader), 1)
            self.assertEqual(reader[0]["person_id"], "4")
            self.assertEqual(reader[0]["severity"], "CRITICAL")


if __name__ == "__main__":
    unittest.main()
