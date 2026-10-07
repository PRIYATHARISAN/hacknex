"""
Unit tests for BehaviorEngine (temporal reasoning, persistence, and state transitions)
"""

import unittest
from src.behavior_engine import BehaviorEngine


class TestBehaviorEngine(unittest.TestCase):

    def setUp(self):
        self.engine = BehaviorEngine(
            restricted_zone_seconds=2.0,
            no_helmet_seconds=2.0,
            no_vest_seconds=2.0,
            fall_duration_seconds=1.5
        )

    def test_normal_behavior_generates_no_violations(self):
        """Worker compliant with PPE and outside zone is classified as NORMAL."""
        workers = {
            1: {
                "person_id": 1,
                "inside_restricted_zone": False,
                "helmet_present": True,
                "vest_present": True,
                "pose_data": None
            }
        }

        violations = self.engine.update(workers, current_time_sec=10.0)
        self.assertEqual(len(violations), 0)
        self.assertEqual(workers[1]["behavior_state"], "NORMAL_PPE")

    def test_restricted_zone_temporal_persistence(self):
        """Zone violation requires continuous residence exceeding duration threshold."""
        workers = {
            7: {
                "person_id": 7,
                "inside_restricted_zone": True,
                "current_zone_id": "zone_1",
                "helmet_present": True,
                "vest_present": True
            }
        }

        # Frame at t=0.0s (transient entry)
        v0 = self.engine.update(workers, current_time_sec=0.0)
        self.assertEqual(len(v0), 0, "No violation should trigger on frame 0")
        self.assertEqual(workers[7]["behavior_state"], "ENTERING_RESTRICTED_ZONE")

        # Frame at t=1.0s (insufficient duration: 1.0s < 2.0s)
        v1 = self.engine.update(workers, current_time_sec=1.0)
        self.assertEqual(len(v1), 0, "No violation should trigger before 2.0s")

        # Frame at t=2.2s (confirmed persistence: 2.2s >= 2.0s)
        v2 = self.engine.update(workers, current_time_sec=2.2)
        self.assertEqual(len(v2), 1)
        self.assertIn(v2[0]["violation_type"], ("RESTRICTED_ZONE_ENTRY", "RESTRICTED_ZONE_VIOLATION"))
        self.assertEqual(v2[0]["severity"], "HIGH")
        self.assertTrue("description" in v2[0])
        self.assertEqual(workers[7]["behavior_state"], "INSIDE_RESTRICTED_ZONE")

    def test_no_helmet_temporal_persistence(self):
        """Missing helmet triggers NO_HELMET after configured persistence threshold."""
        workers = {
            3: {
                "person_id": 3,
                "inside_restricted_zone": False,
                "helmet_present": False,
                "vest_present": True
            }
        }

        # t=0.0s
        self.engine.update(workers, current_time_sec=0.0)
        self.assertEqual(workers[3]["behavior_state"], "NORMAL")

        # t=2.5s -> confirmed violation
        viols = self.engine.update(workers, current_time_sec=2.5)
        self.assertEqual(len(viols), 1)
        self.assertEqual(viols[0]["violation_type"], "NO_HELMET")
        self.assertEqual(viols[0]["severity"], "MEDIUM")
        self.assertEqual(workers[3]["behavior_state"], "NO_HELMET")

    def test_combined_high_risk_violation(self):
        """Worker inside restricted zone without helmet escalates to HIGH RISK."""
        workers = {
            7: {
                "person_id": 7,
                "inside_restricted_zone": True,
                "current_zone_id": "zone_danger",
                "helmet_present": False,
                "vest_present": True
            }
        }

        # Initial timestamp
        self.engine.update(workers, current_time_sec=10.0)

        # After 2.5 seconds
        viols = self.engine.update(workers, current_time_sec=12.5)
        self.assertEqual(len(viols), 1)
        self.assertIn(viols[0]["violation_type"], ("HIGH_RISK_PPE_VIOLATION", "HIGH_RISK_SAFETY_VIOLATION"))
        self.assertIn(viols[0]["severity"], ("HIGH", "CRITICAL"))
        self.assertEqual(workers[7]["behavior_state"], "HIGH_RISK")
        self.assertIn("helmet", viols[0]["description"])

    def test_fall_suspected_persistence(self):
        """Sustained prone posture triggers fall event with CRITICAL severity."""
        workers = {
            2: {
                "person_id": 2,
                "inside_restricted_zone": False,
                "helmet_present": True,
                "vest_present": True,
                "pose_data": {"fall_suspected": True, "confidence": 0.88}
            }
        }

        self.engine.update(workers, current_time_sec=5.0)
        # At t=6.8s (duration 1.8s >= 1.5s)
        viols = self.engine.update(workers, current_time_sec=6.8)
        self.assertEqual(len(viols), 1)
        self.assertEqual(viols[0]["violation_type"], "FALL_SUSPECTED")
        self.assertEqual(viols[0]["severity"], "CRITICAL")
        self.assertEqual(workers[2]["behavior_state"], "FALL_SUSPECTED")

    def test_person_specific_behavior_state(self):
        """State is maintained distinctly per person ID without global leakage."""
        workers = {
            1: {
                "person_id": 1,
                "inside_restricted_zone": False,
                "helmet_present": True,
                "vest_present": True
            },
            2: {
                "person_id": 2,
                "inside_restricted_zone": False,
                "helmet_present": False,
                "vest_present": True
            },
            3: {
                "person_id": 3,
                "inside_restricted_zone": True,
                "current_zone_id": "zone_1",
                "helmet_present": True,
                "vest_present": True
            }
        }

        # t=0.0s initialization
        self.engine.update(workers, current_time_sec=0.0)

        # t=2.5s sustained
        viols = self.engine.update(workers, current_time_sec=2.5)
        self.assertEqual(workers[1]["behavior_state"], "NORMAL_PPE")
        self.assertEqual(workers[2]["behavior_state"], "NO_HELMET")
        self.assertEqual(workers[3]["behavior_state"], "INSIDE_RESTRICTED_ZONE")

        viol_pids = {v["person_id"] for v in viols}
        self.assertIn(2, viol_pids)
        self.assertIn(3, viol_pids)
        self.assertNotIn(1, viol_pids)

    def test_walking_motion_state(self):
        """Worker moving continuously is classified as WALKING."""
        engine = BehaviorEngine(standing_seconds=5.0)
        workers = {
            1: {
                "person_id": 1,
                "center": [100.0, 100.0],
                "inside_restricted_zone": False,
                "helmet_present": True,
                "vest_present": True
            }
        }
        engine.update(workers, current_time_sec=0.0)

        # Worker moves by 60 pixels over 1.0 second (speed = 60 px/s >= 20 px/s)
        workers[1]["center"] = [160.0, 100.0]
        engine.update(workers, current_time_sec=1.0)
        self.assertEqual(workers[1]["motion_state"], "WALKING")
        self.assertFalse(workers[1]["standing_over_5s"])

    def test_standing_over_5s_triggers_violation(self):
        """Worker stationary for > 5s is classified as STANDING and triggers alert."""
        engine = BehaviorEngine(standing_seconds=5.0)
        workers = {
            3: {
                "person_id": 3,
                "center": [300.0, 300.0],
                "inside_restricted_zone": False,
                "helmet_present": True,
                "vest_present": True
            }
        }

        # Frame at t=0.0s
        engine.update(workers, current_time_sec=0.0)

        # Frame at t=2.0s (stationary)
        engine.update(workers, current_time_sec=2.0)
        self.assertEqual(workers[3]["motion_state"], "STANDING")
        self.assertFalse(workers[3]["standing_over_5s"])

        # Frame at t=5.5s (> 5.0s standing)
        viols = engine.update(workers, current_time_sec=5.5)
        self.assertEqual(workers[3]["motion_state"], "STANDING")
        self.assertTrue(workers[3]["standing_over_5s"])
        self.assertEqual(workers[3]["behavior_state"], "PROLONGED_STANDING")
        self.assertTrue(any(v["violation_type"] == "PROLONGED_STANDING" for v in viols))


if __name__ == "__main__":
    unittest.main()
