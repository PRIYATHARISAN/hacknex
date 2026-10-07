"""
Unit tests for ZoneManager (CPU only, no GPU or external models required)
"""

import unittest
from src.zone_manager import ZoneManager


class TestZoneManager(unittest.TestCase):

    def setUp(self):
        # Define a controlled rectangular polygon: [100, 100] to [500, 400]
        self.raw_zones = [{
            "id": "restricted_area_alpha",
            "name": "Loading Dock",
            "type": "restricted",
            "severity": "HIGH",
            "points": [[100, 100], [500, 100], [500, 400], [100, 400]],
            "color": [0, 0, 255]
        }]
        self.zone_mgr = ZoneManager(raw_zones_list=self.raw_zones)

    def test_point_strictly_inside_zone(self):
        """Points clearly within polygon boundaries should return True."""
        inside_point = (250, 250)
        is_in = self.zone_mgr.point_inside_zone(inside_point, self.raw_zones[0]["points"])
        self.assertTrue(is_in)

    def test_point_outside_zone(self):
        """Points outside polygon boundaries should return False."""
        outside_points = [(50, 50), (600, 250), (250, 450), (0, 0)]
        for pt in outside_points:
            is_in = self.zone_mgr.point_inside_zone(pt, self.raw_zones[0]["points"])
            self.assertFalse(is_in, f"Point {pt} should be outside zone.")

    def test_worker_foot_point_inside(self):
        """Worker inside zone detection via foot point."""
        worker_foot = (300, 300)
        is_inside, matched_zone = self.zone_mgr.worker_inside_zone(worker_foot)
        self.assertTrue(is_inside)
        self.assertIsNotNone(matched_zone)
        self.assertEqual(matched_zone["id"], "restricted_area_alpha")

    def test_worker_state_transitions(self):
        """Verify OUTSIDE -> ENTERING -> INSIDE -> EXITING transitions."""
        workers = {
            1: {
                "person_id": 1,
                "bbox": [20, 20, 40, 60],
                "center": [30, 40],
                "foot_point": [30, 60]  # Outside
            }
        }

        # Step 1: Outside
        self.zone_mgr.update_workers_zones(workers, 640, 480)
        self.assertEqual(workers[1]["zone_state"], "OUTSIDE")
        self.assertFalse(workers[1]["inside_restricted_zone"])

        # Step 2: Enters boundary
        workers[1]["foot_point"] = [200, 200]  # Inside
        self.zone_mgr.update_workers_zones(workers, 640, 480)
        self.assertEqual(workers[1]["zone_state"], "ENTERING")
        self.assertTrue(workers[1]["inside_restricted_zone"])

        # Step 3: Sustained inside
        workers[1]["foot_point"] = [210, 210]  # Still inside
        self.zone_mgr.update_workers_zones(workers, 640, 480)
        self.assertEqual(workers[1]["zone_state"], "INSIDE")
        self.assertTrue(workers[1]["inside_restricted_zone"])

        # Step 4: Exiting boundary
        workers[1]["foot_point"] = [50, 50]  # Moved outside
        self.zone_mgr.update_workers_zones(workers, 640, 480)
        self.assertEqual(workers[1]["zone_state"], "EXITING")
        self.assertFalse(workers[1]["inside_restricted_zone"])

        # Step 5: Remains outside
        workers[1]["foot_point"] = [40, 40]
        self.zone_mgr.update_workers_zones(workers, 640, 480)
        self.assertEqual(workers[1]["zone_state"], "OUTSIDE")


if __name__ == "__main__":
    unittest.main()
