"""
Workplace Safety AI - Restricted Zone Manager with Polygonal Geofencing
"""

from __future__ import annotations
import os
import logging
from typing import Dict, List, Tuple, Any, Optional

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

from src.utils import scale_polygon_points, load_yaml_config


class ZoneManager:
    """
    Manages polygonal restricted zones, performs geofencing checks
    against worker foot positions, and tracks spatial transition states.
    """

    def __init__(
        self,
        zones_config_path: Optional[str] = "configs/zones.yaml",
        raw_zones_list: Optional[List[Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ZoneManager.

        Args:
            zones_config_path: Path to YAML zone configuration file.
            raw_zones_list: Optional direct list of zone definitions (useful for testing).
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("WorkplaceSafetyAI.ZoneManager")
        self.zones: List[Dict[str, Any]] = []
        self._cached_contours: Dict[str, np.ndarray] = {}
        self._previous_worker_states: Dict[int, str] = {}  # person_id -> last zone state

        if raw_zones_list is not None:
            self._load_from_list(raw_zones_list)
        elif zones_config_path and os.path.exists(zones_config_path):
            self._load_from_yaml(zones_config_path)
        else:
            self.logger.warning("No zones configuration found at %s. Creating default restricted zone.", zones_config_path)
            self._create_default_zone()

    def _load_from_yaml(self, path: str) -> None:
        """Load zone geometries from YAML configuration file."""
        try:
            data = load_yaml_config(path)
            if "zones" in data:
                if isinstance(data["zones"], list):
                    self._load_from_list(data["zones"])
                elif isinstance(data["zones"], dict):
                    parsed_list = []
                    for zid, zval in data["zones"].items():
                        if isinstance(zval, dict):
                            zdict = dict(zval)
                            zdict.setdefault("id", zid)
                            parsed_list.append(zdict)
                    self._load_from_list(parsed_list)
            elif "restricted_zone" in data and isinstance(data["restricted_zone"], dict):
                rz = data["restricted_zone"]
                self._load_from_list([{
                    "id": "restricted_zone",
                    "name": rz.get("name", "Restricted Zone"),
                    "type": "restricted",
                    "severity": rz.get("severity", "HIGH"),
                    "points": rz.get("points", []),
                    "color": rz.get("color", [30, 30, 220])
                }])
            else:
                self.logger.warning("No standard zones structure found in %s. Creating default.", path)
                self._create_default_zone()
            self.logger.info("Loaded %d safety zone(s) from %s", len(self.zones), path)
        except Exception as exc:
            self.logger.error("Failed to parse zones configuration file %s: %s", path, exc)
            self._create_default_zone()

    def _load_from_list(self, zones_list: List[Dict[str, Any]]) -> None:
        """Parse list of zone dictionaries."""
        self.zones = []
        for z in zones_list:
            zone_id = z.get("id", f"zone_{len(self.zones) + 1}")
            points = z.get("points", [])
            if len(points) < 3:
                self.logger.warning("Skipping invalid zone %s: requires at least 3 vertices.", zone_id)
                continue

            color = z.get("color", [0, 0, 220])
            self.zones.append({
                "id": zone_id,
                "name": z.get("name", zone_id),
                "type": z.get("type", "restricted"),
                "severity": z.get("severity", "HIGH"),
                "points": points,
                "color": tuple(int(c) for c in color)
            })

    def _create_default_zone(self) -> None:
        """Fallback default restricted zone."""
        self.zones = [{
            "id": "restricted_zone_default",
            "name": "Machinery Danger Perimeter",
            "type": "restricted",
            "severity": "HIGH",
            "points": [[100, 100], [500, 100], [500, 420], [100, 420]],
            "color": (0, 0, 220)
        }]

    def point_inside_zone(self, point: Tuple[float, float], zone_points: List[List[float]]) -> bool:
        """
        Check if a given (x, y) point is inside a polygon using OpenCV pointPolygonTest
        or pure Python ray-casting fallback.

        Args:
            point: (x, y) coordinates.
            zone_points: List of polygon vertices [[x1, y1], [x2, y2], ...].

        Returns:
            True if point is strictly inside or on polygon boundary.
        """
        if cv2 is not None and np is not None:
            try:
                poly_arr = np.array(zone_points, dtype=np.int32)
                dist = cv2.pointPolygonTest(poly_arr, (float(point[0]), float(point[1])), measureDist=False)
                return dist >= 0
            except Exception:
                pass

        # Robust Ray-Casting algorithm fallback (CPU/No-dependency)
        x, y = float(point[0]), float(point[1])
        n = len(zone_points)
        inside = False
        p1x, p1y = zone_points[0]
        for i in range(1, n + 1):
            p2x, p2y = zone_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def worker_inside_zone(
        self,
        worker_foot_point: Tuple[float, float],
        zone_id: Optional[str] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Determine whether a worker's foot contact point is inside a restricted zone.

        Args:
            worker_foot_point: (x, y) ground location of worker.
            zone_id: Optional specific zone to test. If None, checks all restricted zones.
            frame_width: Optional frame width to scale points.
            frame_height: Optional frame height to scale points.

        Returns:
            Tuple of (is_inside, matching_zone_dict_or_None).
        """
        for zone in self.zones:
            if zone_id and zone["id"] != zone_id:
                continue
            points = zone["points"]
            if frame_width and frame_height:
                points = scale_polygon_points(points, frame_width, frame_height)
            if self.point_inside_zone(worker_foot_point, points):
                return True, zone
        return False, None

    def update_workers_zones(
        self,
        workers: Dict[int, Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> None:
        """
        Evaluate all workers against restricted zones and update state transition attributes:
        OUTSIDE -> ENTERING -> INSIDE -> EXITING.
        """
        for person_id, worker in workers.items():
            foot = worker.get("foot_point", worker["center"])
            is_inside, matched_zone = self.worker_inside_zone(
                foot, frame_width=frame_width, frame_height=frame_height
            )

            prev_state = self._previous_worker_states.get(person_id, "OUTSIDE")

            if is_inside:
                if prev_state in ("OUTSIDE", "EXITING"):
                    current_state = "ENTERING"
                else:
                    current_state = "INSIDE"
                worker["inside_restricted_zone"] = True
                worker["current_zone_id"] = matched_zone["id"] if matched_zone else None
            else:
                if prev_state in ("INSIDE", "ENTERING"):
                    current_state = "EXITING"
                else:
                    current_state = "OUTSIDE"
                worker["inside_restricted_zone"] = False
                worker["current_zone_id"] = None

            worker["zone_state"] = current_state
            self._previous_worker_states[person_id] = current_state

    def draw_zones(
        self,
        frame: np.ndarray,
        alpha: Optional[float] = None,
        violation_count: int = 0,
        workers: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Render restricted zone boundaries with subtle, professional styling:
        - Thin red border (1px normal, 2px violation)
        - Subtle transparent red fill (alpha ≈ 0.08–0.12 normal, 0.12–0.16 active)
        - Single clean zone label at top-left:
          '● RESTRICTED' when clear
          '⚠ RESTRICTED • {N} INSIDE' when active
        """
        if not self.zones or frame is None:
            return frame

        h, w = frame.shape[:2]

        inside_count = violation_count
        if workers:
            inside_count = sum(1 for wkr in workers.values() if wkr.get("inside_restricted_zone"))

        # Red zone styling - transparent so CCTV footage remains clearly visible
        fill_color = (25, 25, 190)       # Subtle deep red
        border_color = (35, 35, 225)     # Crisp red boundary line

        if inside_count > 0:
            fill_alpha = alpha if alpha is not None else 0.14
            border_thickness = 3
            status_text = f"RESTRICTED ZONE \u2022 {inside_count} INSIDE"
            is_violation = True
        else:
            fill_alpha = alpha if alpha is not None else 0.10
            border_thickness = 2
            status_text = "RESTRICTED ZONE"
            is_violation = False

        # 1. Semi-transparent polygon fill
        overlay = frame.copy()
        for zone in self.zones:
            pts = scale_polygon_points(zone["points"], w, h)
            cv2.fillPoly(overlay, [pts], fill_color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

        # 2. Crisp boundary lines and single clean top-left label
        for zone in self.zones:
            pts = scale_polygon_points(zone["points"], w, h)
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=border_thickness, lineType=cv2.LINE_AA)

            # Top-leftmost point of polygon
            min_y_idx = int(np.argmin(pts[:, 1]))
            top_pt = pts[min_y_idx]
            min_x_val = int(np.min(pts[:, 0]))

            lx = max(14, min_x_val)
            ly = max(24, int(top_pt[1]) - 8)

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.50
            (tw, th), _ = cv2.getTextSize(status_text, font, font_scale, 2)

            # Icon width (dot or warning triangle): 22px + padding
            tag_w = tw + 46
            tag_h = th + 14
            bx1 = lx
            by1 = max(32, ly - tag_h + 2)
            bx2 = bx1 + tag_w
            by2 = by1 + tag_h

            # Dark translucent badge with bold 2px border
            tag_overlay = frame.copy()
            cv2.rectangle(tag_overlay, (bx1, by1), (bx2, by2), (16, 18, 22), -1)
            cv2.addWeighted(tag_overlay, 0.86, frame, 0.14, 0, frame)
            badge_border = (40, 40, 235) if is_violation else (55, 60, 72)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_border, 2, lineType=cv2.LINE_AA)

            # Draw icon
            icon_cx = bx1 + 14
            icon_cy = (by1 + by2) // 2
            if is_violation:
                # Small amber warning triangle with !
                tri_pts = np.array([
                    [icon_cx, icon_cy - 7],
                    [icon_cx - 7, icon_cy + 6],
                    [icon_cx + 7, icon_cy + 6]
                ], dtype=np.int32)
                cv2.fillPoly(frame, [tri_pts], (0, 165, 255), lineType=cv2.LINE_AA)
                cv2.line(frame, (icon_cx, icon_cy - 3), (icon_cx, icon_cy + 1), (0, 0, 0), 2)
                cv2.circle(frame, (icon_cx, icon_cy + 4), 1, (0, 0, 0), -1)
                text_color = (255, 255, 255)
            else:
                # Red dot
                cv2.circle(frame, (icon_cx, icon_cy), 4, (40, 40, 230), -1, lineType=cv2.LINE_AA)
                text_color = (240, 242, 245)

            # Render bold text
            cv2.putText(
                frame,
                status_text,
                (bx1 + 26, by2 - 6),
                font,
                font_scale,
                text_color,
                2,
                cv2.LINE_AA
            )

        return frame
