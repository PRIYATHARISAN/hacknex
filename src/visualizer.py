"""
Workplace Safety AI - Bold Industrial CCTV Safety Visualizer
Delivers high-contrast, bold, and immediately understandable visual monitoring from a distance.
"""

from __future__ import annotations
import cv2
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.utils import seconds_to_timestamp


class SafetyVisualizer:
    """
    Renders bold, high-contrast, and clean industrial CCTV safety overlays:
    - Dominant CCTV video with clear spatial awareness
    - Clean 2px bounding boxes with bold '#1', '#2 \u26a0', or '#4 !' badges
    - Subtle 1px pose skeletons
    - Prominent bold centered top alert: '\u26a0 HIGH \u2022 PERSON #2 \u2022 RESTRICTED ZONE'
    - High-visibility right-hand dashboard with large metrics and clear status
    """

    # High-contrast industrial CCTV color palette
    COLOR_NORMAL = (60, 205, 80)        # Green (Safe)
    COLOR_WARNING = (0, 165, 255)       # Amber / Orange (Warning)
    COLOR_CRITICAL = (40, 40, 235)      # Red (Critical)
    COLOR_WHITE = (245, 248, 250)       # High-contrast white
    COLOR_MUTED = (160, 165, 175)       # Supporting gray
    COLOR_DARK_PANEL = (16, 18, 22)     # Semi-transparent dark slate
    COLOR_BORDER = (48, 52, 60)         # Subtle border

    # Skeleton connections for YOLO pose keypoints
    SKELETON_PAIRS = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]

    def __init__(
        self,
        display_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Visualizer.

        Args:
            display_config: Display flags from config.yaml.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("WorkplaceSafetyAI.Visualizer")
        self.cfg = display_config or {}
        self.show_tracking = self.cfg.get("show_tracking", True)
        self.show_pose = self.cfg.get("show_pose", True)
        self.show_zones = self.cfg.get("show_zones", True)
        self.show_ppe = self.cfg.get("show_ppe", True)
        self.show_behavior = self.cfg.get("show_behavior", True)
        self.show_events = self.cfg.get("show_events", True)
        self.show_fps = self.cfg.get("show_fps", True)
        self.show_dashboard = self.cfg.get("show_dashboard", True)

    def draw_worker_annotations(
        self,
        frame: np.ndarray,
        workers: Dict[int, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw clean, bold worker annotations:
        - Clean 2px bounding box (green for normal, orange for warning, red for violation)
        - Bold, readable label above person: '#1', '#2 \u26a0', or '#4 !'
        - Subtle 1px pose skeleton with small keypoint dots
        - Zero bulky debug boxes
        """
        if not self.show_tracking or frame is None or not workers:
            return frame

        fh, fw = frame.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX

        for pid, worker in workers.items():
            bbox = worker.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw - 1, x2), min(fh - 1, y2)

            behavior = worker.get("behavior_state", "NORMAL")
            in_zone = worker.get("inside_restricted_zone", False)

            motion_state = worker.get("motion_state", "WALKING")
            standing_dur = worker.get("standing_duration", 0.0)
            standing_over_5s = worker.get("standing_over_5s", False) or (standing_dur > 5.0)

            # Determine violation level (Standing > 5s triggers RED alert)
            is_critical = (
                in_zone
                or standing_over_5s
                or behavior in ("HIGH_RISK", "FALL_SUSPECTED", "FALL_CONFIRMED", "INSIDE_RESTRICTED_ZONE", "RESTRICTED_ZONE_VIOLATION", "PROLONGED_STANDING")
            )
            is_warning = (
                not is_critical
                and behavior in ("ENTERING_RESTRICTED_ZONE", "NO_HELMET", "NO_VEST", "PPE_VIOLATION", "ABNORMAL_POSTURE", "WARNING")
            )

            if is_critical:
                box_color = self.COLOR_CRITICAL   # Bold Red
            elif is_warning:
                box_color = self.COLOR_WARNING    # Orange
            else:
                box_color = self.COLOR_NORMAL     # Green

            # 1. Clean 2px bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, lineType=cv2.LINE_AA)

            # 2. Ground Contact Foot Dot
            fx, fy = worker.get("foot_point", [(x1 + x2) // 2, y2])
            cv2.circle(frame, (int(round(fx)), int(round(fy))), 3, box_color, -1, lineType=cv2.LINE_AA)

            # 3. Bold, readable label: '#1 WALKING' or '#3 STANDING' (Red if > 5s)
            state_str = "STANDING" if motion_state == "STANDING" else "WALKING"
            label_text = f"#{pid} {state_str}"
            font_scale = 0.44
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, 2)

            has_symbol = is_critical or is_warning
            tag_w = tw + (28 if has_symbol else 16)
            tag_h = th + 10

            bx1 = x1
            by1 = y1 - tag_h - 4
            if by1 < 6:
                by1 = y1 + 4
            bx2 = bx1 + tag_w
            by2 = by1 + tag_h

            # Dark translucent badge for normal, bold solid RED for standing > 5s / critical
            if is_critical:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 210), -1)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 60, 255), 2, lineType=cv2.LINE_AA)
            elif is_warning:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 100, 200), -1)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 180, 255), 2, lineType=cv2.LINE_AA)
            else:
                tag_overlay = frame.copy()
                cv2.rectangle(tag_overlay, (bx1, by1), (bx2, by2), self.COLOR_DARK_PANEL, -1)
                cv2.addWeighted(tag_overlay, 0.88, frame, 0.12, 0, frame)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2, lineType=cv2.LINE_AA)

            # Draw label text
            cv2.putText(
                frame,
                label_text,
                (bx1 + 6, by2 - 4),
                font,
                font_scale,
                self.COLOR_WHITE,
                2,
                cv2.LINE_AA
            )

            # Draw symbol: warning triangle for critical/warning
            if has_symbol:
                icon_cx = bx1 + tw + 17
                icon_cy = (by1 + by2) // 2
                tri_pts = np.array([
                    [icon_cx, icon_cy - 6],
                    [icon_cx - 6, icon_cy + 5],
                    [icon_cx + 6, icon_cy + 5]
                ], dtype=np.int32)
                icon_col = (0, 220, 255) if is_critical else (255, 255, 255)
                cv2.fillPoly(frame, [tri_pts], icon_col, lineType=cv2.LINE_AA)
                cv2.line(frame, (icon_cx, icon_cy - 3), (icon_cx, icon_cy + 1), (0, 0, 0), 2)
                cv2.circle(frame, (icon_cx, icon_cy + 3), 1, (0, 0, 0), -1)

            # 4. Subtle Pose Skeleton (1px lines, 2px keypoint dots)
            if self.show_pose:
                pose_data = worker.get("pose_data")
                if pose_data and "keypoints" in pose_data:
                    self._draw_subtle_skeleton(frame, pose_data["keypoints"], box_color)

        return frame

    def _draw_subtle_skeleton(
        self,
        frame: np.ndarray,
        kpts: Dict[str, Tuple[float, float, float]],
        color: Tuple[int, int, int]
    ) -> None:
        """
        Draw thin, visually supportive pose skeleton links and dots.
        Never dominates or obscures the worker.
        """
        for p1, p2 in self.SKELETON_PAIRS:
            if p1 in kpts and p2 in kpts:
                x1, y1, c1 = kpts[p1]
                x2, y2, c2 = kpts[p2]
                if c1 > 0.40 and c2 > 0.40:
                    cv2.line(
                        frame,
                        (int(round(x1)), int(round(y1))),
                        (int(round(x2)), int(round(y2))),
                        color,
                        1,
                        cv2.LINE_AA
                    )

        for pt_name, (kx, ky, kc) in kpts.items():
            if kc > 0.45:
                cv2.circle(
                    frame,
                    (int(round(kx)), int(round(ky))),
                    2,
                    (220, 225, 230),
                    -1,
                    cv2.LINE_AA
                )

    def draw_bag_annotations(
        self,
        frame: np.ndarray,
        bags: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Render bag annotations if detected:
        - Clean 2px bounding box
        - Bold badge: 'Bag #3' / 'PICKED'
        """
        if not bags or frame is None:
            return frame

        font = cv2.FONT_HERSHEY_DUPLEX
        for bag in bags:
            box = bag.get("bbox", bag.get("box"))
            if box is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            bid = bag.get("id", 1)
            state = str(bag.get("state", "PLACED")).upper()

            # Clean 2px outline
            cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 190, 70), 2, lineType=cv2.LINE_AA)

            line1 = f"Bag #{bid}"
            line2 = state
            scale = 0.44
            (w1, h1), _ = cv2.getTextSize(line1, font, scale, 1)
            (w2, h2), _ = cv2.getTextSize(line2, font, scale, 1)
            tw = max(w1, w2)

            bx1 = x1
            by1 = max(6, y1 - (h1 + h2 + 14))
            bx2 = bx1 + tw + 14
            by2 = by1 + (h1 + h2 + 12)

            tag_overlay = frame.copy()
            cv2.rectangle(tag_overlay, (bx1, by1), (bx2, by2), self.COLOR_DARK_PANEL, -1)
            cv2.addWeighted(tag_overlay, 0.86, frame, 0.14, 0, frame)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), self.COLOR_BORDER, 1, lineType=cv2.LINE_AA)

            cv2.putText(frame, line1, (bx1 + 6, by1 + h1 + 4), font, scale, self.COLOR_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, line2, (bx1 + 6, by2 - 4), font, scale, self.COLOR_MUTED, 1, cv2.LINE_AA)

        return frame

    def draw_top_status_bar(
        self,
        frame: np.ndarray,
        overall_status: str = "SAFE"
    ) -> np.ndarray:
        """
        Render a clean, thin status bar at the top of the CCTV view.
        Example:
        WORKPLACE SAFETY MONITOR                      \u25cf SAFE / \u25cf WARNING / \u25cf CRITICAL
        """
        if frame is None:
            return frame

        fh, fw = frame.shape[:2]
        bar_h = 30

        # Translucent dark status bar across top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, bar_h), (14, 16, 20), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # 1px bottom accent line
        cv2.line(frame, (0, bar_h), (fw, bar_h), (38, 42, 50), 1)

        # Left title
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frame,
            "WORKPLACE SAFETY MONITOR",
            (16, 20),
            font,
            0.46,
            (230, 235, 240),
            2,
            cv2.LINE_AA
        )

        # Right status indicator
        status_norm = overall_status.upper()
        if status_norm == "CRITICAL":
            dot_color = self.COLOR_CRITICAL
            text_color = (60, 60, 255)
            badge_text = "CRITICAL"
        elif status_norm == "WARNING":
            dot_color = self.COLOR_WARNING
            text_color = (0, 180, 255)
            badge_text = "WARNING"
        else:
            dot_color = self.COLOR_NORMAL
            text_color = (80, 220, 100)
            badge_text = "SAFE"

        # Position right badge cleanly on far right of top bar
        rx = fw - 130
        cv2.circle(frame, (rx, 15), 4, dot_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            badge_text,
            (rx + 11, 20),
            font,
            0.46,
            text_color,
            2,
            cv2.LINE_AA
        )

        return frame

    def draw_active_alert_banner(
        self,
        frame: np.ndarray,
        active_violations: Optional[List[Dict[str, Any]]] = None,
        workers: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Render a prominent, bold top alert centered below the top bar when an active violation occurs.
        Example:
        \u26a0  HIGH  \u2022  PERSON #2  \u2022  RESTRICTED ZONE
        """
        if not self.show_events or frame is None:
            return frame

        pid = None
        sev = "HIGH"
        vtype = "RESTRICTED ZONE"

        # 1. Prioritize active violations from behavior engine
        crit_viols = [v for v in active_violations if v.get("severity") in ("CRITICAL", "HIGH")] if active_violations else []
        if crit_viols:
            top_viol = crit_viols[0]
            pid = top_viol.get("person_id", "?")
            sev = str(top_viol.get("severity", "HIGH")).upper()
            raw_vtype = str(top_viol.get("violation_type", "RESTRICTED ZONE"))
            if "STANDING" in raw_vtype:
                vtype = "STANDING > 5s"
            else:
                vtype = (
                    raw_vtype
                    .replace("_VIOLATION", "")
                    .replace("_ENTRY", "")
                    .replace("_", " ")
                )
        elif workers:
            # 2. Check active workers inside restricted perimeter or standing > 5s
            inside_workers = [
                (w_id, w) for w_id, w in workers.items()
                if w.get("inside_restricted_zone") or w.get("behavior_state") in ("INSIDE_RESTRICTED_ZONE", "ENTERING_RESTRICTED_ZONE", "HIGH_RISK")
            ]
            standing_workers = [
                (w_id, w) for w_id, w in workers.items()
                if w.get("standing_over_5s") or w.get("behavior_state") == "PROLONGED_STANDING"
            ]
            if inside_workers:
                pid, w = inside_workers[0]
                sev = "HIGH"
                vtype = "RESTRICTED ZONE"
            elif standing_workers:
                pid, w = standing_workers[0]
                sev = "HIGH"
                vtype = "STANDING > 5s"

        if pid is None:
            return frame

        banner_text = f"{sev}   \u2022   PERSON #{pid}   \u2022   {vtype}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.58
        (tw, th), _ = cv2.getTextSize(banner_text, font, font_scale, 2)

        fh, fw = frame.shape[:2]
        pill_w = tw + 56
        pill_h = th + 18
        bx1 = (fw - pill_w) // 2
        bx2 = bx1 + pill_w
        by1 = 34
        by2 = by1 + pill_h

        # Translucent dark pill with prominent 2px border
        pill_overlay = frame.copy()
        cv2.rectangle(pill_overlay, (bx1, by1), (bx2, by2), (18, 14, 20), -1)
        cv2.addWeighted(pill_overlay, 0.90, frame, 0.10, 0, frame)

        border_col = self.COLOR_CRITICAL if sev == "CRITICAL" else self.COLOR_WARNING
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_col, 2, lineType=cv2.LINE_AA)

        # Bold warning icon on the left
        icon_cx = bx1 + 18
        icon_cy = (by1 + by2) // 2
        tri_pts = np.array([
            [icon_cx, icon_cy - 8],
            [icon_cx - 8, icon_cy + 7],
            [icon_cx + 8, icon_cy + 7]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [tri_pts], (0, 165, 255), lineType=cv2.LINE_AA)
        cv2.line(frame, (icon_cx, icon_cy - 3), (icon_cx, icon_cy + 2), (0, 0, 0), 2)
        cv2.circle(frame, (icon_cx, icon_cy + 5), 1, (0, 0, 0), -1)

        # Bold banner text
        cv2.putText(
            frame,
            banner_text,
            (bx1 + 34, by2 - 7),
            font,
            font_scale,
            self.COLOR_WHITE,
            2,
            cv2.LINE_AA
        )

        return frame

    def draw_safety_dashboard(
        self,
        frame: np.ndarray,
        workers: Dict[int, Dict[str, Any]],
        recent_events: List[Dict[str, Any]],
        fps: float,
        bags: Optional[List[Dict[str, Any]]] = None,
        active_violations: Optional[List[Dict[str, Any]]] = None,
        overall_status: Optional[str] = None,
        detection_ms: float = 0.0,
        tracking_ms: float = 0.0,
        pose_ms: float = 0.0
    ) -> np.ndarray:
        """
        Render larger, bolder right-hand safety dashboard (approx 17-18% of video width).
        Matches exact layout from Requirement 9:
        ┌──────────────────────────┐
        │ WORKPLACE SAFETY         │
        │                          │
        │ STATUS                   │
        │ ● CRITICAL               │
        │                          │
        │ PEOPLE        BAGS       │
        │   08           02        │
        │                          │
        │ ──────────────────────── │
        │                          │
        │ RESTRICTED ZONE          │
        │ ⚠ 4 INSIDE              │
        │                          │
        │ ──────────────────────── │
        │                          │
        │ RECENT ALERTS            │
        │                          │
        │ #2  Restricted Zone      │
        │ #3  Restricted Zone      │
        │ #4  Restricted Zone      │
        │                          │
        │ ──────────────────────── │
        │ PPE                      │
        │ NOT CONFIGURED           │
        └──────────────────────────┘
        """
        if not self.show_dashboard or frame is None:
            return frame

        fh, fw = frame.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX

        # Dashboard geometry: ~17-18% of frame width (340px for 1900px wide video)
        dash_w = 340
        x1 = fw - dash_w - 16
        x2 = fw - 16
        y1 = 36
        dash_h = 590
        y2 = min(fh - 14, y1 + dash_h)

        # 1. Dark semi-transparent panel background (alpha ≈ 0.85)
        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (x1, y1), (x2, y2), self.COLOR_DARK_PANEL, -1)
        cv2.addWeighted(panel_overlay, 0.86, frame, 0.14, 0, frame)

        # 1px border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_BORDER, 1, lineType=cv2.LINE_AA)

        pad_x = x1 + 20
        curr_y = y1 + 28

        def draw_divider(y_pos: int) -> int:
            cv2.line(frame, (x1 + 16, y_pos), (x2 - 16, y_pos), (45, 50, 60), 1)
            return y_pos + 20

        # Compute dynamic status if not provided
        if overall_status is None:
            has_crit = (
                any(w.get("inside_restricted_zone") for w in workers.values())
                or any(w.get("standing_over_5s") for w in workers.values())
                or any(w.get("behavior_state") in ("HIGH_RISK", "FALL_SUSPECTED", "FALL_CONFIRMED", "PROLONGED_STANDING") for w in workers.values())
                or (active_violations and any(v.get("severity") in ("CRITICAL", "HIGH") for v in active_violations))
            )
            has_warn = (
                any(w.get("behavior_state") in ("ENTERING_RESTRICTED_ZONE", "NO_HELMET", "NO_VEST", "PPE_VIOLATION", "WARNING") for w in workers.values())
                or (active_violations and any(v.get("severity") == "MEDIUM" for v in active_violations))
            )
            overall_status = "CRITICAL" if has_crit else ("WARNING" if has_warn else "SAFE")

        # ---------------------------------------------------------
        # Section 1: Dashboard Title (Bold)
        # ---------------------------------------------------------
        cv2.putText(frame, "WORKPLACE SAFETY", (pad_x, curr_y), font, 0.58, self.COLOR_WHITE, 2, cv2.LINE_AA)
        curr_y = draw_divider(curr_y + 12)

        # ---------------------------------------------------------
        # Section 2: STATUS (Significantly larger & very obvious!)
        # ---------------------------------------------------------
        cv2.putText(frame, "STATUS", (pad_x, curr_y), font, 0.40, self.COLOR_MUTED, 1, cv2.LINE_AA)
        curr_y += 26

        status_norm = overall_status.upper()
        if status_norm == "CRITICAL":
            s_color = self.COLOR_CRITICAL
            s_text = "CRITICAL"
        elif status_norm == "WARNING":
            s_color = self.COLOR_WARNING
            s_text = "WARNING"
        else:
            s_color = self.COLOR_NORMAL
            s_text = "SAFE"

        # Large status indicator circle + Bold large text
        cv2.circle(frame, (pad_x + 6, curr_y - 7), 6, s_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(frame, s_text, (pad_x + 22, curr_y), font, 0.74, s_color, 2, cv2.LINE_AA)
        curr_y = draw_divider(curr_y + 18)

        # ---------------------------------------------------------
        # Section 3: Large Metrics - PEOPLE & BAGS (Side-by-side)
        # ---------------------------------------------------------
        people_count = len(workers)
        bags_count = len(bags) if bags is not None else 0

        col1_x = pad_x
        col2_x = pad_x + 160

        cv2.putText(frame, "PEOPLE", (col1_x, curr_y), font, 0.40, self.COLOR_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, "BAGS", (col2_x, curr_y), font, 0.40, self.COLOR_MUTED, 1, cv2.LINE_AA)
        curr_y += 36

        cv2.putText(frame, f"{people_count:02d}", (col1_x, curr_y), font, 0.96, self.COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{bags_count:02d}", (col2_x, curr_y), font, 0.96, self.COLOR_WHITE, 2, cv2.LINE_AA)
        curr_y = draw_divider(curr_y + 18)

        # ---------------------------------------------------------
        # Section 4: RESTRICTED ZONE Status (Bold)
        # ---------------------------------------------------------
        cv2.putText(frame, "RESTRICTED ZONE", (pad_x, curr_y), font, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)
        curr_y += 24

        inside_count = sum(1 for w in workers.values() if w.get("inside_restricted_zone"))
        if inside_count > 0:
            # Bold warning icon + count
            icon_cx = pad_x + 6
            icon_cy = curr_y - 6
            tri_pts = np.array([
                [icon_cx, icon_cy - 7],
                [icon_cx - 7, icon_cy + 6],
                [icon_cx + 7, icon_cy + 6]
            ], dtype=np.int32)
            cv2.fillPoly(frame, [tri_pts], (0, 165, 255), lineType=cv2.LINE_AA)
            cv2.line(frame, (icon_cx, icon_cy - 3), (icon_cx, icon_cy + 1), (0, 0, 0), 2)
            cv2.circle(frame, (icon_cx, icon_cy + 4), 1, (0, 0, 0), -1)

            plural = "PERSON" if inside_count == 1 else "PERSONS"
            zone_desc = f"{inside_count} INSIDE"
            cv2.putText(frame, zone_desc, (pad_x + 22, curr_y), font, 0.60, self.COLOR_CRITICAL, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (pad_x + 6, curr_y - 6), 5, self.COLOR_NORMAL, -1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "CLEAR", (pad_x + 22, curr_y), font, 0.60, self.COLOR_NORMAL, 2, cv2.LINE_AA)

        curr_y = draw_divider(curr_y + 18)

        # ---------------------------------------------------------
        # Section 5: RECENT ALERTS (Bold & Readable, max 3)
        # ---------------------------------------------------------
        cv2.putText(frame, "RECENT ALERTS", (pad_x, curr_y), font, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)
        curr_y += 24

        if not recent_events:
            cv2.putText(frame, "No active alerts", (pad_x, curr_y), font, 0.44, (125, 130, 140), 1, cv2.LINE_AA)
            curr_y += 24
        else:
            # Show up to 3 most recent events
            for evt in recent_events[:3]:
                epid = evt.get("person_id", "?")
                etype = evt.get("event_type", "").upper()

                if "RESTRICTED" in etype or "ZONE" in etype:
                    short_desc = "Restricted Zone"
                elif "STANDING" in etype:
                    short_desc = "Standing > 5s"
                elif "HELMET" in etype:
                    short_desc = "No Helmet"
                elif "VEST" in etype:
                    short_desc = "No Vest"
                elif "FALL" in etype:
                    short_desc = "Fall Detected"
                elif "BAG" in etype:
                    short_desc = "Bag Event"
                else:
                    short_desc = etype.replace("_", " ").title()

                alert_line = f"#{epid}  {short_desc}"
                cv2.putText(frame, alert_line, (pad_x, curr_y), font, 0.46, self.COLOR_WHITE, 1, cv2.LINE_AA)
                curr_y += 22

        curr_y = draw_divider(curr_y + 4)

        # ---------------------------------------------------------
        # Section 6: PPE MONITORING STATUS
        # ---------------------------------------------------------
        cv2.putText(frame, "PPE", (pad_x, curr_y), font, 0.40, self.COLOR_MUTED, 1, cv2.LINE_AA)
        curr_y += 22
        cv2.putText(frame, "NOT CONFIGURED", (pad_x, curr_y), font, 0.46, (145, 150, 160), 1, cv2.LINE_AA)
        curr_y += 26

        # Subtle FPS indicator at very bottom
        if self.show_fps and curr_y < y2 - 8:
            cv2.putText(frame, f"FPS: {fps:.1f}", (pad_x, y2 - 12), font, 0.34, (90, 95, 105), 1, cv2.LINE_AA)

        return frame
