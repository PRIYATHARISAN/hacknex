"""
Workplace Safety AI - Temporal Behavior Understanding & State Machine Engine
"""

import logging
from collections import deque
from typing import Dict, List, Any, Optional, Tuple


class BehaviorEngine:
    """
    Temporal behavior reasoning engine.
    Maintains time-series state machines per worker to distinguish transient noise
    from sustained unsafe workplace behaviors.
    """

    def __init__(
        self,
        restricted_zone_seconds: float = 2.0,
        no_helmet_seconds: float = 2.0,
        no_vest_seconds: float = 2.0,
        fall_duration_seconds: float = 1.5,
        standing_seconds: float = 5.0,
        severity_rules: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Behavior Engine.

        Args:
            restricted_zone_seconds: Required persistence inside restricted perimeter.
            no_helmet_seconds: Duration before flagging missing helmet.
            no_vest_seconds: Duration before flagging missing safety vest.
            fall_duration_seconds: Duration before confirming suspected fall.
            standing_seconds: Duration of stationary standing before alert (default: 5.0s).
            severity_rules: Mapping from violation type to severity string.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("WorkplaceSafetyAI.BehaviorEngine")
        self.restricted_zone_seconds = float(restricted_zone_seconds)
        self.no_helmet_seconds = float(no_helmet_seconds)
        self.no_vest_seconds = float(no_vest_seconds)
        self.fall_duration_seconds = float(fall_duration_seconds)
        self.standing_seconds = float(standing_seconds)

        self.severity_rules = severity_rules or {
            "RESTRICTED_ZONE_VIOLATION": "HIGH",
            "NO_HELMET": "MEDIUM",
            "NO_VEST": "MEDIUM",
            "PPE_VIOLATION": "MEDIUM",
            "HIGH_RISK_SAFETY_VIOLATION": "CRITICAL",
            "FALL_SUSPECTED": "CRITICAL",
            "PROLONGED_STANDING": "HIGH"
        }

        # Temporal memory tracked per person_id
        self._temporal_state: Dict[int, Dict[str, Any]] = {}

    def _get_or_init_worker_memory(self, person_id: int) -> Dict[str, Any]:
        if person_id not in self._temporal_state:
            self._temporal_state[person_id] = {
                "zone_enter_time": None,
                "no_helmet_start": None,
                "no_vest_start": None,
                "fall_start": None,
                "standing_start": None,
                "pos_history": deque(maxlen=35)
            }
        return self._temporal_state[person_id]

    def update(
        self,
        workers: Dict[int, Dict[str, Any]],
        current_time_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Process worker states for the current frame, execute temporal reasoning,
        and classify behavior states.

        Args:
            workers: Active workers dictionary from Tracker & Managers.
            current_time_sec: Current video timestamp in seconds.

        Returns:
            List of confirmed active violations in the current frame:
            [
                {
                    "person_id": int,
                    "violation_type": str,
                    "severity": str,
                    "duration": float,
                    "zone_id": Optional[str],
                    "confidence": float,
                    "timestamp": float
                },
                ...
            ]
        """
        active_violations: List[Dict[str, Any]] = []

        for person_id, worker in workers.items():
            mem = self._get_or_init_worker_memory(person_id)

            # -------------------------------------------------------------
            # 1. Temporal Persistence: Restricted Zone
            # -------------------------------------------------------------
            inside_zone = worker.get("inside_restricted_zone", False)
            zone_id = worker.get("current_zone_id")
            zone_duration = 0.0

            if inside_zone:
                if mem["zone_enter_time"] is None:
                    mem["zone_enter_time"] = current_time_sec
                zone_duration = current_time_sec - mem["zone_enter_time"]
            else:
                mem["zone_enter_time"] = None

            # -------------------------------------------------------------
            # 2. Temporal Persistence: Missing Helmet
            # -------------------------------------------------------------
            helmet_present = worker.get("helmet_present")
            no_helmet_duration = 0.0
            if helmet_present is False:
                if mem["no_helmet_start"] is None:
                    mem["no_helmet_start"] = current_time_sec
                no_helmet_duration = current_time_sec - mem["no_helmet_start"]
            else:
                mem["no_helmet_start"] = None

            # -------------------------------------------------------------
            # 3. Temporal Persistence: Missing Vest
            # -------------------------------------------------------------
            vest_present = worker.get("vest_present")
            no_vest_duration = 0.0
            if vest_present is False:
                if mem["no_vest_start"] is None:
                    mem["no_vest_start"] = current_time_sec
                no_vest_duration = current_time_sec - mem["no_vest_start"]
            else:
                mem["no_vest_start"] = None

            # -------------------------------------------------------------
            # 4. Temporal Persistence: Suspected Fall
            # -------------------------------------------------------------
            pose_data = worker.get("pose_data")
            is_falling = pose_data.get("fall_suspected", False) if pose_data else False
            fall_duration = 0.0
            if is_falling:
                if mem["fall_start"] is None:
                    mem["fall_start"] = current_time_sec
                fall_duration = current_time_sec - mem["fall_start"]
            else:
                mem["fall_start"] = None

            # -------------------------------------------------------------
            # Motion Activity: WALKING vs STANDING (>5s Alert)
            # -------------------------------------------------------------
            center_pt = worker.get("center", worker.get("foot_point", [0.0, 0.0]))
            cx, cy = float(center_pt[0]), float(center_pt[1])
            pos_history = mem.setdefault("pos_history", deque(maxlen=35))
            pos_history.append((current_time_sec, cx, cy))

            # Measure movement speed over ~0.25s - 0.75s window
            speed = 0.0
            if len(pos_history) >= 2:
                oldest_t, oldest_x, oldest_y = pos_history[0]
                dt = current_time_sec - oldest_t
                if dt >= 0.20:
                    dist = ((cx - oldest_x)**2 + (cy - oldest_y)**2)**0.5
                    speed = dist / dt

            is_walking = speed >= 20.0
            motion_state = "WALKING" if is_walking else "STANDING"

            if not is_walking:
                if mem["standing_start"] is None:
                    mem["standing_start"] = current_time_sec
                standing_duration = current_time_sec - mem["standing_start"]
            else:
                mem["standing_start"] = None
                standing_duration = 0.0

            is_standing_violation = standing_duration > self.standing_seconds
            worker["motion_state"] = motion_state
            worker["standing_duration"] = round(standing_duration, 1)
            worker["standing_over_5s"] = is_standing_violation

            # -------------------------------------------------------------
            # 5. Temporal Confirmation & Behavior Classification
            # -------------------------------------------------------------
            is_zone_violation = zone_duration >= self.restricted_zone_seconds
            is_helmet_violation = no_helmet_duration >= self.no_helmet_seconds
            is_vest_violation = no_vest_duration >= self.no_vest_seconds
            is_fall_violation = fall_duration >= self.fall_duration_seconds

            worker_violations: List[str] = []
            behavior_state = "NORMAL"

            # Rule 1: Fall in Restricted Zone -> CRITICAL (Highest Hazard)
            if is_fall_violation and inside_zone:
                behavior_state = "FALL_CONFIRMED"
                v_type = "FALL_CONFIRMED"
                desc = f"Person #{person_id} confirmed fall inside restricted zone {zone_id or ''}".strip()
                worker_violations.append(v_type)
                active_violations.append({
                    "person_id": person_id,
                    "violation_type": v_type,
                    "severity": "CRITICAL",
                    "duration": round(fall_duration, 1),
                    "zone_id": zone_id,
                    "confidence": round(pose_data.get("confidence", 0.90), 2) if pose_data else 0.90,
                    "timestamp": current_time_sec,
                    "description": desc
                })

            # Rule 2: Fall Suspected Outside Zone -> CRITICAL
            elif is_fall_violation:
                behavior_state = "FALL_SUSPECTED"
                v_type = "FALL_SUSPECTED"
                desc = f"Person #{person_id} possible fall detected"
                worker_violations.append(v_type)
                active_violations.append({
                    "person_id": person_id,
                    "violation_type": v_type,
                    "severity": self.severity_rules.get(v_type, "CRITICAL"),
                    "duration": round(fall_duration, 1),
                    "zone_id": zone_id,
                    "confidence": round(pose_data.get("confidence", 0.85), 2) if pose_data else 0.85,
                    "timestamp": current_time_sec,
                    "description": desc
                })

            # Rule 3: High-Risk Combination: Restricted Zone + Missing PPE
            elif is_zone_violation and (is_helmet_violation or is_vest_violation):
                behavior_state = "HIGH_RISK"
                v_type = "HIGH_RISK_PPE_VIOLATION"
                if is_helmet_violation and is_vest_violation:
                    desc = f"Person #{person_id} entered restricted zone without helmet and vest"
                    sev = "CRITICAL"
                elif is_helmet_violation:
                    desc = f"Person #{person_id} entered restricted zone without helmet"
                    sev = "HIGH"
                else:
                    desc = f"Person #{person_id} entered restricted zone without safety vest"
                    sev = "HIGH"

                worker_violations.append(v_type)
                active_violations.append({
                    "person_id": person_id,
                    "violation_type": v_type,
                    "severity": sev,
                    "duration": round(max(zone_duration, no_helmet_duration, no_vest_duration), 1),
                    "zone_id": zone_id,
                    "confidence": 0.95,
                    "timestamp": current_time_sec,
                    "description": desc
                })

            # Rule 4: Restricted Zone Entry Violation (without PPE breach)
            elif is_zone_violation:
                behavior_state = "INSIDE_RESTRICTED_ZONE"
                v_type = "RESTRICTED_ZONE_ENTRY"
                desc = f"Person #{person_id} entered restricted zone {zone_id or ''}".strip()
                worker_violations.append(v_type)
                active_violations.append({
                    "person_id": person_id,
                    "violation_type": v_type,
                    "severity": self.severity_rules.get(v_type, "HIGH"),
                    "duration": round(zone_duration, 1),
                    "zone_id": zone_id,
                    "confidence": 0.94,
                    "timestamp": current_time_sec,
                    "description": desc
                })

            # Rule 5: Missing PPE Violations Outside Zone
            elif is_helmet_violation or is_vest_violation:
                if is_helmet_violation and is_vest_violation:
                    behavior_state = "PPE_VIOLATION"
                    v_type = "PPE_VIOLATION"
                    desc = f"Person #{person_id} missing helmet and safety vest"
                    worker_violations.append(v_type)
                    active_violations.append({
                        "person_id": person_id,
                        "violation_type": v_type,
                        "severity": self.severity_rules.get(v_type, "MEDIUM"),
                        "duration": round(max(no_helmet_duration, no_vest_duration), 1),
                        "zone_id": zone_id,
                        "confidence": 0.90,
                        "timestamp": current_time_sec,
                        "description": desc
                    })
                elif is_helmet_violation:
                    behavior_state = "NO_HELMET"
                    v_type = "NO_HELMET"
                    desc = f"Person #{person_id} working without helmet"
                    worker_violations.append(v_type)
                    active_violations.append({
                        "person_id": person_id,
                        "violation_type": v_type,
                        "severity": self.severity_rules.get(v_type, "MEDIUM"),
                        "duration": round(no_helmet_duration, 1),
                        "zone_id": zone_id,
                        "confidence": 0.91,
                        "timestamp": current_time_sec,
                        "description": desc
                    })
                elif is_vest_violation:
                    behavior_state = "NO_VEST"
                    v_type = "NO_VEST"
                    desc = f"Person #{person_id} working without safety vest"
                    worker_violations.append(v_type)
                    active_violations.append({
                        "person_id": person_id,
                        "violation_type": v_type,
                        "severity": self.severity_rules.get(v_type, "MEDIUM"),
                        "duration": round(no_vest_duration, 1),
                        "zone_id": zone_id,
                        "confidence": 0.89,
                        "timestamp": current_time_sec,
                        "description": desc
                    })

            # Rule 6: Prolonged Standing Alert (> 5s outside zone)
            elif is_standing_violation and not inside_zone:
                behavior_state = "PROLONGED_STANDING"
                v_type = "PROLONGED_STANDING"
                desc = f"Person #{person_id} standing for {round(standing_duration, 1)}s (> 5s)"
                worker_violations.append(v_type)
                active_violations.append({
                    "person_id": person_id,
                    "violation_type": v_type,
                    "severity": self.severity_rules.get(v_type, "HIGH"),
                    "duration": round(standing_duration, 1),
                    "zone_id": zone_id,
                    "confidence": 0.95,
                    "timestamp": current_time_sec,
                    "description": desc
                })

            # Rule 7: Intermediate and Normal behavior states
            else:
                if inside_zone and zone_duration < self.restricted_zone_seconds:
                    behavior_state = "ENTERING_RESTRICTED_ZONE"
                elif worker.get("zone_state") == "EXITING":
                    behavior_state = "EXITING_RESTRICTED_ZONE"
                elif pose_data and pose_data.get("torso_angle_deg", 90.0) < 40.0:
                    behavior_state = "ABNORMAL_POSTURE"
                elif worker.get("helmet_present") and worker.get("vest_present"):
                    behavior_state = "NORMAL_PPE"
                else:
                    behavior_state = "NORMAL"

            worker["behavior_state"] = behavior_state
            worker["active_violations"] = worker_violations
            worker["zone_duration"] = round(zone_duration, 1)

        # Cleanup memory for disconnected workers
        active_pids = set(workers.keys())
        for pid in list(self._temporal_state.keys()):
            if pid not in active_pids:
                del self._temporal_state[pid]

        return active_violations
