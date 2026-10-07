"""
Pipeline Orchestrator Module.
Connects Tracker -> ActionEngine -> AnomalyEvaluator -> EventLogger -> Visual Renderer.
"""

from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional
import cv2
import numpy as np
import subprocess

import config
from core.action_engine import ActionEngine, EntityKinematics
from core.anomaly_evaluator import AnomalyEvaluator
from core.attribution import EventLogger, frame_to_timestamp
from core.tracker import PoseTracker, TrackedEntity

# Standard COCO 17-keypoint skeleton connections for visualization
SKELETON_PAIRS = [
    (5, 6),   # L shoulder - R shoulder
    (5, 7),   # L shoulder - L elbow
    (7, 9),   # L elbow - L wrist
    (6, 8),   # R shoulder - R elbow
    (8, 10),  # R elbow - R wrist
    (5, 11),  # L shoulder - L hip
    (6, 12),  # R shoulder - R hip
    (11, 12), # L hip - R hip
    (11, 13), # L hip - L knee
    (13, 15), # L knee - L ankle
    (12, 14), # R hip - R knee
    (14, 16), # R knee - R ankle
]


class VisionPipeline:
    """
    Main execution pipeline for Autonomous Vision & Behaviour Understanding.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        log_file: Path = config.DEFAULT_LOG_FILE,
        device: str = config.DEVICE,
    ):
        self.tracker = PoseTracker(model_name=model_name, device=device)
        self.action_engine = ActionEngine(
            window_size=config.SLIDING_WINDOW_FRAMES,
            stationary_threshold=config.STATIONARY_SPEED_THRESHOLD,
            running_threshold=config.RUNNING_SPEED_THRESHOLD,
            fall_angle_max=config.FALL_TORSO_ANGLE_MAX,
            fall_aspect_ratio_min=config.FALL_ASPECT_RATIO_MIN,
        )
        self.anomaly_evaluator = AnomalyEvaluator(
            loitering_threshold_sec=config.LOITERING_DURATION_SECONDS,
            fall_confirm_sec=config.FALL_CONFIRMATION_SECONDS,
            running_confirm_sec=config.RUNNING_CONFIRMATION_SECONDS,
        )
        self.event_logger = EventLogger(log_path=log_file)

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, List[Dict]], None]] = None,
    ) -> EventLogger:
        """
        Runs the end-to-end pipeline on an input video clip.
        Saves annotated video and exports event audit log.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = config.FPS_FALLBACK
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video_path:
            out_p = Path(output_video_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                timestamp_sec = frame_idx / fps

                # 1. Unified Detection, Tracking, and Pose Estimation
                entities: List[TrackedEntity] = self.tracker.track_frame(frame)
                active_ids = [e.track_id for e in entities]

                frame_active_events: List[Dict] = []

                # 2. Kinematics, Anomaly Evaluation & Rendering per Entity
                for entity in entities:
                    self.event_logger.register_seen_entity(entity.track_id)
                    kinematics = self.action_engine.update_entity(
                        entity_id=entity.track_id,
                        bbox=entity.bbox,
                        keypoints=entity.keypoints,
                        timestamp_sec=timestamp_sec,
                    )

                    is_abnormal, anomaly_types, reasons, box_color = self.anomaly_evaluator.evaluate(
                        kinematics=kinematics,
                        timestamp_sec=timestamp_sec,
                    )

                    # 3. Attribution Logging
                    if is_abnormal:
                        for a_type, reason in zip(anomaly_types, reasons):
                            self.event_logger.trigger_or_update(
                                entity_id=entity.track_id,
                                action_type=a_type,
                                frame_idx=frame_idx,
                                fps=fps,
                                reason=reason,
                                bbox=kinematics.bbox,
                                centroid=kinematics.centroid,
                            )
                            frame_active_events.append({
                                "entity": entity.track_id,
                                "type": a_type,
                                "reason": reason,
                            })
                    else:
                        self.event_logger.finalize_if_resolved(
                            entity_id=entity.track_id,
                            active_actions=[],
                            current_frame=frame_idx,
                            fps=fps,
                        )

                    # 4. Visual Rendering
                    self._render_entity(frame, kinematics, entity.keypoints, is_abnormal, box_color)

                # 5. Purge lost entities
                self.action_engine.purge_lost_entities(active_ids)
                self.anomaly_evaluator.purge_lost_entities(active_ids)

                # 6. Render Global HUD
                self._render_hud(frame, frame_idx, fps, total_frames, len(entities), len(frame_active_events))

                if writer:
                    writer.write(frame)

                if progress_callback:
                    progress_callback(frame_idx, total_frames, self.event_logger.get_all_events())

        finally:
            cap.release()
            if writer:
                writer.release()
                if output_video_path and Path(output_video_path).exists():
                    self._convert_to_web_h264(output_video_path)
            self.event_logger.close_all()
            self.event_logger.save_to_json()

        print(f"[Pipeline] Finished. Processed {frame_idx} frames. Events saved to {self.event_logger.log_path}")
        return self.event_logger

    def _convert_to_web_h264(self, video_path: str):
        """Converts an OpenCV-generated MP4 to browser-friendly H.264 using FFmpeg."""
        import os
        import shutil
        import time

        src_path = Path(video_path).resolve()
        temp_web_path = src_path.with_name(f"web_{src_path.name}")
        time.sleep(0.3)

        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(src_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(temp_web_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and temp_web_path.exists() and temp_web_path.stat().st_size > 0:
                for _ in range(6):
                    try:
                        os.replace(str(temp_web_path), str(src_path))
                        print(f"[Pipeline] Output video successfully converted to browser-ready H.264.")
                        return
                    except PermissionError:
                        time.sleep(0.4)
                print(f"[Pipeline] Warning: Could not replace destination due to file lock.")
            elif res.returncode != 0:
                print(f"[Pipeline] Warning: FFmpeg could not create browser-compatible video: {res.stderr[-500:]}")
        except Exception as e:
            print(f"[Pipeline] Note: Could not re-encode with ffmpeg ({e}), using default video.")

    def _render_entity(
        self,
        frame: np.ndarray,
        kinematics: EntityKinematics,
        keypoints: Optional[np.ndarray],
        is_abnormal: bool,
        box_color: tuple,
    ):
        """Draws bounding box, skeletal connections, and informational badges."""
        x1, y1, x2, y2 = kinematics.bbox

        # 1. Bounding Box
        thickness = 3 if is_abnormal else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # 2. Draw Pose Skeleton (if keypoints exist)
        if keypoints is not None and len(keypoints) >= 13:
            for pt1_idx, pt2_idx in SKELETON_PAIRS:
                if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                    p1 = keypoints[pt1_idx]
                    p2 = keypoints[pt2_idx]
                    # Check confidence if available
                    c1 = p1[2] if len(p1) > 2 else 1.0
                    c2 = p2[2] if len(p2) > 2 else 1.0
                    if c1 > 0.3 and c2 > 0.3:
                        c_pt1 = (int(p1[0]), int(p1[1]))
                        c_pt2 = (int(p2[0]), int(p2[1]))
                        cv2.line(frame, c_pt1, c_pt2, (255, 200, 0), 2)
                        cv2.circle(frame, c_pt1, 3, (0, 255, 255), -1)
                        cv2.circle(frame, c_pt2, 3, (0, 255, 255), -1)

        # 3. Label Badge Header
        status_text = "ABNORMAL" if is_abnormal else "NORMAL"
        badge_text = f"{kinematics.entity_id} | {kinematics.current_action} [{status_text}]"
        if kinematics.is_stationary and kinematics.stationary_duration_sec > 0:
            badge_text += f" ({kinematics.stationary_duration_sec:.1f}s)"

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - th - 10)
        bg_y2 = y1
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, bg_y2), box_color, -1)
        cv2.putText(
            frame,
            badge_text,
            (x1 + 5, bg_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255) if is_abnormal else (0, 0, 0),
            2,
        )

    def _render_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        total_frames: int,
        active_count: int,
        anomaly_count: int,
    ):
        """Renders executive heads-up display at the top of the frame."""
        now_ts = frame_to_timestamp(frame_idx, fps)
        hud_bg = (20, 20, 20)
        cv2.rectangle(frame, (10, 10), (450, 75), hud_bg, -1)
        cv2.rectangle(frame, (10, 10), (450, 75), (80, 80, 80), 1)

        cv2.putText(
            frame,
            f"TIME: {now_ts} | FRAME: {frame_idx}/{total_frames}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            frame,
            f"ENTITIES: {active_count} | ACTIVE ALERTS: {anomaly_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255) if anomaly_count > 0 else (0, 255, 0),
            2,
        )
