"""
Workplace Safety AI - Main Orchestration Pipeline
HNX26PSI07: Autonomous Workplace Safety Vision & Behaviour Understanding
"""

import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np

from src.utils import setup_logger, load_yaml_config, LatencyBenchmark, seconds_to_timestamp
from src.detector import ObjectDetector
from src.tracker import WorkerTracker
from src.zone_manager import ZoneManager
from src.ppe_manager import PPEManager
from src.pose_estimator import PoseEstimator
from src.behavior_engine import BehaviorEngine
from src.event_manager import EventManager
from src.visualizer import SafetyVisualizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Workplace Safety Vision & Behaviour Understanding System"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input video file (e.g. input/sample.mp4)"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Input source: video file path or webcam index (e.g. '0')"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save annotated output video (overrides config.yaml)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without opening GUI window"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (optional, for benchmarking)"
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_arguments()

    # 1. Initialize Logging
    logger = setup_logger(log_file="logs/application.log", level=logging.INFO)
    logger.info("=" * 60)
    logger.info("INITIALIZING AUTONOMOUS WORKPLACE SAFETY MONITOR")
    logger.info("=" * 60)

    # 2. Load Configuration
    config_path = args.config
    if not os.path.isfile(config_path):
        logger.error("Configuration file not found at: %s", config_path)
        return 1

    try:
        config = load_yaml_config(config_path)
        logger.info("Configuration successfully loaded from: %s", config_path)
    except Exception as err:
        logger.error("Failed to parse configuration: %s", err)
        return 1

    # Extract config sections
    models_cfg = config.get("models", {})
    tracking_cfg = config.get("tracking", {})
    video_cfg = config.get("video", {})
    behavior_cfg = config.get("behavior", {})
    display_cfg = config.get("display", {})
    output_cfg = config.get("output", {})
    perf_cfg = config.get("performance", {})
    severity_rules = config.get("severity_rules", {})
    zones_file = config.get("zones_file", "configs/zones.yaml")

    # 3. Determine Input Source & Output Path
    input_source = args.source or args.input or video_cfg.get("input", "input/sample.mp4")
    output_path = args.output or video_cfg.get("output", "output/annotated/result.mp4")

    # Parse numeric webcam indices
    if isinstance(input_source, str) and input_source.isdigit():
        video_source_arg = int(input_source)
        is_live_stream = True
        logger.info("Input source configured as live webcam device #%d", video_source_arg)
    else:
        video_source_arg = input_source
        is_live_stream = False
        if not os.path.isfile(video_source_arg):
            logger.error("Specified video file does not exist: %s", video_source_arg)
            logger.info("Tip: Place video at input/sample.mp4 or pass --input path/to/video.mp4")
            return 1
        logger.info("Input source configured as video file: %s", video_source_arg)

    # 4. Initialize Pipeline Components
    try:
        # A. Object Detector
        detector = ObjectDetector(
            model_path=models_cfg.get("detection", models_cfg.get("detector", "models/yolo11m.pt")),
            confidence=tracking_cfg.get("confidence", 0.35),
            image_size=perf_cfg.get("image_size", 640),
            device=perf_cfg.get("device", "auto"),
            logger=logger
        )

        # B. Worker Tracker
        tracker = WorkerTracker(
            model=detector.model,
            tracker_config=tracking_cfg.get("tracker", "bytetrack.yaml"),
            confidence=tracking_cfg.get("confidence", 0.35),
            iou_threshold=tracking_cfg.get("iou", 0.5),
            logger=logger
        )

        # C. Zone Manager
        zone_manager = ZoneManager(
            zones_config_path=zones_file,
            logger=logger
        )

        # D. PPE Manager
        ppe_manager = PPEManager(
            model_path=models_cfg.get("ppe", "models/workplace_ppe.pt"),
            confidence=0.40,
            logger=logger
        )

        # E. Pose Estimator
        pose_estimator = PoseEstimator(
            model_path=models_cfg.get("pose", "models/yolo11s-pose.pt"),
            confidence=0.35,
            pose_every_n_frames=perf_cfg.get("pose_every_n_frames", 2),
            device=perf_cfg.get("device", "auto"),
            logger=logger
        )

        # F. Temporal Behavior Engine
        behavior_engine = BehaviorEngine(
            restricted_zone_seconds=behavior_cfg.get("restricted_zone_min_duration", behavior_cfg.get("restricted_zone_seconds", 1.5)),
            no_helmet_seconds=behavior_cfg.get("no_helmet_seconds", 2.0),
            no_vest_seconds=behavior_cfg.get("no_vest_seconds", 2.0),
            fall_duration_seconds=behavior_cfg.get("fall_persistence", behavior_cfg.get("fall_duration_seconds", 1.0)),
            standing_seconds=behavior_cfg.get("standing_seconds", 5.0),
            severity_rules=severity_rules,
            logger=logger
        )

        # G. Safety Event Manager
        event_manager = EventManager(
            cooldown_seconds=behavior_cfg.get("event_cooldown", behavior_cfg.get("event_cooldown_seconds", 5.0)),
            output_dir=os.path.dirname(output_cfg.get("events_json", "output/events/events.json")) or "output/events",
            logger=logger
        )

        # H. Safety Visualizer & Benchmarking
        visualizer = SafetyVisualizer(display_config=display_cfg, logger=logger)
        benchmark = LatencyBenchmark(smoothing_window=25)

    except Exception as exc:
        logger.critical("Fatal error initializing pipeline modules: %s", exc, exc_info=True)
        return 1

    # 5. Open Video Stream
    cap = cv2.VideoCapture(video_source_arg)
    if not cap.isOpened():
        logger.error("Failed to open video source: %s", video_source_arg)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Stream opened: %dx%d @ %.1f FPS | Total Frames: %s",
        frame_width, frame_height, fps, total_frames if total_frames > 0 else "Unknown"
    )

    # 6. Initialize Video Writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not video_writer.isOpened():
        logger.warning("Could not open VideoWriter for %s. Output video will not be saved.", output_path)
        video_writer = None
    else:
        logger.info("Saving annotated stream to: %s", output_path)

    # 7. Processing Loop
    frame_idx = 0
    start_wall_time = time.time()
    logger.info("Commencing processing loop...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("Reached end of video stream.")
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                logger.info("Reached maximum frames limit (%d).", args.max_frames)
                break

            benchmark.start_frame()
            current_time_sec = (time.time() - start_wall_time) if is_live_stream else (frame_idx / fps)

            # Stage 1: Detection & Tracking
            t0 = time.perf_counter()
            workers = tracker.update(frame, current_time_sec)
            t1 = time.perf_counter()
            benchmark.record_stage("tracking", t1 - t0)
            benchmark.record_stage("detection", (t1 - t0) * 0.8)

            # Stage 2: Restricted Zone Evaluation
            zone_manager.update_workers_zones(workers, frame_width, frame_height)

            # Stage 3: PPE Detection & Association
            if ppe_manager.is_available:
                ppe_detections = ppe_manager.detect_ppe(frame)
                ppe_manager.associate_ppe(workers, ppe_detections)
            else:
                ppe_manager.associate_ppe(workers, [])

            # Stage 4: Pose Estimation & Biomechanical Fall Detection
            t_pose0 = time.perf_counter()
            pose_estimator.estimate(frame, workers)
            t_pose1 = time.perf_counter()
            benchmark.record_stage("pose", t_pose1 - t_pose0)

            # Stage 5: Temporal Behavior Understanding & State Machine
            active_violations = behavior_engine.update(workers, current_time_sec)

            # Stage 6: Event Management & Deduplication
            event_manager.process_violations(active_violations, current_time_sec)

            # Stage 7: Visualization & HUD Dashboard Overlay
            # Dynamically compute safety status from active behavioral violations
            has_crit = (
                any(w.get("inside_restricted_zone") for w in workers.values())
                or any(w.get("standing_over_5s") for w in workers.values())
                or any(w.get("behavior_state") in ("HIGH_RISK", "FALL_SUSPECTED", "FALL_CONFIRMED", "PROLONGED_STANDING") for w in workers.values())
                or any(v.get("severity") in ("CRITICAL", "HIGH") for v in active_violations)
            )
            has_warn = (
                any(w.get("behavior_state") in ("ENTERING_RESTRICTED_ZONE", "NO_HELMET", "NO_VEST", "PPE_VIOLATION", "WARNING") for w in workers.values())
                or any(v.get("severity") == "MEDIUM" for v in active_violations)
            )
            overall_status = "CRITICAL" if has_crit else ("WARNING" if has_warn else "SAFE")

            if display_cfg.get("show_zones", True):
                frame = zone_manager.draw_zones(frame, workers=workers)

            frame = visualizer.draw_worker_annotations(frame, workers)
            frame = visualizer.draw_active_alert_banner(frame, active_violations, workers=workers)
            frame = visualizer.draw_top_status_bar(frame, overall_status=overall_status)

            recent_events = event_manager.get_recent_events(limit=4)
            frame = visualizer.draw_safety_dashboard(
                frame,
                workers=workers,
                recent_events=recent_events,
                fps=benchmark.get_fps() or fps,
                overall_status=overall_status,
                active_violations=active_violations
            )

            # Write annotated frame
            if video_writer is not None:
                video_writer.write(frame)

            # Display GUI window unless --no-display is set
            if not args.no_display:
                win_name = "Workplace Safety AI Monitor"
                disp_w = display_cfg.get("window_width", 1280)
                disp_h = display_cfg.get("window_height", 720)
                disp_frame = cv2.resize(frame, (disp_w, disp_h))
                cv2.imshow(win_name, disp_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    logger.info("User termination requested via keyboard interrupt.")
                    break

            if frame_idx % 60 == 0:
                logger.info(
                    "Processed Frame #%d [%s] | Active Workers: %d | Active Violations: %d | FPS: %.1f",
                    frame_idx, seconds_to_timestamp(current_time_sec), len(workers),
                    len(active_violations), benchmark.get_fps()
                )

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received.")
    except Exception as exc:
        logger.error("Unexpected error in main processing loop: %s", exc, exc_info=True)
    finally:
        # Final cleanup and report generation
        final_video_time = frame_idx / fps
        event_manager.close_all(final_video_time)
        json_target = os.path.basename(output_cfg.get("events_json", "events.json"))
        csv_target = os.path.basename(output_cfg.get("events_csv", "events.csv"))
        json_file, csv_file = event_manager.save_events(json_filename=json_target, csv_filename=csv_target)

        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

        elapsed_total = time.time() - start_wall_time
        avg_fps = frame_idx / max(0.001, elapsed_total)

        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info("Total Frames Processed : %d", frame_idx)
        logger.info("Total Processing Time   : %.2f seconds", elapsed_total)
        logger.info("Average Processing FPS  : %.1f", avg_fps)
        logger.info("Total Safety Events     : %d", len(event_manager.get_all_events()))
        logger.info("Annotated Video Saved   : %s", output_path)
        logger.info("JSON Safety Events Log  : %s", json_file)
        logger.info("CSV Safety Events Log   : %s", csv_file)
        logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
