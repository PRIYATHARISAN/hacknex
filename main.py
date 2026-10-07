"""
Main CLI Runner for Autonomous Vision & Behaviour Understanding (HNX26PSI07).
Usage:
    python main.py --input data/input_videos/clip.mp4
    python main.py --generate-demo   (creates synthetic test video to demonstrate all rules)
"""

import argparse
import json
import sys
from pathlib import Path

import config
from core.pipeline import VisionPipeline


def run_pipeline(input_video: str, output_video: str = None, log_file: str = None):
    input_path = Path(input_video)
    if not input_path.exists():
        print(f"[ERROR] Input video file not found: {input_path}")
        sys.exit(1)

    out_video_path = output_video or str(config.DEFAULT_ANNOTATED_VIDEO)
    out_log_path = Path(log_file) if log_file else config.DEFAULT_LOG_FILE

    print("=" * 70)
    print(" HNX26PSI07: Autonomous Vision & Behaviour Understanding")
    print("=" * 70)
    print(f" Input Video  : {input_path}")
    print(f" Output Video : {out_video_path}")
    print(f" Audit Logs   : {out_log_path}")
    print(f" Hardware     : {config.DEVICE}")
    print(f" Model        : {config.MODEL_NAME}")
    print("=" * 70)

    pipeline = VisionPipeline(
        model_name=config.MODEL_NAME,
        log_file=out_log_path,
        device=config.DEVICE,
    )

    def on_progress(frame_idx, total_frames, active_events):
        if total_frames > 0 and frame_idx % 30 == 0:
            pct = (frame_idx / total_frames) * 100
            print(f"[*] Processing: {frame_idx}/{total_frames} frames ({pct:.1f}%) | Active Alerts: {len(active_events)}")

    logger = pipeline.process_video(
        video_path=str(input_path),
        output_video_path=out_video_path,
        progress_callback=on_progress,
    )

    # Print Executive Incident Summary
    events = logger.get_all_events()
    print("\n" + "=" * 70)
    print(f" AUDIT COMPLETE: {len(events)} INCIDENTS DETECTED")
    print("=" * 70)

    for i, evt in enumerate(events, 1):
        print(f"[{i:02d}] WHO: {evt['entity_id']:<12} | WHEN: {evt['start_time']} -> {evt['end_time']} ({evt['duration_seconds']}s)")
        print(f"     WHAT: {evt['action_type']:<28} | STATUS: {evt['status']}")
        print(f"     WHY: {evt['reason']}")
        print("-" * 70)

    print(f"\n[+] Full JSON Audit Report exported to: {out_log_path}")
    print(f"[+] Rendered Video saved to: {out_video_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Vision & Behaviour Understanding CLI")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path for rendered output video")
    parser.add_argument("--log", "-l", type=str, default=None, help="Path for JSON audit log")

    args = parser.parse_args()

    if args.input:
        run_pipeline(args.input, args.output, args.log)
    else:
        # Check if any videos exist in input_videos directory
        videos = list(config.INPUT_VIDEOS_DIR.glob("*.mp4")) + list(config.INPUT_VIDEOS_DIR.glob("*.avi"))
        if videos:
            print(f"[*] Found input video: {videos[0]}")
            run_pipeline(str(videos[0]), args.output, args.log)
        else:
            print("No input video specified. Please provide a video with:")
            print("  python main.py --input path/to/video.mp4")
            print(f"Or place an MP4 video in: {config.INPUT_VIDEOS_DIR}")
