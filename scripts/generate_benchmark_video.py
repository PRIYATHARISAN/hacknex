"""
Benchmark Video Generator for Autonomous Vision & Behaviour Understanding.
Extracts real human subjects from sample assets and simulates a warehouse scenario featuring:
1. Person #1 (Normal): Walking along the transit corridor at steady pace.
2. Person #2 (Abnormal): Stops in corridor and loiters motionless for > 12 seconds.
3. Person #3 (Abnormal): Sudden slip & collapse to the ground (man down) > 4 seconds.
4. Person #4 (Abnormal): Sprints in panic across the aisle at high speed.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import ultralytics
import config

def create_benchmark_video(output_path: str = None, duration_sec: int = 18, fps: int = 25):
    if output_path is None:
        output_path = str(config.INPUT_VIDEOS_DIR / "benchmark_warehouse_safety.mp4")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load real human sprite from ultralytics assets
    assets_dir = Path(ultralytics.__file__).parent / "assets"
    bus_path = assets_dir / "bus.jpg"
    if not bus_path.exists():
        raise FileNotFoundError(f"Asset not found at {bus_path}")

    bus_img = cv2.imread(str(bus_path))
    # Person 1 in bus.jpg is at bbox [48, 400, 243, 902] (H=502, W=195)
    person_crop = bus_img[400:902, 48:243]
    orig_h, orig_w = person_crop.shape[:2]

    # Target video specs (720p: 1280x720)
    vw, vh = 1280, 720
    target_person_h = 240
    target_person_w = int(orig_w * (target_person_h / orig_h))
    person_sprite = cv2.resize(person_crop, (target_person_w, target_person_h))

    # Rotated prone sprite for fall detection (horizontal: 90 deg clockwise)
    fallen_sprite = cv2.rotate(person_sprite, cv2.ROTATE_90_CLOCKWISE)
    fallen_h, fallen_w = fallen_sprite.shape[:2]

    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (vw, vh))

    print(f"[*] Generating benchmark video: {out_file} ({total_frames} frames, {fps} FPS)...")

    # Background: Warehouse Corridor (Floor with perspective lines and safety lane markings)
    bg_base = np.zeros((vh, vw, 3), dtype=np.uint8)
    # Wall / ceiling
    bg_base[0:240, :] = [45, 42, 40]
    # Concrete floor
    bg_base[240:, :] = [70, 75, 80]
    # Yellow hazard safety lines on floor
    cv2.line(bg_base, (100, 240), (0, 720), (0, 215, 255), 4)
    cv2.line(bg_base, (1180, 240), (1280, 720), (0, 215, 255), 4)
    # Transit lane markings
    cv2.putText(bg_base, "TRANSIT LANE A-04 [HAZARD ZONE]", (420, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
    cv2.putText(bg_base, "RESTRICTED PEDESTRIAN WALKWAY", (430, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Actor 1: Normal Walker (starts left at frame 0, walks right smoothly)
    # Actor 2: Loiterer (walks in at frame 0, stops at center frame 50, remains still until frame 380)
    # Actor 3: Fall Incident (walks in from top-right, falls prone at frame 150, remains motionless until end)
    # Actor 4: Running Panic (appears at frame 320, sprints rapidly across at 3.5x speed)

    for f in range(total_frames):
        frame = bg_base.copy()
        t = f / fps

        # --- Actor 1: Normal Walker (steady 50 px/sec) ---
        a1_x = int(60 + (t * 60))
        a1_y = 360
        if 0 <= a1_x < vw - target_person_w:
            _overlay_sprite(frame, person_sprite, a1_x, a1_y)

        # --- Actor 2: Loiterer (walks in, stops at x=550 from t=2.0s onwards) ---
        if t < 2.0:
            a2_x = int(450 + (t * 50))
        else:
            a2_x = 550  # Stationary in one spot for over 12 seconds!
        a2_y = 350
        if 0 <= a2_x < vw - target_person_w:
            _overlay_sprite(frame, person_sprite, a2_x, a2_y)

        # --- Actor 3: Fall Accident (falls at t=5.0s, stays prone for > 8s) ---
        if t < 5.0:
            a3_x = int(880 - (t * 20))
            a3_y = 360
            _overlay_sprite(frame, person_sprite, a3_x, a3_y)
        else:
            # Person collapsed flat on floor
            a3_x = 780
            a3_y = 480  # Ground level
            _overlay_sprite(frame, fallen_sprite, a3_x, a3_y)

        # --- Actor 4: Running / Panic (starts at t=12.0s, sprints 300 px/sec) ---
        if t >= 12.0:
            run_t = t - 12.0
            a4_x = int(1200 - (run_t * 280))
            a4_y = 340
            if 0 <= a4_x < vw - target_person_w:
                _overlay_sprite(frame, person_sprite, a4_x, a4_y)

        # Add camera timestamp banner
        cv2.putText(
            frame,
            f"CCTV CAM-02 [NORTH BAY] | {t:05.2f}s | FPS: {fps}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    writer.release()
    print(f"[+] Benchmark video successfully saved to: {out_file}")
    return str(out_file)


def _overlay_sprite(background, sprite, x, y):
    """Blends a sprite onto the background with boundary clipping."""
    sh, sw = sprite.shape[:2]
    bh, bw = background.shape[:2]

    # Clip coordinates
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + sw)
    y2 = min(bh, y + sh)

    sx1 = max(0, -x)
    sy1 = max(0, -y)
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    if x2 > x1 and y2 > y1 and sx2 > sx1 and sy2 > sy1:
        background[y1:y2, x1:x2] = sprite[sy1:sy2, sx1:sx2]


if __name__ == "__main__":
    create_benchmark_video()
