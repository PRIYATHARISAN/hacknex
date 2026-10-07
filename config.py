"""
Central Configuration for Autonomous Vision & Behaviour Understanding (HNX26PSI07).
Contains all tunable parameters, model settings, and anomaly thresholds.
Zero magic numbers inside detection loops.
"""

from pathlib import Path
import torch

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_VIDEOS_DIR = DATA_DIR / "input_videos"
OUTPUT_LOGS_DIR = DATA_DIR / "output_logs"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
INPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hardware & Model Settings
MODEL_NAME = "yolo26m-pose.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
TRACKER_CONFIG = "bytetrack.yaml"

# Temporal & Action Analysis Settings
SLIDING_WINDOW_FRAMES = 30  # Sliding window size for velocity & angle smoothing
FPS_FALLBACK = 30.0         # Fallback FPS if video metadata cannot read it

# Kinematic Thresholds
# Velocities in pixels per second (normalized against bounding-box height)
STATIONARY_SPEED_THRESHOLD = 50.0      # Below this is considered stationary/idle (accounts for CCTV distance)
RUNNING_SPEED_THRESHOLD = 150.0        # Above this is classified as running/panic

# Posture & Fall Detection Thresholds
# Angle in degrees relative to ground horizontal (0 deg = flat on ground, 90 deg = standing upright)
FALL_TORSO_ANGLE_MAX = 35.0            # Angle <= 35 deg indicates horizontal/prone posture
FALL_ASPECT_RATIO_MIN = 0.95           # Width / Height >= 0.95 indicates prone/horizontal body

# Anomaly Rules (Normal vs Abnormal)
LOITERING_DURATION_SECONDS = 5.0       # Stationary in one spot > 5.0s is Abnormal (calibrated for 20-30s clips)
FALL_CONFIRMATION_SECONDS = 2.0        # Prone on floor > 2.0s is Abnormal (Man Down)
RUNNING_CONFIRMATION_SECONDS = 1.0     # Sprinting/panic > 1.0s is Abnormal

# Output & Logging
DEFAULT_LOG_FILE = OUTPUT_LOGS_DIR / "events.json"
DEFAULT_ANNOTATED_VIDEO = OUTPUT_LOGS_DIR / "annotated_output.mp4"
