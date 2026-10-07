# Autonomous Workplace Safety Behaviour Understanding System
### Hackathon Problem Statement: HNX26PSI07 — Autonomous Vision & Behaviour Understanding
**Scenario Focus: Workplace Safety — Warehouse & Factory Environments**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://docs.ultralytics.com/)
[![ByteTrack](https://img.shields.io/badge/tracking-ByteTrack-brightgreen.svg)](https://github.com/ifzhang/ByteTrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Overview

Traditional computer-vision applications in industrial monitoring stop at basic per-frame object detection. In real-world warehouses and manufacturing plants, detection alone fails to answer the critical questions: **WHO did WHAT, WHERE, and WHEN?**

This system delivers a production-grade, temporal reasoning architecture that tracks workers persistently, evaluates spatial geofencing around dangerous heavy machinery, verifies PPE compliance, inspects biomechanical posture for slip/trip/fall events, and generates deduplicated, timestamped safety events with severity escalation.

```
       ┌────────────────────────────────────────────────────────┐
       │   Video Stream (Warehouse / Factory CCTV / Webcam)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     YOLO11m Detection (Workers, Machinery, Objects)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │    ByteTrack Multi-Object Tracker (Persistent IDs)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
 ┌───────────────┐        ┌──────────────────┐       ┌─────────────────┐
 │ Polygonal     │        │ Custom Workplace │       │ YOLO11s-Pose    │
 │ Geofencing    │        │ PPE Association  │       │ Biomechanics    │
 │ (Foot Contact)│        │ (Helmet & Vest)  │       │ (Fall Detection)│
 └──────┬────────┘        └────────┬─────────┘       └────────┬────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     Temporal Behavior Engine & Worker State Machine    │
       │    (Noise Rejection, Temporal Persistence Hysteresis)  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │      Event Manager (Deduplication, Cooldown, Severity) │
       │                -> JSON & CSV Event Stores              │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     High-Aesthetic HUD Visualizer & Real-Time Dashboard │
       │               -> Annotated Production MP4              │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Key Architecture & Features

| Capability | Implementation | Why It Matters |
|---|---|---|
| **Person Detection** | Ultralytics YOLO11m | Detects workers across varying warehouse lighting and occlusions. |
| **Persistent Tracking** | ByteTrack | Prevents ID switching and maintains per-worker spatial history across frames. |
| **Zone Understanding** | Polygonal Geofencing (OpenCV / Ray-Casting) | Uses **foot contact ground points** rather than box centers to prevent false alarms. |
| **PPE Association** | Anatomical Head & Torso Overlap | Resolves individual worker compliance: a helmet in the scene is only credited if worn on that specific worker's head. |
| **Pose & Fall Detection** | YOLO11s-Pose & Biomechanical Angle Analysis | Detects horizontal torso alignment (<35°) and aspect ratio collapse indicating falls. |
| **Temporal Reasoning** | Multi-state Worker State Machine | Requires persistence (e.g. 2.0s inside zone) before confirming violations; ignores 1-frame glitches. |
| **Event Deduplication** | Lifecycle Hysteresis with Cooldown | Groups a 100-frame continuous violation into a single consolidated safety incident with start, end, and duration. |
| **Real-Time HUD Dashboard** | OpenCV Custom UI Engine | Displays live metrics, worker compliance counters, pipeline latency benchmarks, and active alerts. |

---

## 3. Project Structure

```text
workplace_safety_ai/
│
├── main.py                   # CLI entry point & processing pipeline coordinator
├── config.yaml               # System parameters, timing thresholds, & visual configs
├── requirements.txt          # Python dependencies
├── README.md                 # Complete documentation & benchmark manual
│
├── models/                   # Neural network weights
│   ├── yolo11m.pt            # Object & worker detector
│   ├── yolo11s-pose.pt       # Pose estimator & fall analyzer
│   └── workplace_ppe.pt      # Custom workplace PPE model (optional / fallback handled)
│
├── src/                      # Modular production codebase
│   ├── __init__.py           # Package namespace definition
│   ├── detector.py           # Clean YOLO11m object detector
│   ├── tracker.py            # ByteTrack worker tracking & persistent ID registry
│   ├── zone_manager.py       # Polygonal zone geofencing (foot contact point)
│   ├── ppe_manager.py        # Worker-PPE anatomical association
│   ├── pose_estimator.py     # Human pose estimation & torso posture / fall analysis
│   ├── behavior_engine.py    # Temporal state machine & anomaly classifier
│   ├── event_manager.py      # Incident lifecycle, deduplication, JSON/CSV exports
│   ├── visualizer.py         # High-aesthetic HUD overlay, alert banners, & dashboard
│   └── utils.py              # Benchmarking profiler, IoU, logging, & geometry utils
│
├── configs/
│   └── zones.yaml            # Polygonal danger zones (coordinates & severity)
│
├── input/
│   └── sample.mp4            # Sample warehouse footage
│
├── output/
│   ├── annotated/            # Rendered MP4 output videos with dashboard overlays
│   └── events/               # Persisted JSON and CSV incident logs
│
├── logs/
│   └── application.log       # Production file & console audit log
│
└── tests/                    # GPU-free automated test suite
    ├── __init__.py
    ├── test_zone_manager.py
    ├── test_behavior_engine.py
    └── test_event_manager.py
```

---

## 4. Safety Event Specification

Every safety incident captured by the system adheres to a strict schema:

```json
{
  "event_id": "EVT-0001",
  "event_type": "HIGH_RISK_SAFETY_VIOLATION",
  "person_id": 7,
  "timestamp": "00:49.2",
  "start_time": 47.0,
  "end_time": 52.4,
  "duration": 5.4,
  "confidence": 0.95,
  "severity": "CRITICAL",
  "zone": "restricted_zone_1",
  "status": "CLOSED"
}
```

### Violation Severity Matrix
* **`NORMAL`**: Worker fully compliant, outside restricted perimeter, wearing PPE. (No safety incident generated).
* **`MEDIUM`**: `NO_HELMET` or `NO_VEST` sustained for > 2.0s in general areas.
* **`HIGH`**: `RESTRICTED_ZONE_VIOLATION` (worker inside perimeter for > 2.0s).
* **`CRITICAL`**: `HIGH_RISK_SAFETY_VIOLATION` (worker inside restricted zone **and** missing helmet) or `FALL_SUSPECTED`.

---

## 5. Installation & Setup

### Prerequisites
* Python 3.10 or higher
* CUDA-compatible GPU (optional, automatically defaults to CPU if GPU is unavailable)

### 1. Clone & Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Weights Setup
Place weights inside the `models/` folder:
* **`models/yolo11m.pt`**: Primary detector (auto-downloads if absent).
* **`models/yolo11s-pose.pt`**: Human pose estimator for fall detection.
* **`models/workplace_ppe.pt`**: Custom trained workplace safety PPE model.

> **Graceful Degradation Note:** If `models/workplace_ppe.pt` is missing, the application outputs:
> ```text
> WARNING: Custom PPE model not found. PPE detection disabled.
> ```
> and continues executing tracking, zone monitoring, posture analysis, and event logging without crashing.

---

## 6. How to Run

### Run on Video File
```bash
python main.py --input input/sample.mp4
```

### Run with Custom Configuration or Output
```bash
python main.py --input input/sample.mp4 --config config.yaml --output output/annotated/warehouse_audit.mp4
```

### Run on Live Camera / USB Stream
```bash
python main.py --source 0
```

### Run in Headless / Server Mode (No GUI)
```bash
python main.py --input input/sample.mp4 --no-display
```

### Run Automated Unit Tests (CPU Only)
```bash
python -m unittest discover tests
```

---

## 7. Configuration Guide (`config.yaml`)

All timing thresholds and visual preferences are configurable without touching code:

```yaml
models:
  detector: "models/yolo11m.pt"
  pose: "models/yolo11s-pose.pt"
  ppe: "models/workplace_ppe.pt"

tracking:
  tracker: "bytetrack.yaml"
  confidence: 0.35
  iou: 0.5

video:
  input: "input/sample.mp4"
  output: "output/annotated/output.mp4"

zones_file: "configs/zones.yaml"

behavior:
  restricted_zone_seconds: 2.0    # Duration required to trigger zone violation
  no_helmet_seconds: 2.0          # Duration required to trigger missing helmet
  no_vest_seconds: 2.0            # Duration required to trigger missing vest
  event_cooldown_seconds: 5.0     # Time before re-triggering closed violation
  fall_duration_seconds: 1.5      # Sustained horizontal posture before fall event

display:
  show_tracking: true
  show_zones: true
  show_ppe: true
  show_behavior: true
  show_events: true
  show_fps: true
  show_dashboard: true

performance:
  image_size: 640
  pose_every_n_frames: 2          # Runs pose inference on alternate frames
  device: "auto"                  # 'auto', 'cuda', or 'cpu'
```

---

## 8. Hackathon Relevance (HNX26PSI07)

| Problem Statement Pillar | System Implementation |
|---|---|
| **Autonomous Vision** | End-to-end processing pipeline requiring zero manual operator intervention from ingestion to alert generation. |
| **Multi-Worker Persistent Tracking** | ByteTrack maintains track histories across occlusions and dynamic worker movements. |
| **Spatial Anomaly Detection** | Polygonal danger zones using accurate foot-ground contact geofencing. |
| **Object / Worker Association** | Spatial intersection between detected PPE and worker anatomical regions (head / torso). |
| **Temporal Behavior Understanding** | Temporal persistence state machines eliminating false positives from single-frame noise. |
| **Person-Specific Events** | Precise time-stamped logging specifying exact Worker ID, location, incident duration, and severity level. |
