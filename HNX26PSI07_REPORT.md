# HNX26PSI07: Autonomous Vision & Behaviour Understanding

## Project Report

**Domain:** Computer Vision · Action Recognition · Object Tracking · Behaviour Analysis  
**Chosen scenario:** Smart workplace and warehouse safety monitoring

---

## 1. Problem Statement

Finding objects in a video is relatively easy. Understanding what those objects are doing, whether their behavior is normal, and when their behavior becomes dangerous is more difficult.

The objective of this project is to build a computer vision system that:

- Detects people in workplace or warehouse video.
- Tracks each person consistently as they move through the scene.
- Understands physical behavior instead of only reporting object labels.
- Separates normal behavior from safety-related abnormal behavior.
- Produces meaningful, explainable events with the responsible entity and exact timing.
- Presents the result through an annotated video and an incident audit dashboard.

For example, a person walking through a warehouse is normal. A person remaining stationary for a prolonged period, running unexpectedly, or falling and remaining on the floor may require attention.

---

## 2. Proposed Solution

The project implements an end-to-end warehouse safety monitoring pipeline. Each video frame passes through detection, tracking, pose analysis, behavior classification, anomaly evaluation, event attribution, and visual rendering.

The system uses a YOLO pose model to detect people and extract 17 human body keypoints. ByteTrack maintains persistent identities, allowing the system to distinguish one person from another across consecutive frames.

Instead of generating only labels such as `Person`, the system calculates motion and posture measurements for every tracked entity:

- Centroid movement and velocity.
- Torso inclination angle.
- Bounding-box aspect ratio.
- Stationary dwell duration.
- Persistent track identity.

These measurements are converted into understandable action states and evaluated by transparent safety rules.

---

## 3. System Architecture

### Input

The system accepts uploaded or locally stored workplace and warehouse video files, including MP4, AVI, MOV, and MKV formats.

### Perception and Tracking

- YOLO pose detection identifies people.
- Bounding boxes locate each person.
- Seventeen skeletal keypoints describe body posture.
- ByteTrack assigns persistent IDs such as `Person #01` and `Person #02`.

### Behaviour Understanding

The action engine maintains a temporal history for each tracked person. It uses the history to estimate movement speed, posture, and time spent stationary.

### Anomaly Evaluation

The anomaly evaluator applies configurable temporal rules to determine whether behavior is normal or abnormal. Events are confirmed only after the behavior remains present for the configured duration, which helps reduce single-frame false alarms.

### Output

The system produces:

- An annotated video with bounding boxes, track IDs, pose skeletons, action labels, and normal/abnormal status.
- A structured JSON audit log.
- A web dashboard with side-by-side original and AI-annotated video.
- An incident timeline containing entity, time, behavior, duration, and reason.

---

## 4. Behaviour Taxonomy

| Behaviour | Detection Evidence | Classification | Safety Meaning |
|---|---|---|---|
| Walking | Upright posture and movement above the stationary threshold | Normal | Routine movement through the workplace |
| Stationary / Idle | Low centroid velocity for a short duration | Normal | Normal pause or temporary stop |
| Loitering / Prolonged Inactivity | Stationary behavior continues beyond the configured dwell threshold | Abnormal | Possible blockage, unsafe inactivity, or unusual presence |
| Running / Panic | Sustained velocity above the running threshold | Abnormal | Possible emergency, panic, or unsafe movement |
| Fallen / Prone | Horizontal torso posture or fallback aspect-ratio evidence sustained for the confirmation period | Abnormal | Possible slip, trip, fall, or person down |

The current demonstration configuration uses a five-second loitering threshold, a two-second fall confirmation period, and a one-second running confirmation period. These values are configurable for a real deployment; for example, a warehouse policy may increase the loitering threshold to ten minutes.

---

## 5. How the Project Satisfies the Key Rules

### Rule 1: The system must explain what people are doing

The system does not stop at detecting a person. It classifies the current physical state as Walking, Stationary / Idle, Running, or Fallen / Prone. It also displays the action state directly on the annotated video.

### Rule 2: Every unusual behavior must identify who and when

Every confirmed event is associated with a persistent entity ID. The audit record includes:

- Entity ID, such as `Person #03`.
- Action or anomaly type.
- Start timestamp.
- End timestamp.
- Duration.
- Start and end frames.
- Bounding-box location and centroid.
- Human-readable reason for the alert.

This satisfies the required “Who and When” attribution rule rather than reporting an unexplained generic warning.

### Rule 3: Events must be meaningful

The system confirms prolonged or sustained behavior instead of alerting from a single noisy frame. Loitering, running, and fall events are therefore based on time and motion evidence.

---

## 6. Evaluation Against the Judging Criteria

| Judging Criterion | Implementation in This Project | Result |
|---|---|---|
| Can it recognize what people are doing? | Pose keypoints, motion speed, torso angle, dwell time, and action classification | Satisfied |
| Does it spot meaningful events? | Loitering, running/panic, and slip/fall/man-down rules | Satisfied |
| Can it tell normal from abnormal behavior? | Normal action states are separated from threshold-confirmed anomalies | Satisfied |
| Does it track the same object across the video? | ByteTrack persistent IDs | Satisfied |
| Does it find objects accurately? | YOLO pose detection with confidence and IoU thresholds | Satisfied for people in the selected scenario |
| Are events detected at the right time? | Frame timestamps, FPS-based timing, confirmation durations, and event finalization | Satisfied |

---

## 7. Dashboard and User Experience

The web dashboard provides an operational view of the analysis:

- Upload or drag and drop a video.
- Preview the original footage.
- View the AI-annotated output beside the original.
- Play, pause, and synchronize both videos.
- Monitor analysis progress.
- View detected incident count.
- View tracked entity count.
- Review the incident audit log.
- See the active acceleration mode.

The annotated-video delivery path also converts generated output to a browser-compatible H.264 format, ensuring that the detected video can be displayed in modern browsers.

---

## 8. Evidence and Explainability

The project is designed to be explainable. Each abnormal event is supported by measurable evidence rather than an opaque label.

Examples of explanations include:

- “Stationary for 8.2 seconds, exceeding the configured threshold.”
- “Entity prone on floor for 2.4 seconds.”
- “Abnormal velocity spike of 178.5 pixels per second for 1.3 seconds.”

The annotated video provides visual evidence through bounding boxes, skeletal keypoints, action labels, and color-coded normal or abnormal status.

---

## 9. Project Components

| Component | Responsibility |
|---|---|
| `core/tracker.py` | YOLO pose detection, keypoint extraction, and ByteTrack identity tracking |
| `core/action_engine.py` | Motion, posture, speed, dwell, and action-state calculations |
| `core/anomaly_evaluator.py` | Configurable normal-versus-abnormal safety rules |
| `core/attribution.py` | Structured event logging and timestamp attribution |
| `core/pipeline.py` | End-to-end frame processing and annotated video generation |
| `ui/app.py` | FastAPI APIs, video streaming, analysis execution, and browser-compatible playback |
| `ui/static/js/dashboard.js` | Upload workflow, progress polling, synchronized playback, and incident rendering |
| `config.py` | Model, hardware, confidence, motion, and timing configuration |

---

## 10. Strengths of the System

- Goes beyond object detection to describe behavior.
- Maintains identity across video frames.
- Uses pose information for posture-based fall analysis.
- Uses temporal confirmation to reduce false alerts.
- Produces interpretable safety rules instead of unexplained predictions.
- Provides strict entity and timestamp attribution.
- Supports GPU acceleration through CUDA when available.
- Provides both machine-readable logs and a human-friendly dashboard.
- Supports multiple abnormal behavior classes in one pipeline.

---

## 11. Current Scope and Limitations

The implemented system is optimized for detecting and analyzing people in workplace and warehouse scenarios. It does not currently provide unrestricted, general-purpose anomaly discovery for every possible object or behavior.

The system currently recognizes the behavior classes defined by its rule engine. Detection quality can also vary with camera angle, lighting, occlusion, video resolution, and the number of people in the scene.

These limitations do not prevent the project from satisfying the core challenge requirements. They define the next stage of improvement for a production deployment.

---

## 12. Future Enhancements

- Add zone-aware rules for restricted areas, loading bays, and emergency exits.
- Add personal protective equipment detection.
- Add abandoned-object and unauthorized-entry behavior.
- Add automatic calibration of thresholds from scene context.
- Add multi-camera identity association.
- Add a learned behavior model for unknown anomaly detection.
- Add notification integrations for SMS, email, or a control-room system.
- Expand evaluation with labelled videos and precision, recall, and event-timing metrics.

---

## 13. Final Compliance Statement

The HNX26PSI07 project satisfies the core problem statement for an autonomous workplace and warehouse behaviour-understanding system.

It detects people, tracks persistent identities, understands motion and posture, classifies normal and abnormal behavior, and produces meaningful events that identify exactly who acted and when the behavior occurred. The system therefore addresses the central challenge: moving from “there is a person” to “this identified person is walking normally, loitering, running unexpectedly, or possibly fallen.”

The advanced goal of fully general anomaly detection remains a future enhancement, while the implemented multi-behavior rule engine fulfills the required demonstration scope.
