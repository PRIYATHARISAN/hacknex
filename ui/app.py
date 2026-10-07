"""
FastAPI Dashboard Server for Autonomous Vision & Behaviour Understanding.
Provides REST APIs for video playback, analysis execution, and real-time incident auditing.
"""

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from core.pipeline import VisionPipeline

app = FastAPI(title="Autonomous Vision & Behaviour Understanding Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
TEMPLATES_DIR = UI_DIR / "templates"

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Shared state for analysis progress
analysis_state = {
    "running": False,
    "progress": 0.0,
    "current_frame": 0,
    "total_frames": 0,
    "output_video": None,
    "error": None,
}


class RunRequest(BaseModel):
    video_path: str


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index template not found")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/videos")
async def list_videos():
    """Lists available input videos and output videos."""
    videos = []
    # Input videos
    if config.INPUT_VIDEOS_DIR.exists():
        for f in config.INPUT_VIDEOS_DIR.iterdir():
            if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                videos.append({"name": f"[Input] {f.name}", "path": str(f.resolve())})

    # Output videos
    if config.OUTPUT_LOGS_DIR.exists():
        for f in config.OUTPUT_LOGS_DIR.iterdir():
            if f.suffix.lower() in [".mp4", ".avi", ".mov"]:
                videos.append({"name": f"[Annotated] {f.name}", "path": str(f.resolve())})

    return {"videos": videos}


@app.get("/api/events")
async def get_events():
    """Reads latest JSON event audit log."""
    log_file = config.DEFAULT_LOG_FILE
    if not log_file.exists():
        return {"events": [], "total_incidents": 0}
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"events": [], "error": str(e)}


@app.get("/api/status")
async def get_status():
    """Returns current analysis status."""
    return analysis_state


def _run_pipeline_worker(video_path: str):
    global analysis_state
    try:
        pipeline = VisionPipeline(
            model_name=config.MODEL_NAME,
            log_file=config.DEFAULT_LOG_FILE,
            device=config.DEVICE,
        )

        def progress_callback(frame_idx: int, total_frames: int, active_events):
            analysis_state["current_frame"] = frame_idx
            analysis_state["total_frames"] = total_frames
            if total_frames > 0:
                analysis_state["progress"] = frame_idx / total_frames

        out_video = str(config.DEFAULT_ANNOTATED_VIDEO)
        pipeline.process_video(
            video_path=video_path,
            output_video_path=out_video,
            progress_callback=progress_callback,
        )

        analysis_state["output_video"] = out_video
        analysis_state["progress"] = 1.0
        analysis_state["running"] = False
    except Exception as e:
        analysis_state["running"] = False
        analysis_state["error"] = str(e)
        print(f"[Worker Error] {e}")


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a user-provided video to data/input_videos."""
    try:
        config.INPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        filename = file.filename or "uploaded_video.mp4"
        save_path = config.INPUT_VIDEOS_DIR / filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"name": filename, "path": str(save_path.resolve())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/run")
async def start_analysis(req: RunRequest):
    """Triggers analysis on the selected video."""
    global analysis_state
    if analysis_state["running"]:
        return {"error": "Analysis is already running."}

    v_path = Path(req.video_path)
    if not v_path.exists():
        raise HTTPException(status_code=400, detail="Video file does not exist")

    analysis_state["running"] = True
    analysis_state["progress"] = 0.0
    analysis_state["current_frame"] = 0
    analysis_state["total_frames"] = 0
    analysis_state["output_video"] = None
    analysis_state["error"] = None

    thread = threading.Thread(target=_run_pipeline_worker, args=(str(v_path),), daemon=True)
    thread.start()

    return {"message": "Analysis started successfully."}


@app.get("/api/video_stream")
async def stream_video(file: str = Query(...)):
    """Streams video file with support for HTTP Range requests (scrubbing/seeking)."""
    file_path = Path(file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path=str(file_path), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui.app:app", host="127.0.0.1", port=8000, reload=True)
