"""
Inference Router
Handles video upload, processing job management, and result retrieval.
"""

import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from app.core.config import settings
from app.services.inference_service import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store (suitable for single-worker demo use)
_jobs: dict[str, dict] = {}


# ── Upload & start job ────────────────────────────────────────────────────────

@router.post("/process")
async def process_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a video and start processing in the background."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    video_filename = f"{job_id}_input{ext}"
    video_path = os.path.join(settings.UPLOAD_DIR, video_filename)

    # Save upload
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    output_video = os.path.join(settings.OUTPUT_DIR, f"{job_id}_output.mp4")
    output_csv = os.path.join(settings.OUTPUT_DIR, f"{job_id}_offenders.csv")

    _jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "frame": 0,
        "total_frames": 0,
        "fps_proc": 0.0,
        "result": None,
        "error": None,
        "video_path": video_path,
        "output_video": output_video,
        "output_csv": output_csv,
    }

    manager = request.app.state.model_manager
    background_tasks.add_task(_run_job, job_id, video_path, output_video, output_csv, manager)

    return {"job_id": job_id, "status": "queued"}


# ── Job status polling ────────────────────────────────────────────────────────

@router.get("/job/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "frame": job["frame"],
        "total_frames": job["total_frames"],
        "fps_proc": job["fps_proc"],
        "result": job["result"],
        "error": job["error"],
    }


# ── Download output video ─────────────────────────────────────────────────────

@router.get("/job/{job_id}/video")
async def download_video(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Video not ready.")
    path = job["output_video"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output file missing.")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}_annotated.mp4")


# ── Download CSV ──────────────────────────────────────────────────────────────

@router.get("/job/{job_id}/csv")
async def download_csv(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="CSV not ready.")
    path = job["output_csv"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="CSV file missing.")
    return FileResponse(path, media_type="text/csv", filename=f"{job_id}_offenders.csv")


# ── List all jobs ─────────────────────────────────────────────────────────────

@router.get("/jobs")
async def list_jobs():
    return [
        {"job_id": jid, "status": j["status"], "progress": j["progress"]}
        for jid, j in _jobs.items()
    ]


# ── Background worker ─────────────────────────────────────────────────────────

async def _run_job(job_id: str, video_path: str, output_video: str, output_csv: str, manager):
    _jobs[job_id]["status"] = "processing"

    def _progress(frame_num, total, fps_proc):
        pct = int(100 * frame_num / total) if total > 0 else 0
        _jobs[job_id]["progress"] = pct
        _jobs[job_id]["frame"] = frame_num
        _jobs[job_id]["total_frames"] = total
        _jobs[job_id]["fps_proc"] = round(fps_proc, 2)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                video_path=video_path,
                output_video_path=output_video,
                output_csv_path=output_csv,
                manager=manager,
                progress_callback=_progress,
            ),
        )
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["progress"] = 100
        _jobs[job_id]["result"] = result
        logger.info(f"Job {job_id} completed. Offenders: {result['offenders_count']}")
    except Exception as exc:
        logger.exception(f"Job {job_id} failed: {exc}")
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)
