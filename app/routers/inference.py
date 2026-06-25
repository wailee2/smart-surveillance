"""
Inference Router v3 — adds /preview MJPEG endpoint for live frame streaming.
"""

import os, uuid, asyncio, logging, json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks, Form
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import settings
from app.core.vtl_state import set_color as vtl_set, get_state as vtl_get
from app.core.session_store import add_session, update_session
from app.services.inference_service import run_pipeline, get_live_events
from app.services import frame_streamer

logger = logging.getLogger(__name__)
router  = APIRouter()
_jobs: dict[str, dict] = {}


# ── Upload & process ───────────────────────────────────────────────────────────
@router.post("/process")
async def process_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    stop_line_y: Optional[int] = Form(None),
    vtl_override: bool = Form(False),
    speed_limit: float = Form(50.0),
    conf_threshold: float = Form(0.4),
):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video.")

    settings.SPEED_LIMIT_KMH = speed_limit
    settings.CONF_THRESHOLD  = conf_threshold

    job_id     = str(uuid.uuid4())[:8]
    ext        = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}_input{ext}")
    out_video  = os.path.join(settings.OUTPUT_DIR,  f"{job_id}_output.mp4")
    out_csv    = os.path.join(settings.OUTPUT_DIR,  f"{job_id}_offenders.csv")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(await file.read())

    _jobs[job_id] = {
        "status":"queued","progress":0,"frame":0,"total_frames":0,
        "fps_proc":0.0,"result":None,"error":None,
        "video_path":video_path,"output_video":out_video,"output_csv":out_csv,
        "filename":file.filename,
    }

    add_session(job_id, file.filename or "video", {
        "status":"queued","speed_limit":speed_limit,
        "conf_threshold":conf_threshold,"stop_line_y":stop_line_y,
    })

    manager = request.app.state.model_manager
    background_tasks.add_task(
        _run_job, job_id, video_path, out_video, out_csv,
        manager, stop_line_y, vtl_override,
    )
    return {"job_id": job_id, "status": "queued"}


# ── MJPEG live preview ─────────────────────────────────────────────────────────
@router.get("/job/{job_id}/preview")
async def live_preview(job_id: str):
    """
    MJPEG stream that delivers annotated frames as they are produced
    by the pipeline. Use as <img src="/api/job/{id}/preview"> in the browser.
    """
    async def generator():
        last_ts = 0.0
        no_frame_count = 0
        while True:
            jpeg = frame_streamer.get_frame(job_id)
            if jpeg is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg + b"\r\n")
                no_frame_count = 0
            else:
                no_frame_count += 1

            if frame_streamer.is_done(job_id):
                break
            if no_frame_count > 60:   # 30s timeout with no frame
                break
            await asyncio.sleep(0.05)   # ~20fps poll

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── VTL ────────────────────────────────────────────────────────────────────────
@router.post("/vtl")
async def set_vtl(color: str, override: bool = True):
    if color not in ("red","amber","green"):
        raise HTTPException(400, "color must be red|amber|green")
    vtl_set(color, override)
    return vtl_get()


@router.get("/vtl")
async def get_vtl():
    return vtl_get()


# ── Job status ─────────────────────────────────────────────────────────────────
@router.get("/job/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return {
        "job_id":job_id,"status":job["status"],"progress":job["progress"],
        "frame":job["frame"],"total_frames":job["total_frames"],
        "fps_proc":job["fps_proc"],"result":job["result"],"error":job["error"],
    }


# ── SSE violations ─────────────────────────────────────────────────────────────
@router.get("/job/{job_id}/events")
async def job_events(job_id: str):
    async def gen():
        sent = 0
        while True:
            job  = _jobs.get(job_id)
            evts = get_live_events(job_id)
            for e in evts[sent:]:
                yield f"data: {json.dumps(e)}\n\n"
                sent += 1
            if job and job["status"] in ("done","error"):
                yield f"data: {json.dumps({'type':'done','status':job['status']})}\n\n"
                break
            await asyncio.sleep(0.4)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ── File downloads ─────────────────────────────────────────────────────────────
@router.get("/job/{job_id}/video")
async def download_video(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Video not ready.")
    path = job["output_video"]
    if not os.path.exists(path):
        raise HTTPException(404, "Output missing.")
    return FileResponse(path, media_type="video/mp4",
                        filename=f"{job_id}_annotated.mp4",
                        headers={"Accept-Ranges":"bytes"})


@router.get("/job/{job_id}/csv")
async def download_csv(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "CSV not ready.")
    path = job["output_csv"]
    if not os.path.exists(path):
        raise HTTPException(404, "CSV missing.")
    return FileResponse(path, media_type="text/csv",
                        filename=f"{job_id}_offenders.csv")


@router.get("/jobs")
async def list_jobs():
    return [{"job_id":jid,"status":j["status"],"progress":j["progress"],
             "filename":j.get("filename")} for jid,j in _jobs.items()]


# ── Background worker ──────────────────────────────────────────────────────────
async def _run_job(job_id, video_path, out_video, out_csv,
                   manager, stop_line_y, vtl_override):
    _jobs[job_id]["status"] = "processing"
    update_session(job_id, {"status":"processing"})

    def _progress(frame_num, total, fps_proc):
        pct = int(100*frame_num/total) if total else 0
        _jobs[job_id].update(progress=pct, frame=frame_num,
                             total_frames=total, fps_proc=round(fps_proc,2))

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: run_pipeline(
            video_path=video_path, output_video_path=out_video,
            output_csv_path=out_csv, manager=manager,
            job_id=job_id, stop_line_y=stop_line_y,
            vtl_override=vtl_override, progress_callback=_progress,
        ))
        _jobs[job_id].update(status="done", progress=100, result=result)
        update_session(job_id, {
            "status":"done",
            "offenders_count":result["offenders_count"],
            "speeding_count":result["speeding_count"],
            "red_light_count":result["red_light_count"],
            "duration_s":result["duration_s"],
            "processing_time_s":result["processing_time_s"],
        })
    except Exception as exc:
        logger.exception(f"Job {job_id} failed: {exc}")
        _jobs[job_id].update(status="error", error=str(exc))
        update_session(job_id, {"status":"error","error":str(exc)})
        frame_streamer.mark_done(job_id)
