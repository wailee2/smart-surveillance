"""
Frame Streamer
==============
During processing, the pipeline writes annotated frames into a shared
in-memory JPEG buffer (one slot per job).  The MJPEG endpoint reads from
that buffer and streams it to the browser as multipart/x-mixed-replace,
giving the "watch it process live" view.

Architecture:
  run_pipeline()  ──write──▶  _frame_buffers[job_id]
  GET /api/job/{id}/preview  ──read──▶  browser <img>
"""

import threading
import time
from typing import Optional

# job_id → {"jpeg": bytes, "ts": float, "done": bool}
_frame_buffers: dict[str, dict] = {}
_lock = threading.Lock()


def init_job(job_id: str):
    with _lock:
        _frame_buffers[job_id] = {"jpeg": None, "ts": 0.0, "done": False}


def push_frame(job_id: str, jpeg_bytes: bytes):
    with _lock:
        buf = _frame_buffers.get(job_id)
        if buf is not None:
            buf["jpeg"] = jpeg_bytes
            buf["ts"]   = time.time()


def mark_done(job_id: str):
    with _lock:
        buf = _frame_buffers.get(job_id)
        if buf is not None:
            buf["done"] = True


def get_frame(job_id: str) -> Optional[bytes]:
    with _lock:
        buf = _frame_buffers.get(job_id)
        return buf["jpeg"] if buf else None


def is_done(job_id: str) -> bool:
    with _lock:
        buf = _frame_buffers.get(job_id)
        return buf["done"] if buf else True


def cleanup(job_id: str):
    with _lock:
        _frame_buffers.pop(job_id, None)
