"""
Stream router — MJPEG live preview from IP camera (DroidCam/RTSP/webcam).
"""

import cv2, asyncio, logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_cam_source: dict = {"url": None, "active": False}


@router.post("/stream/connect")
async def connect_camera(url: str):
    """
    Connect to a camera source.
    Examples:
      url=0                        → local webcam
      url=http://192.168.x.x:4747/video   → DroidCam
      url=rtsp://user:pass@ip/stream      → IP camera
    """
    # Try to parse as int for local webcam index
    try:
        src = int(url)
    except ValueError:
        src = url
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(400, f"Cannot open camera source: {url}")
    cap.release()
    _cam_source["url"] = src
    _cam_source["active"] = True
    return {"status": "connected", "source": url}


@router.post("/stream/disconnect")
async def disconnect_camera():
    _cam_source["active"] = False
    _cam_source["url"] = None
    return {"status": "disconnected"}


@router.get("/stream/status")
async def stream_status():
    return {"active": _cam_source["active"], "url": str(_cam_source.get("url", ""))}


@router.get("/stream/feed")
async def live_feed():
    """MJPEG stream endpoint for the browser <img> tag."""
    if not _cam_source.get("active") or _cam_source.get("url") is None:
        raise HTTPException(404, "No camera connected.")

    def frame_generator():
        cap = cv2.VideoCapture(_cam_source["url"])
        if not cap.isOpened():
            return
        try:
            while _cam_source.get("active"):
                ret, frame = cap.read()
                if not ret:
                    break
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
        finally:
            cap.release()

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
