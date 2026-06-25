"""
Inference Service v3
Adds live frame streaming via frame_streamer buffer.
Every annotated frame is JPEG-encoded and pushed so the browser
can watch the video being processed frame-by-frame.
"""

import os, re, cv2, time, logging, subprocess, shutil
import numpy as np
import pandas as pd
import supervision as sv
from typing import Optional
from collections import defaultdict

from app.core.config import settings
from app.core.model_manager import ModelManager
from app.core.vtl_state import get_state as vtl_get
from app.services.speed_pipeline import DepthScaleEstimator, MotionEstimator, SpeedTracker
from app.services import frame_streamer

logger = logging.getLogger(__name__)

# Live violation event store  job_id → [events]
_live_events: dict[str, list] = {}


def get_live_events(job_id: str) -> list:
    return _live_events.get(job_id, [])


def _push_event(job_id: str, event: dict):
    _live_events.setdefault(job_id, []).append(event)


# ── Snapshot buffer ────────────────────────────────────────────────────────────
class SnapshotBuffer:
    def __init__(self):
        self._buf: dict[int, dict] = {}

    def update(self, tid, plate_crop, plate_conf, vehicle_snap, frame_num):
        if plate_crop is None or plate_crop.size == 0:
            return
        ex = self._buf.get(tid)
        if ex is None or plate_conf > ex["plate_confidence"]:
            self._buf[tid] = {
                "plate_crop": plate_crop.copy(),
                "plate_confidence": plate_conf,
                "vehicle_snapshot": vehicle_snap.copy() if vehicle_snap is not None else None,
                "snapshot_frame": frame_num,
            }

    def ensure_vehicle_snap(self, tid, vehicle_snap, frame_num):
        if tid not in self._buf and vehicle_snap is not None and vehicle_snap.size > 0:
            self._buf[tid] = {
                "plate_crop": None, "plate_confidence": 0.0,
                "vehicle_snapshot": vehicle_snap.copy(),
                "snapshot_frame": frame_num,
            }

    def get(self, tid) -> Optional[dict]:
        return self._buf.get(tid)


# ── Stop-line detection ────────────────────────────────────────────────────────
def detect_stop_line(frame: np.ndarray, roi_y1: int, roi_y2: int) -> Optional[int]:
    roi = frame[roi_y1:roi_y2, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                             minLineLength=int(frame.shape[1]*0.3), maxLineGap=40)
    if lines is None:
        return None
    h_lines = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if angle < 10 or angle > 170:
            h_lines.append((y1+y2)//2)
    return (int(np.median(h_lines)) + roi_y1) if h_lines else None


def _crosses_line(prev_cy: float, curr_cy: float, line_y: int) -> bool:
    return prev_cy < line_y <= curr_cy


# ── Re-encode ─────────────────────────────────────────────────────────────────
def _reencode_for_browser(src: str, dst: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", src,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart", "-an", dst,
        ], check=True, capture_output=True)
        return True
    except Exception as exc:
        logger.warning(f"ffmpeg failed: {exc}")
        return False


# ── Colour helpers ────────────────────────────────────────────────────────────
VIOLATION_BOX_COLOR  = (0, 0, 255)      # red  BGR
NORMAL_BOX_COLOR     = (0, 255, 65)     # green BGR
PLATE_BOX_COLOR      = (0, 255, 255)    # yellow
STOP_LINE_RED        = (0, 0, 255)
STOP_LINE_GREEN      = (0, 200, 0)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(
    video_path: str,
    output_video_path: str,
    output_csv_path: str,
    manager: ModelManager,
    job_id: str = "job",
    stop_line_y: Optional[int] = None,
    vtl_override: bool = False,
    progress_callback=None,
) -> dict:

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video {W}×{H} @ {fps:.1f}fps  {total_frames}f")

    q = H / 4
    ROI_Y1, ROI_Y2 = int(q*1.65), int(q*3.95)

    raw_path = output_video_path.replace(".mp4", "_raw.mp4")
    writer   = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_interval = settings.get_frame_interval()
    depth_est   = DepthScaleEstimator(manager.depth_model, manager.depth_processor, H, W)
    motion_est  = MotionEstimator(manager.raft_model, manager.raft_transforms, frame_interval)
    speed_trk   = SpeedTracker(fps, depth_est, motion_est)

    snap_buf        = SnapshotBuffer()
    offender_flags: dict[int, set]  = {}
    vehicle_type_map: dict[int, str] = {}
    prev_cy: dict[int, float]        = {}
    current_traffic_state = "green"
    auto_stop_line: Optional[int]    = None
    frame_number  = 0
    start_time    = time.time()
    _live_events[job_id] = []

    # Init live frame buffer
    frame_streamer.init_job(job_id)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        annotated = frame.copy()
        roi_crop  = frame[ROI_Y1:ROI_Y2, :]

        # Auto stop-line (first frame)
        if frame_number == 1 and stop_line_y is None:
            auto_stop_line = detect_stop_line(frame, ROI_Y1, ROI_Y2)

        effective_stop = stop_line_y if stop_line_y is not None else auto_stop_line

        # ── Traffic signal ─────────────────────────────────────────────────
        vtl = vtl_get()
        if vtl_override or vtl["override"]:
            current_traffic_state = vtl["color"]
        elif manager.traffic_model:
            tl_res = manager.traffic_model.predict(
                frame, conf=settings.CONF_THRESHOLD, device=settings.DEVICE, verbose=False)
            for box in tl_res[0].boxes:
                current_traffic_state = manager.traffic_model.names[int(box.cls[0])]

        is_red = "red" in current_traffic_state.lower()

        # ── Vehicle detect + track ─────────────────────────────────────────
        frame_has_violation = False
        if manager.vehicle_model:
            v_res = manager.vehicle_model.track(
                roi_crop,
                conf=settings.CONF_THRESHOLD, iou=settings.IOU_THRESHOLD,
                tracker="bytetrack.yaml", device=settings.DEVICE,
                persist=True, verbose=False,
            )
            if v_res[0].boxes.id is not None:
                dets = sv.Detections.from_ultralytics(v_res[0])
                dets.xyxy[:, [1,3]] += ROI_Y1  # Y offset

                labels, keep, box_colors = [], [], []

                for i, tid in enumerate(dets.tracker_id):
                    cls = manager.vehicle_model.names[dets.class_id[i]]
                    if cls.lower() not in settings.VEHICLE_ALLOWED_CLASSES:
                        keep.append(False); continue
                    keep.append(True)
                    vehicle_type_map[tid] = cls

                    x1,y1,x2,y2 = map(int, dets.xyxy[i])
                    cx = (x1+x2)/2; cy = (y1+y2)/2
                    v_snap = frame[y1:y2, x1:x2].copy()
                    snap_buf.ensure_vehicle_snap(tid, v_snap, frame_number)

                    # License plate
                    if manager.lp_model and v_snap.size > 0:
                        lp_res = manager.lp_model.predict(
                            v_snap, conf=settings.CONF_THRESHOLD,
                            device=settings.DEVICE, verbose=False)
                        for lb in lp_res[0].boxes:
                            lx1,ly1,lx2,ly2 = map(int, lb.xyxy[0])
                            plate_crop = v_snap[ly1:ly2, lx1:lx2]
                            snap_buf.update(tid, plate_crop, float(lb.conf[0]), v_snap, frame_number)
                            # Draw plate box on annotated (offset to full frame)
                            cv2.rectangle(annotated,
                                (x1+lx1, y1+ly1), (x1+lx2, y1+ly2),
                                PLATE_BOX_COLOR, 2)

                    # Speed
                    spd_int = speed_trk.get_internal_speed(tid)
                    if spd_int is not None and spd_int > settings.SPEED_LIMIT_KMH:
                        if "speeding" not in offender_flags.get(tid, set()):
                            offender_flags.setdefault(tid, set()).add("speeding")
                            _push_event(job_id, {
                                "type":"violation","track_id":int(tid),
                                "violation":"speeding","speed":round(spd_int,1),
                                "vehicle":cls,"frame":frame_number,
                                "timestamp":round(frame_number/fps,2),
                            })

                    # Red-light / stop-line crossing
                    if is_red:
                        crossed = False
                        p_cy = prev_cy.get(tid)
                        if effective_stop is not None and p_cy is not None:
                            crossed = _crosses_line(p_cy, cy, effective_stop)
                        elif effective_stop is None:
                            crossed = True
                        if crossed and "red_light" not in offender_flags.get(tid, set()):
                            offender_flags.setdefault(tid, set()).add("red_light")
                            _push_event(job_id, {
                                "type":"violation","track_id":int(tid),
                                "violation":"red_light","speed":spd_int,
                                "vehicle":cls,"frame":frame_number,
                                "timestamp":round(frame_number/fps,2),
                            })

                    prev_cy[tid] = cy
                    is_vio = bool(offender_flags.get(tid))
                    if is_vio:
                        frame_has_violation = True

                    spd = speed_trk.get_speed(tid)
                    spd_str = f"{spd}km/h" if spd is not None else "--"
                    vio_str = " VIOLATION" if is_vio else ""
                    labels.append(f"#{tid} {spd_str}{vio_str}")
                    box_colors.append(VIOLATION_BOX_COLOR if is_vio else NORMAL_BOX_COLOR)

                keep_arr = np.array(keep, dtype=bool)
                filtered = dets[keep_arr] if keep_arr.any() else sv.Detections.empty()
                speed_trk.update(filtered, frame, frame_number, manager.vehicle_model)

                # Draw boxes manually so we can colour per-track
                for i in range(len(filtered)):
                    x1,y1,x2,y2 = map(int, filtered.xyxy[i])
                    col = box_colors[i] if i < len(box_colors) else NORMAL_BOX_COLOR
                    thick = 3 if col == VIOLATION_BOX_COLOR else 2
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), col, thick)
                    # Label background
                    lbl = labels[i] if i < len(labels) else ""
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+6, y1), col, -1)
                    cv2.putText(annotated, lbl, (x1+3, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

        # ── Draw stop line ─────────────────────────────────────────────────
        if effective_stop:
            sl_col = STOP_LINE_RED if is_red else STOP_LINE_GREEN
            cv2.line(annotated, (0, effective_stop), (W, effective_stop), sl_col, 3)
            cv2.putText(annotated, "STOP LINE", (8, effective_stop-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, sl_col, 2)

        # ── HUD overlay ────────────────────────────────────────────────────
        annotated = _draw_hud(annotated, current_traffic_state,
                               frame_number, total_frames, fps, frame_has_violation)

        writer.write(annotated)

        # ── Push JPEG to live preview buffer ───────────────────────────────
        # Resize to max 720p-width for streaming efficiency
        preview = annotated
        if W > 960:
            scale = 960 / W
            preview = cv2.resize(annotated, (960, int(H*scale)))
        _, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 72])
        frame_streamer.push_frame(job_id, jpeg.tobytes())

        if progress_callback and frame_number % 5 == 0:
            elapsed = time.time() - start_time
            progress_callback(frame_number, total_frames,
                              frame_number/elapsed if elapsed > 0 else 0)

    cap.release()
    writer.release()
    frame_streamer.mark_done(job_id)

    # Re-encode
    if not _reencode_for_browser(raw_path, output_video_path):
        os.replace(raw_path, output_video_path)
    else:
        try: os.remove(raw_path)
        except: pass

    # OCR pass
    offender_log = _run_ocr(offender_flags, snap_buf, speed_trk,
                             vehicle_type_map, fps, manager, job_id)
    offender_log.to_csv(output_csv_path, index=False)

    elapsed_total = time.time() - start_time
    return {
        "frames_processed":  frame_number,
        "total_frames":      total_frames,
        "duration_s":        round(frame_number/fps, 2),
        "processing_time_s": round(elapsed_total, 1),
        "fps_processing":    round(frame_number/elapsed_total, 2) if elapsed_total else 0,
        "offenders_count":   len(offender_log),
        "speeding_count":    int(offender_log["violation_type"].str.contains("speeding").sum()) if len(offender_log) else 0,
        "red_light_count":   int(offender_log["violation_type"].str.contains("red_light").sum()) if len(offender_log) else 0,
        "stop_line_y":       effective_stop,
        "output_video":      output_video_path,
        "output_csv":        output_csv_path,
        "offenders":         offender_log.to_dict(orient="records"),
    }


def _run_ocr(offender_flags, snap_buf, speed_trk, vehicle_type_map,
             fps, manager, job_id) -> pd.DataFrame:
    rows = []
    for tid, violations in offender_flags.items():
        vtype     = vehicle_type_map.get(tid,"unknown")
        spd       = speed_trk.get_speed(tid)
        method    = speed_trk.get_method(tid)
        vtype_str = ",".join(sorted(violations))
        plate_text = "UNREADABLE"
        entry = snap_buf.get(tid)

        if entry and entry.get("plate_crop") is not None and manager.ocr_reader:
            try:
                results = manager.ocr_reader.readtext(entry["plate_crop"])
                raw = "".join(t for _,t,c in results if c > settings.OCR_CONFIDENCE_THRESHOLD)
                plate_text = re.sub(r"[^A-Z0-9]","",raw.upper()) or "UNREADABLE"
            except Exception as exc:
                logger.warning(f"OCR #{tid}: {exc}")

        snap_web = None
        if entry and entry.get("vehicle_snapshot") is not None:
            os.makedirs(settings.SNAPSHOTS_DIR, exist_ok=True)
            fn = f"{job_id}_track{tid}_{vtype_str.replace(',','_')}_f{entry['snapshot_frame']}.jpg"
            disk = os.path.join(settings.SNAPSHOTS_DIR, fn)
            cv2.imwrite(disk, entry["vehicle_snapshot"])
            snap_web = f"/outputs/snapshots/{fn}"

        rows.append({
            "track_id":       tid,
            "vehicle_type":   vtype,
            "plate_number":   plate_text,
            "speed_kmh":      spd,
            "speed_method":   method,
            "violation_type": vtype_str,
            "snapshot_path":  snap_web,
            "timestamp_frame":entry["snapshot_frame"] if entry else 0,
            "timestamp_real": round((entry["snapshot_frame"] if entry else 0)/fps,2),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "track_id","vehicle_type","plate_number","speed_kmh",
        "speed_method","violation_type","snapshot_path",
        "timestamp_frame","timestamp_real",
    ])


def _draw_hud(frame, signal, frame_num, total, fps, has_violation):
    """Draw translucent HUD bar at the bottom of the frame."""
    H, W = frame.shape[:2]
    bar_h = 38
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H-bar_h), (W, H), (10,10,12), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    sig_col = (0,60,220) if "red" in signal else (0,200,220) if "amber" in signal else (0,200,80)
    cv2.putText(frame, f"SIGNAL:{signal.upper()}", (8, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, sig_col, 2)
    pct = int(100*frame_num/total) if total else 0
    cv2.putText(frame, f"FRAME {frame_num}/{total} ({pct}%)", (W//2-110, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1)
    cv2.putText(frame, f"LIMIT:{settings.SPEED_LIMIT_KMH:.0f}km/h", (W-175, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1)
    if has_violation:
        # Flashing red border
        cv2.rectangle(frame, (0,0), (W-1,H-1), (0,0,255), 4)
    return frame
