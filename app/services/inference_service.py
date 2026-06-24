"""
Inference Service
Runs the full video processing pipeline:
  1. Traffic light detection  → red-light violation flag
  2. Vehicle detection + ByteTrack  → tracking IDs
  3. License plate detection  → OCR crop buffer
  4. Depth Anything V2 + Farneback/RAFT  → speed estimate
  5. Postprocessing OCR  → plate text
  6. Annotated video output + offender CSV
"""

import os
import re
import cv2
import time
import logging
import numpy as np
import pandas as pd
import supervision as sv
from typing import Optional, Generator
from collections import defaultdict

from app.core.config import settings
from app.core.model_manager import ModelManager
from app.services.speed_pipeline import (
    DepthScaleEstimator,
    MotionEstimator,
    SpeedTracker,
)

logger = logging.getLogger(__name__)


TRAFFIC_LIGHT_CLASSES: list[str] = []  # populated at runtime from model


def _is_red_light(cls_name: str) -> bool:
    return "red" in cls_name.lower()


def _encode_traffic_light(cls_name: str, classes: list[str]) -> np.ndarray:
    vec = np.zeros(len(classes), dtype=int)
    if cls_name in classes:
        vec[classes.index(cls_name)] = 1
    return vec


# ── Snapshot buffer helpers ───────────────────────────────────────────────────

class SnapshotBuffer:
    def __init__(self):
        self._buf: dict[int, dict] = {}

    def update(self, tid: int, plate_crop, plate_conf: float, vehicle_snap, frame_num: int):
        if plate_crop is None or plate_crop.size == 0:
            return
        existing = self._buf.get(tid)
        if existing is None or plate_conf > existing["plate_confidence"]:
            self._buf[tid] = {
                "plate_crop": plate_crop.copy(),
                "plate_confidence": plate_conf,
                "vehicle_snapshot": vehicle_snap.copy() if vehicle_snap is not None else None,
                "snapshot_frame": frame_num,
            }

    def get(self, tid: int) -> Optional[dict]:
        return self._buf.get(tid)

    def all_keys(self) -> set:
        return set(self._buf.keys())


# ── Main pipeline function ────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    output_video_path: str,
    output_csv_path: str,
    manager: ModelManager,
    progress_callback=None,   # callable(frame_num, total_frames, fps_proc)
) -> dict:
    """
    Process a video file end-to-end.
    Returns a summary dict.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ── Video reader ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {W}×{H} @ {fps:.1f} fps, {total_frames} frames")

    # ── ROI ───────────────────────────────────────────────────────────────
    q = H / 4
    ROI_X1, ROI_X2 = 0, W
    ROI_Y1 = int(q * 1.65)
    ROI_Y2 = int(q * 3.95)

    # ── Video writer ──────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    # ── Annotators ────────────────────────────────────────────────────────
    box_ann = sv.BoxAnnotator(thickness=2)
    lbl_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, smart_position=True)

    # ── Traffic light class list ─────────────────────────────────────────
    tl_classes: list[str] = []
    if manager.traffic_model:
        tl_classes = list(manager.traffic_model.names.values())

    # ── Pipeline components ───────────────────────────────────────────────
    frame_interval = settings.get_frame_interval()
    depth_est = DepthScaleEstimator(manager.depth_model, manager.depth_processor, H, W)
    motion_est = MotionEstimator(
        raft_model=manager.raft_model,
        raft_transforms=manager.raft_transforms,
        frame_interval=frame_interval,
    )
    speed_tracker = SpeedTracker(fps, depth_est, motion_est)

    snap_buf = SnapshotBuffer()
    offender_flags: dict[int, set] = {}
    vehicle_type_map: dict[int, str] = {}
    current_traffic_state = "unknown"
    frame_number = 0
    start_time = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        annotated = frame.copy()
        roi_crop = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

        # 1. Traffic light
        if manager.traffic_model:
            tl_res = manager.traffic_model.predict(
                frame, conf=settings.CONF_THRESHOLD,
                device=settings.DEVICE, verbose=False
            )
            for box in tl_res[0].boxes:
                current_traffic_state = manager.traffic_model.names[int(box.cls[0])]

        # 2. Vehicle detection + tracking
        current_tids: list[int] = []
        if manager.vehicle_model:
            v_res = manager.vehicle_model.track(
                roi_crop,
                conf=settings.CONF_THRESHOLD,
                iou=settings.IOU_THRESHOLD,
                tracker="bytetrack.yaml",
                device=settings.DEVICE,
                persist=True,
                verbose=False,
            )

            if v_res[0].boxes.id is not None:
                dets = sv.Detections.from_ultralytics(v_res[0])
                # Offset back to full-frame coordinates
                dets.xyxy[:, 0] += ROI_X1
                dets.xyxy[:, 1] += ROI_Y1
                dets.xyxy[:, 2] += ROI_X1
                dets.xyxy[:, 3] += ROI_Y1

                labels: list[str] = []
                keep = []

                for i, tid in enumerate(dets.tracker_id):
                    cls = manager.vehicle_model.names[dets.class_id[i]]
                    if cls.lower() not in settings.VEHICLE_ALLOWED_CLASSES:
                        keep.append(False)
                        continue
                    keep.append(True)

                    vehicle_type_map[tid] = cls
                    current_tids.append(tid)

                    x1, y1, x2, y2 = map(int, dets.xyxy[i])
                    v_snap = frame[y1:y2, x1:x2].copy()

                    # 3. License plate detection
                    if manager.lp_model and v_snap.size > 0:
                        lp_res = manager.lp_model.predict(
                            v_snap, conf=settings.CONF_THRESHOLD,
                            device=settings.DEVICE, verbose=False
                        )
                        for lp_box in lp_res[0].boxes:
                            lx1, ly1, lx2, ly2 = map(int, lp_box.xyxy[0])
                            lp_conf = float(lp_box.conf[0])
                            plate_crop = v_snap[ly1:ly2, lx1:lx2]
                            snap_buf.update(tid, plate_crop, lp_conf, v_snap, frame_number)

                    # Violation flags
                    if _is_red_light(current_traffic_state):
                        offender_flags.setdefault(tid, set()).add("red_light")
                    spd_int = speed_tracker.get_internal_speed(tid)
                    if spd_int is not None and spd_int > settings.SPEED_LIMIT_KMH:
                        offender_flags.setdefault(tid, set()).add("speeding")

                    spd = speed_tracker.get_speed(tid)
                    labels.append(f"#{tid} {spd} km/h" if spd is not None else f"#{tid} --")

                keep_arr = np.array(keep, dtype=bool)
                filtered = dets[keep_arr] if keep_arr.any() else dets[np.zeros(len(dets), dtype=bool)]

                # 4. Update speed tracker
                speed_tracker.update(filtered, frame, frame_number, manager.vehicle_model)

                if len(filtered) > 0:
                    annotated = box_ann.annotate(scene=annotated, detections=filtered)
                    annotated = lbl_ann.annotate(scene=annotated, detections=filtered, labels=labels)

        # Overlay info bar
        annotated = _draw_info_bar(annotated, current_traffic_state, frame_number, total_frames, fps)

        writer.write(annotated)

        # Progress callback every 10 frames
        if progress_callback and frame_number % 10 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_number / elapsed if elapsed > 0 else 0
            progress_callback(frame_number, total_frames, fps_proc)

    cap.release()
    writer.release()

    # ── Postprocessing OCR ────────────────────────────────────────────────
    offender_log = _run_ocr(offender_flags, snap_buf, speed_tracker, vehicle_type_map, fps, manager)

    # ── Save CSV ──────────────────────────────────────────────────────────
    offender_log.to_csv(output_csv_path, index=False)
    logger.info(f"Offender log saved: {output_csv_path}")

    elapsed_total = time.time() - start_time
    return {
        "frames_processed": frame_number,
        "total_frames": total_frames,
        "duration_s": round(frame_number / fps, 2),
        "processing_time_s": round(elapsed_total, 1),
        "fps_processing": round(frame_number / elapsed_total, 2) if elapsed_total > 0 else 0,
        "offenders_count": len(offender_log),
        "speeding_count": int(offender_log["violation_type"].str.contains("speeding").sum()) if len(offender_log) else 0,
        "red_light_count": int(offender_log["violation_type"].str.contains("red_light").sum()) if len(offender_log) else 0,
        "output_video": output_video_path,
        "output_csv": output_csv_path,
        "offenders": offender_log.to_dict(orient="records"),
    }


def _run_ocr(offender_flags, snap_buf: SnapshotBuffer, speed_tracker: SpeedTracker,
             vehicle_type_map, fps, manager: ModelManager) -> pd.DataFrame:
    rows = []
    for tid, violations in offender_flags.items():
        vtype = vehicle_type_map.get(tid, "unknown")
        spd = speed_tracker.get_speed(tid)
        method = speed_tracker.get_method(tid)
        vtype_str = ",".join(sorted(violations))
        plate_text = "UNREADABLE"

        entry = snap_buf.get(tid)
        if entry and manager.ocr_reader:
            try:
                results = manager.ocr_reader.readtext(entry["plate_crop"])
                raw = ""
                for _, text, conf in results:
                    if conf > settings.OCR_CONFIDENCE_THRESHOLD:
                        raw += text.strip() + " "
                cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper().strip())
                plate_text = cleaned or "UNREADABLE"
            except Exception as exc:
                logger.warning(f"OCR error for track {tid}: {exc}")

        snap_path = None
        if entry and entry.get("vehicle_snapshot") is not None:
            os.makedirs(settings.SNAPSHOTS_DIR, exist_ok=True)
            safe = vtype_str.replace(",", "_")
            fn = f"track{tid}_{safe}_f{entry['snapshot_frame']}.jpg"
            snap_path = os.path.join(settings.SNAPSHOTS_DIR, fn)
            cv2.imwrite(snap_path, entry["vehicle_snapshot"])

        rows.append({
            "track_id": tid,
            "vehicle_type": vtype,
            "plate_number": plate_text,
            "speed_kmh": spd,
            "speed_method": method,
            "violation_type": vtype_str,
            "snapshot_path": snap_path,
            "timestamp_frame": entry["snapshot_frame"] if entry else 0,
            "timestamp_real": round((entry["snapshot_frame"] if entry else 0) / fps, 2),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "track_id", "vehicle_type", "plate_number", "speed_kmh",
        "speed_method", "violation_type", "snapshot_path",
        "timestamp_frame", "timestamp_real",
    ])


def _draw_info_bar(frame: np.ndarray, traffic_state: str, frame_num: int, total: int, fps: float) -> np.ndarray:
    """Draws a semi-transparent info bar at the top of the frame."""
    overlay = frame.copy()
    bar_h = 36
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    color = (0, 200, 80)  # green default
    if "red" in traffic_state.lower():
        color = (0, 0, 220)
    elif "yellow" in traffic_state.lower():
        color = (0, 200, 220)

    cv2.putText(frame, f"Light: {traffic_state.upper()}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    pct = int(100 * frame_num / total) if total > 0 else 0
    cv2.putText(frame, f"Frame {frame_num}/{total} ({pct}%)", (300, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    limit_txt = f"Limit: {settings.SPEED_LIMIT_KMH:.0f} km/h"
    cv2.putText(frame, limit_txt, (frame.shape[1] - 220, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame
