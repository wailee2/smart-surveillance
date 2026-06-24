"""
Speed Estimation Pipeline
CPU-adapted version of the original notebook pipeline.

Key CPU adaptations:
  1. Farneback dense optical flow replaces RAFT (10-50× faster on CPU).
  2. Depth Anything V2 runs with torch.inference_mode() and smaller crops.
  3. All torch operations stay on CPU (no .to('cuda') calls).
  4. Frame-skip intervals are larger to keep real-time throughput viable.
"""

import re
import os
import cv2
import numpy as np
import torch
import logging
from collections import defaultdict, deque
from typing import Optional

import supervision as sv

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── EMA Filter ────────────────────────────────────────────────────────────────
class EMAFilter:
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.value = None

    def update(self, v: float) -> float:
        self.value = v if self.value is None else self.alpha * v + (1 - self.alpha) * self.value
        return self.value


# ── Kalman Speed Filter ───────────────────────────────────────────────────────
class KalmanSpeedFilter:
    def __init__(self, Q=0.5, R=8.0, P0=10.0):
        self.Q = Q
        self.R = R
        self.x = None
        self.P = P0

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z
            return max(z, 0.0)
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return max(self.x, 0.0)

    def get(self) -> Optional[float]:
        return max(self.x, 0.0) if self.x is not None else None


# ── Median Speed Buffer ───────────────────────────────────────────────────────
class MedianSpeedBuffer:
    def __init__(self, n: int = 9):
        self.n = n
        self.bufs: dict[int, deque] = defaultdict(lambda: deque(maxlen=n))

    def push(self, tid: int, v: float):
        self.bufs[tid].append(v)

    def get(self, tid: int) -> Optional[float]:
        buf = self.bufs[tid]
        return float(np.median(list(buf))) if buf else None


# ── Depth Scale Estimator ─────────────────────────────────────────────────────
class DepthScaleEstimator:
    """
    Computes metres-per-pixel using Depth Anything V2 relative depth
    combined with known vehicle class height priors.
    """

    def __init__(self, depth_model, depth_processor, frame_h: int, frame_w: int):
        self.depth_model = depth_model
        self.depth_processor = depth_processor
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.current_depth: Optional[np.ndarray] = None
        self.frame_counter = 0
        self.depth_every_n = settings.get_depth_every_n()

        self._depth_ema: dict[int, EMAFilter] = {}
        self._box_ema: dict[int, EMAFilter] = {}
        self._scale_ema: dict[int, EMAFilter] = {}

    def update_depth(self, frame: np.ndarray):
        self.frame_counter += 1
        if self.frame_counter % self.depth_every_n != 0:
            return
        if self.depth_model is None:
            return

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            inputs = self.depth_processor(images=img_rgb, return_tensors="pt")
            inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.depth_model(**inputs)
                predicted = outputs.predicted_depth

            depth = torch.nn.functional.interpolate(
                predicted.unsqueeze(1),
                size=(self.frame_h, self.frame_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            d_min, d_max = depth.min(), depth.max()
            self.current_depth = (
                (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)
            )
        except Exception as exc:
            logger.warning(f"Depth estimation failed: {exc}")

    def get_scale(self, tid: int, cx: float, cy: float, bh: float, cls_name: str) -> float:
        box_ema = self._box_ema.setdefault(tid, EMAFilter(settings.EMA_ALPHA_BOX))
        smooth_bh = max(box_ema.update(bh), 1.0)

        real_h = settings.VEHICLE_HEIGHT_M.get(cls_name.lower(), settings.VEHICLE_HEIGHT_M["default"])

        raw_depth = 1.0
        if self.current_depth is not None:
            cy_i = min(int(cy), self.current_depth.shape[0] - 1)
            cx_i = min(int(cx), self.current_depth.shape[1] - 1)
            raw_depth = max(float(self.current_depth[cy_i, cx_i]), 0.01)

        depth_ema = self._depth_ema.setdefault(tid, EMAFilter(settings.EMA_ALPHA_DEPTH))
        smooth_d = depth_ema.update(raw_depth)

        raw_mpp = (real_h / smooth_bh) * smooth_d
        scale_ema = self._scale_ema.setdefault(tid, EMAFilter(settings.EMA_ALPHA_SCALE))
        return scale_ema.update(raw_mpp)


# ── Motion Estimator (Farneback CPU / RAFT GPU) ───────────────────────────────
class MotionEstimator:
    """
    CPU mode  → OpenCV Farneback dense optical flow (fast, good enough for speed estimation).
    GPU mode  → RAFT large (accurate, requires CUDA).
    """

    def __init__(self, raft_model=None, raft_transforms=None, frame_interval: int = 20):
        self.use_raft = (raft_model is not None) and (not settings.CPU_MODE)
        self.raft_model = raft_model
        self.raft_transforms = raft_transforms
        self.frame_interval = frame_interval

        self.frame_counter = 0
        self.current_flow: Optional[np.ndarray] = None
        self.failed = False
        self._buffer: deque = deque(maxlen=frame_interval + 1)

    def update(self, frame: np.ndarray):
        self.frame_counter += 1
        self._buffer.append(frame.copy())

        if len(self._buffer) < self.frame_interval + 1:
            return

        ref = self._buffer[0]
        cur = frame

        if self.use_raft:
            self._run_raft(ref, cur)
        else:
            self._run_farneback(ref, cur)

    def _run_farneback(self, img1: np.ndarray, img2: np.ndarray):
        try:
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            self.current_flow = flow
        except Exception as exc:
            logger.warning(f"Farneback flow failed: {exc}")
            self.current_flow = None

    def _run_raft(self, img1: np.ndarray, img2: np.ndarray):
        import torchvision.transforms.functional as TF

        try:
            h, w = img1.shape[:2]
            new_h, new_w = (h // 8) * 8, (w // 8) * 8

            t1 = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float()
            t2 = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float()
            t1 = TF.resize(t1, [new_h, new_w], antialias=False)
            t2 = TF.resize(t2, [new_h, new_w], antialias=False)
            t1, t2 = self.raft_transforms(t1, t2)

            with torch.no_grad():
                flows = self.raft_model(t1.to(settings.DEVICE), t2.to(settings.DEVICE))

            flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()
            flow_resized = cv2.resize(flow, (w, h))
            flow_resized[..., 0] *= w / new_w
            flow_resized[..., 1] *= h / new_h
            self.current_flow = flow_resized
        except Exception as exc:
            logger.warning(f"RAFT failed: {exc}. Falling back to Farneback.")
            self._run_farneback(img1, img2)

    def get_box_displacement(self, x1, y1, x2, y2):
        if self.current_flow is None:
            return None, None
        h, w = self.current_flow.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        x1c = max(0, int(x1 + bw * 0.2))
        y1c = max(0, int(y1 + bh * 0.2))
        x2c = min(w, int(x2 - bw * 0.2))
        y2c = min(h, int(y2 - bh * 0.2))
        if x2c <= x1c or y2c <= y1c:
            return None, None
        region = self.current_flow[y1c:y2c, x1c:x2c]
        return float(np.median(region[..., 0])), float(np.median(region[..., 1]))


# ── Speed Tracker ─────────────────────────────────────────────────────────────
class SpeedTracker:
    def __init__(self, fps: float, scale_est: DepthScaleEstimator, motion_est: MotionEstimator):
        self.fps = fps
        self.estimator = scale_est
        self.motion = motion_est

        self._traj: dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self._kalman: dict[int, KalmanSpeedFilter] = {}
        self._median = MedianSpeedBuffer(n=settings.MEDIAN_BUFFER_FRAMES)
        self._display_speeds: dict[int, float] = {}
        self._internal_speeds: dict[int, float] = {}
        self._methods: dict[int, str] = {}

    def _is_stationary(self, tid: int, cx: float, cy: float, bh: float) -> bool:
        traj = self._traj.get(tid)
        n = max(int(round(settings.STATIONARY_CHECK_SECONDS * self.fps)), 3)
        if not traj or len(traj) < n:
            return False
        recent = list(traj)[-n:]
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        spread = np.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
        threshold = max(settings.STATIONARY_PIXEL_THRESHOLD, bh * settings.STATIONARY_RELATIVE_FRACTION)
        return spread < threshold

    def update(self, detections: sv.Detections, frame: np.ndarray, frame_number: int, vehicle_model):
        self.estimator.update_depth(frame)
        self.motion.update(frame)

        if detections.tracker_id is None:
            return

        for i, tid in enumerate(detections.tracker_id):
            cls_name = vehicle_model.names[detections.class_id[i]]
            if cls_name.lower() not in settings.VEHICLE_ALLOWED_CLASSES:
                continue

            x1, y1, x2, y2 = detections.xyxy[i]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            bh = float(y2 - y1)

            self._traj[tid].append((frame_number, cx, cy))
            stationary = self._is_stationary(tid, cx, cy, bh)

            speed_raw = None

            if stationary:
                speed_raw = 0.0
            else:
                elapsed = max(self.motion.frame_interval, 1) / self.fps
                mpp = self.estimator.get_scale(tid, cx, cy, bh, cls_name)
                dx, dy = self.motion.get_box_displacement(x1, y1, x2, y2)

                if dx is not None and mpp is not None and elapsed > 0:
                    mag = np.sqrt(dx ** 2 + dy ** 2)
                    if mag >= settings.MIN_FLOW_PIXELS:
                        speed_raw = min(max((mag * mpp / elapsed) * 3.6, 0.0), 250.0)
                        self._methods[tid] = "depth_v2+farneback" if settings.CPU_MODE else "depth_v2+raft"

            if speed_raw is not None:
                if tid not in self._kalman:
                    self._kalman[tid] = KalmanSpeedFilter(
                        settings.KALMAN_PROCESS_NOISE,
                        settings.KALMAN_MEASUREMENT_NOISE,
                        settings.KALMAN_INITIAL_VARIANCE,
                    )
                k_speed = self._kalman[tid].update(speed_raw)
                self._internal_speeds[tid] = round(k_speed, 1)

                self._median.push(tid, k_speed)
                med = self._median.get(tid)

                prev = self._display_speeds.get(tid)
                if prev is None or abs(med - prev) >= settings.SPEED_DISPLAY_DELTA_KMH:
                    self._display_speeds[tid] = round(med, 1)

    def get_speed(self, tid: int) -> Optional[float]:
        return self._display_speeds.get(tid)

    def get_internal_speed(self, tid: int) -> Optional[float]:
        return self._internal_speeds.get(tid)

    def get_method(self, tid: int) -> str:
        return self._methods.get(tid, "unknown")
