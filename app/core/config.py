"""
Configuration — all tunable parameters in one place.
Edit this file or override via environment variables.
"""

import os
import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────
    APP_NAME: str = "Smart Surveillance & Speed Detection"
    DEBUG: bool = False

    # ── Paths ─────────────────────────────────────────────────────────────
    WEIGHTS_DIR: str = "weights"
    UPLOAD_DIR: str = "static/uploads"
    OUTPUT_DIR: str = "static/outputs"
    SNAPSHOTS_DIR: str = "static/outputs/snapshots"

    # ── Device (CPU only when no CUDA is available) ───────────────────────
    DEVICE: str = "cpu"          # override to "cuda:0" if GPU is present
    CPU_MODE: bool = True        # disables RAFT iterations for speed on CPU

    # ── Model weights ─────────────────────────────────────────────────────
    TRAFFIC_LIGHT_WEIGHTS: str = "weights/traffic_light_best.pt"
    VEHICLE_WEIGHTS: str = "weights/vehicle_best.pt"
    LICENSE_PLATE_WEIGHTS: str = "weights/license_plate_best.pt"

    # ── Depth model ───────────────────────────────────────────────────────
    DEPTH_MODEL_NAME: str = "depth-anything/Depth-Anything-V2-Small-hf"

    # ── Detection thresholds ──────────────────────────────────────────────
    CONF_THRESHOLD: float = 0.4
    IOU_THRESHOLD: float = 0.5

    # ── Speed & tracking ─────────────────────────────────────────────────
    SPEED_LIMIT_KMH: float = 50.0
    MIN_TRAJ_POINTS: int = 6
    MIN_FLOW_PIXELS: float = 1.5
    SPEED_DISPLAY_DELTA_KMH: float = 2.0

    # ── Kalman filter ─────────────────────────────────────────────────────
    KALMAN_PROCESS_NOISE: float = 0.5
    KALMAN_MEASUREMENT_NOISE: float = 8.0
    KALMAN_INITIAL_VARIANCE: float = 10.0

    # ── EMA smoothing ─────────────────────────────────────────────────────
    EMA_ALPHA_BOX: float = 0.15
    EMA_ALPHA_DEPTH: float = 0.10
    EMA_ALPHA_SCALE: float = 0.15

    # ── Median buffer ─────────────────────────────────────────────────────
    MEDIAN_BUFFER_FRAMES: int = 9

    # ── Stationary detection ──────────────────────────────────────────────
    STATIONARY_CHECK_SECONDS: float = 0.5
    STATIONARY_PIXEL_THRESHOLD: float = 3.0
    STATIONARY_RELATIVE_FRACTION: float = 0.03

    # ── RAFT / Depth update rates ─────────────────────────────────────────
    # CPU mode uses larger intervals to keep processing tractable.
    FRAME_INTERVAL_CPU: int = 30        # compare frame N vs N-30 on CPU
    FRAME_INTERVAL_GPU: int = 20
    DEPTH_EVERY_N_CPU: int = 40         # run depth model every 40 frames
    DEPTH_EVERY_N_GPU: int = 20

    # ── Vehicle priors ────────────────────────────────────────────────────
    VEHICLE_HEIGHT_M: dict = {
        "car": 1.5,
        "truck": 2.5,
        "bus": 3.0,
        "motorcycle": 1.2,
        "bike": 1.2,
        "3wheeler": 1.4,
        "default": 1.5,
    }
    VEHICLE_ALLOWED_CLASSES: set = {"3wheeler", "car", "truck", "bus", "bike", "motorcycle"}

    # ── OCR ───────────────────────────────────────────────────────────────
    OCR_CONFIDENCE_THRESHOLD: float = 0.4
    OCR_DEBUG_LOGGING: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_frame_interval(self) -> int:
        return self.FRAME_INTERVAL_CPU if self.CPU_MODE else self.FRAME_INTERVAL_GPU

    def get_depth_every_n(self) -> int:
        return self.DEPTH_EVERY_N_CPU if self.CPU_MODE else self.DEPTH_EVERY_N_GPU


def _detect_device() -> tuple[str, bool]:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda:0", False
    return "cpu", True


_device, _cpu = _detect_device()

settings = Settings(DEVICE=_device, CPU_MODE=_cpu)
