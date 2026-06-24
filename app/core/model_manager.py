"""
ModelManager — loads and caches all ML models at startup.
CPU-mode notes:
  • RAFT large is replaced by Farneback dense optical flow (OpenCV, CPU-native).
  • Depth Anything V2 Small runs on CPU with torch.inference_mode().
  • EasyOCR is initialised without GPU.
  • YOLO models run on CPU with half=False.
"""

import os
import logging
from typing import Optional

import torch
import easyocr
from ultralytics import YOLO

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton-style container for all loaded models."""

    def __init__(self):
        self.traffic_model = None
        self.vehicle_model = None
        self.lp_model = None
        self.depth_processor = None
        self.depth_model = None
        self.raft_model = None
        self.raft_transforms = None
        self.ocr_reader = None
        self._loaded = False

    # ── Public ─────────────────────────────────────────────────────────────
    def load_all(self):
        if self._loaded:
            return
        self._load_yolo_models()
        self._load_depth_model()
        self._load_optical_flow()
        self._load_ocr()
        self._loaded = True

    def is_ready(self) -> bool:
        return self._loaded

    # ── YOLO ───────────────────────────────────────────────────────────────
    def _load_yolo_models(self):
        logger.info("Loading YOLO models...")

        for attr, path, label in [
            ("traffic_model", settings.TRAFFIC_LIGHT_WEIGHTS, "Traffic light"),
            ("vehicle_model", settings.VEHICLE_WEIGHTS, "Vehicle"),
            ("lp_model", settings.LICENSE_PLATE_WEIGHTS, "License plate"),
        ]:
            if not os.path.exists(path):
                logger.warning(
                    f"⚠️  {label} weights not found at '{path}'. "
                    "Download or train the model first. "
                    "Inference will skip this detector."
                )
                setattr(self, attr, None)
                continue

            model = YOLO(path)
            model.to(settings.DEVICE)
            setattr(self, attr, model)
            logger.info(f"  ✓ {label} model loaded — {path}")

    # ── Depth Anything V2 ──────────────────────────────────────────────────
    def _load_depth_model(self):
        logger.info("Loading Depth Anything V2 Small (CPU)...")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.depth_processor = AutoImageProcessor.from_pretrained(
                settings.DEPTH_MODEL_NAME
            )
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                settings.DEPTH_MODEL_NAME
            )
            self.depth_model.to(settings.DEVICE)
            self.depth_model.eval()
            logger.info("  ✓ Depth Anything V2 loaded.")
        except Exception as exc:
            logger.error(f"  ✗ Depth model failed to load: {exc}")
            self.depth_model = None
            self.depth_processor = None

    # ── Optical flow ───────────────────────────────────────────────────────
    def _load_optical_flow(self):
        if not settings.CPU_MODE:
            # GPU path — load RAFT large
            try:
                from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

                weights = Raft_Large_Weights.DEFAULT
                self.raft_transforms = weights.transforms()
                self.raft_model = raft_large(weights=weights, progress=False)
                self.raft_model.to(settings.DEVICE).eval()
                logger.info("  ✓ RAFT large optical flow loaded on GPU.")
            except Exception as exc:
                logger.warning(f"  ⚠️  RAFT failed: {exc}. Falling back to Farneback.")
                self.raft_model = None
        else:
            # CPU path — Farneback is orders-of-magnitude faster on CPU
            self.raft_model = None
            logger.info("  ✓ CPU mode: Farneback optical flow will be used (no RAFT).")

    # ── OCR ────────────────────────────────────────────────────────────────
    def _load_ocr(self):
        logger.info("Initialising EasyOCR (CPU)...")
        try:
            use_gpu = not settings.CPU_MODE
            self.ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
            logger.info(f"  ✓ EasyOCR ready (gpu={use_gpu}).")
        except Exception as exc:
            logger.error(f"  ✗ EasyOCR failed: {exc}")
            self.ocr_reader = None
