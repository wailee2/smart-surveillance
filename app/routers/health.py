"""Health check endpoint."""

from fastapi import APIRouter, Request
from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    manager = getattr(request.app.state, "model_manager", None)
    return {
        "status": "ok",
        "device": settings.DEVICE,
        "cpu_mode": settings.CPU_MODE,
        "models_loaded": manager.is_ready() if manager else False,
        "traffic_model": manager.traffic_model is not None if manager else False,
        "vehicle_model": manager.vehicle_model is not None if manager else False,
        "lp_model": manager.lp_model is not None if manager else False,
        "depth_model": manager.depth_model is not None if manager else False,
        "ocr": manager.ocr_reader is not None if manager else False,
    }
