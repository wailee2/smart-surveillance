"""
Smart Surveillance & Speed Detection System
FastAPI Application Entry Point
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import inference, health
from app.core.config import settings
from app.core.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    print("🚀 Starting Smart Surveillance System...")
    print(f"   Device   : {settings.DEVICE}")
    print(f"   CPU mode : {settings.CPU_MODE}")
    print(f"   Weights  : {settings.WEIGHTS_DIR}")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.SNAPSHOTS_DIR, exist_ok=True)

    # Pre-load models at startup
    manager = ModelManager()
    manager.load_all()
    app.state.model_manager = manager

    print("✅ Models loaded. Ready to receive requests.")
    yield

    print("🛑 Shutting down Smart Surveillance System.")


# Ensure directories exist BEFORE mounting (StaticFiles crashes otherwise)
for _dir in [
    "static",
    "static/uploads",
    "static/outputs",
    "static/outputs/snapshots",
    "weights",
    "demo",
]:
    os.makedirs(_dir, exist_ok=True)

app = FastAPI(
    title="Smart Surveillance & Speed Detection",
    description="YOLOv11 + optical flow + Depth Anything V2 for vehicle speed and violation detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="static/outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="static/uploads"), name="uploads")

# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(inference.router, prefix="/api", tags=["Inference"])

# Templates
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request}
    )
