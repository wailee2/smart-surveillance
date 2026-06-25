"""
Smart Surveillance & Speed Detection System — FastAPI Entry Point
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import inference, health, stream, sessions
from app.core.config import settings
from app.core.model_manager import ModelManager

# ── Ensure dirs exist before mounts ───────────────────────────────────────────
for _dir in [
    "static", "static/uploads", "static/outputs",
    "static/outputs/snapshots", "weights", "demo",
]:
    os.makedirs(_dir, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Smart Surveillance System...")
    print(f"   Device : {settings.DEVICE}  |  CPU mode : {settings.CPU_MODE}")
    manager = ModelManager()
    manager.load_all()
    app.state.model_manager = manager
    print("✅ Models loaded. Ready.")
    yield
    print("🛑 Shutdown.")


app = FastAPI(
    title="Smart Surveillance & Speed Detection",
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static",   StaticFiles(directory="static"),          name="static")
app.mount("/outputs",  StaticFiles(directory="static/outputs"),  name="outputs")
app.mount("/uploads",  StaticFiles(directory="static/uploads"),  name="uploads")

app.include_router(health.router,    tags=["Health"])
app.include_router(inference.router, prefix="/api", tags=["Inference"])
app.include_router(stream.router,    prefix="/api", tags=["Stream"])
app.include_router(sessions.router,  prefix="/api", tags=["Sessions"])

templates = Jinja2Templates(directory="templates")


@app.get("/", include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )
