"""Sessions router — history panel API."""

from fastapi import APIRouter
from app.core.session_store import get_all, get_session

router = APIRouter()


@router.get("/sessions")
async def list_sessions():
    return get_all()


@router.get("/sessions/{job_id}")
async def get_session_detail(job_id: str):
    s = get_session(job_id)
    if not s:
        from fastapi import HTTPException
        raise HTTPException(404, "Session not found.")
    return s
