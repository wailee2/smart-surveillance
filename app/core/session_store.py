"""
Session Store — persists job history to a local JSON file so the
sidebar panel survives server restarts.
"""

import os
import json
import time
import threading
from typing import Optional

SESSION_FILE = "static/outputs/sessions.json"
_lock = threading.Lock()


def _load() -> list:
    if not os.path.exists(SESSION_FILE):
        return []
    try:
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save(sessions: list):
    with open(SESSION_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


def add_session(job_id: str, filename: str, meta: dict):
    with _lock:
        sessions = _load()
        sessions.insert(0, {
            "job_id": job_id,
            "filename": filename,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
            **meta,
        })
        # Keep last 50 sessions
        sessions = sessions[:50]
        _save(sessions)


def update_session(job_id: str, updates: dict):
    with _lock:
        sessions = _load()
        for s in sessions:
            if s["job_id"] == job_id:
                s.update(updates)
                break
        _save(sessions)


def get_all() -> list:
    with _lock:
        return _load()


def get_session(job_id: str) -> Optional[dict]:
    with _lock:
        for s in _load():
            if s["job_id"] == job_id:
                return s
        return None
