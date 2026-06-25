"""
Virtual Traffic Light (VTL) State
Thread-safe singleton that holds the operator-controlled signal.
The pipeline reads this when the physical traffic model is absent or overridden.
"""

import threading

_lock = threading.Lock()
_state = {
    "color": "green",      # "red" | "amber" | "green"
    "override": False,      # True = VTL overrides model detection
}


def set_color(color: str, override: bool = True):
    assert color in ("red", "amber", "green")
    with _lock:
        _state["color"] = color
        _state["override"] = override


def get_state() -> dict:
    with _lock:
        return dict(_state)


def is_red() -> bool:
    with _lock:
        return _state["color"] == "red"
