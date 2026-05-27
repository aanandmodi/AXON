import time
from typing import Dict, Tuple, Optional

class CooldownTimer:
    """Manages cooldowns for triggers to avoid multiple rapid executions."""
    def __init__(self):
        self.last_fired: Dict[str, float] = {}

    def is_cooling_down(self, action_id: str, cooldown_ms: float) -> bool:
        now = time.perf_counter() * 1000.0
        if action_id in self.last_fired:
            elapsed = now - self.last_fired[action_id]
            if elapsed < cooldown_ms:
                return True
        return False

    def fire(self, action_id: str):
        self.last_fired[action_id] = time.perf_counter() * 1000.0

    def reset(self):
        self.last_fired.clear()

    def __repr__(self) -> str:
        return f"CooldownTimer(active_keys={list(self.last_fired.keys())})"


class DwellTimer:
    """Tracks if an action has been held or dwelt on for a minimum duration."""
    def __init__(self):
        self.start_times: Dict[str, float] = {}

    def update(self, action_id: str, active: bool) -> bool:
        """Returns True if the action has been active for the threshold duration."""
        now = time.perf_counter() * 1000.0
        if active:
            if action_id not in self.start_times:
                self.start_times[action_id] = now
                return False
            else:
                return True
        else:
            self.start_times.pop(action_id, None)
            return False

    def get_elapsed(self, action_id: str) -> float:
        if action_id in self.start_times:
            return (time.perf_counter() * 1000.0) - self.start_times[action_id]
        return 0.0

    def reset(self, action_id: Optional[str] = None):
        if action_id:
            self.start_times.pop(action_id, None)
        else:
            self.start_times.clear()

    def __repr__(self) -> str:
        return f"DwellTimer(active_keys={list(self.start_times.keys())})"


class VelocityTracker:
    """Tracks the velocity of 2D coordinates over time."""
    def __init__(self, history_size: int = 5):
        self.history_size = history_size
        self.history: list[Tuple[float, float, float]] = []  # List of (t, x, y)

    def add(self, x: float, y: float):
        now = time.perf_counter()
        self.history.append((now, x, y))
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def get_velocity(self) -> Tuple[float, float]:
        """Returns velocity in pixels per second (vx, vy)."""
        if len(self.history) < 2:
            return 0.0, 0.0
        
        dt = self.history[-1][0] - self.history[0][0]
        if dt <= 0:
            return 0.0, 0.0
            
        dx = self.history[-1][1] - self.history[0][1]
        dy = self.history[-1][2] - self.history[0][2]
        
        return dx / dt, dy / dt

    def clear(self):
        self.history.clear()

    def __repr__(self) -> str:
        return f"VelocityTracker(points={len(self.history)})"
