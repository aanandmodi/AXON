import numpy as np

class EMASmoother:
    """Exponential Moving Average filter for smoothing coordinates."""
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.prev_val = None

    def smooth(self, val: np.ndarray) -> np.ndarray:
        if self.prev_val is None:
            self.prev_val = np.array(val, dtype=np.float32)
            return self.prev_val
        smoothed = self.alpha * np.array(val, dtype=np.float32) + (1.0 - self.alpha) * self.prev_val
        self.prev_val = smoothed
        return smoothed

    def reset(self):
        self.prev_val = None

    def __repr__(self) -> str:
        return f"EMASmoother(alpha={self.alpha}, active={self.prev_val is not None})"


class OneEuroFilter:
    """One Euro filter for low-latency, jitter-free signal smoothing."""
    def __init__(self, t0: float, x0: np.ndarray, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = np.array(x0, dtype=np.float32)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = float(t0)

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        # Calculate derivative
        dx = (x - self.x_prev) / dt
        edx = self.dx_prev + self._alpha(self.dcutoff, dt) * (dx - self.dx_prev)
        
        # Calculate cutoff based on velocity
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        
        # Smooth signal
        rx = self.x_prev + self._alpha(cutoff, dt) * (x - self.x_prev)
        
        # Update history
        self.x_prev = rx
        self.dx_prev = edx
        self.t_prev = t
        return rx

    def __repr__(self) -> str:
        return f"OneEuroFilter(mincutoff={self.mincutoff}, beta={self.beta})"
