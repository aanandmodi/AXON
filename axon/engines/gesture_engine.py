import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict
from ..utils.logger import logger
from ..utils.smoother import EMASmoother
from ..utils.timer import VelocityTracker, CooldownTimer

@dataclass
class GestureResult:
    active_gesture: str
    hand_screen_x: int
    hand_screen_y: int
    pinch_distance: float
    is_pinching: bool
    swipe_direction: Optional[str]
    hand_velocity_x: float
    hand_velocity_y: float
    hand_type: str  # "Left" or "Right"


class GestureEngine:
    """Processes hand landmarks to classify gestures and map index fingertip to screen coordinates."""
    def __init__(self, config: dict):
        self.config = config
        self.gesture_config = config.get("gestures", {})
        self.monitors_config = config.get("monitors", {})
        
        # Smoothers for cursor movement (separate smoothers for Left and Right hands)
        alpha = self.gesture_config.get("cursor_ema_alpha", 0.3)
        self.smoothers = {
            "Left": EMASmoother(alpha=alpha),
            "Right": EMASmoother(alpha=alpha)
        }
        
        # Velocity trackers for swipe detection
        self.velocity_trackers = {
            "Left": VelocityTracker(history_size=5),
            "Right": VelocityTracker(history_size=5)
        }
        
        self.cooldown_timer = CooldownTimer()
        
        # Virtual desktop dimensions
        self.laptop_w = self.monitors_config.get("laptop_width", 1920)
        self.laptop_h = self.monitors_config.get("laptop_height", 1080)
        self.monitor_w = self.monitors_config.get("monitor_width", 1920)
        self.monitor_h = self.monitors_config.get("monitor_height", 1080)
        self.virtual_width = self.laptop_w + self.monitor_w
        self.virtual_height = max(self.laptop_h, self.monitor_h)
        
        # Landmark indices aliases
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.INDEX_PIP = 6
        self.INDEX_MCP = 5
        self.MIDDLE_TIP = 12
        self.MIDDLE_PIP = 10
        self.MIDDLE_MCP = 9
        self.RING_TIP = 16
        self.RING_PIP = 14
        self.RING_MCP = 13
        self.PINKY_TIP = 20
        self.PINKY_PIP = 18
        self.PINKY_MCP = 17

    def process(self, hand_result: Any, frame_width: int, frame_height: int) -> Dict[str, GestureResult]:
        """Processes hand landmarks from MediaPipe and returns active gestures and cursor coords."""
        results: Dict[str, GestureResult] = {}
        
        if not hand_result or not hand_result.hand_landmarks:
            return results
            
        # Iterate over all detected hands
        for i, landmarks in enumerate(hand_result.hand_landmarks):
            # Hand classification (Left vs Right)
            # hand_result.handedness: list of lists containing category name
            hand_type = "Right"
            if i < len(hand_result.handedness):
                hand_type = hand_result.handedness[i][0].category_name  # "Left" or "Right"
                
            # Convert to numpy array for vector operations
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            
            # 1. Classify Gesture using the rule_based module
            from ..gestures.rule_based import classify_gesture
            gesture, pinch_dist, is_pinching = classify_gesture(pts, hand_type, self.gesture_config.get("pinch_threshold", 0.05))

            # 4. Cursor position mapping
            # Extract Index fingertip raw coordinates (flip X coordinate since camera image is mirrored)
            raw_x = 1.0 - pts[self.INDEX_TIP][0]
            raw_y = pts[self.INDEX_TIP][1]
            
            # Apply EMA smoothing
            smoothed_coords = self.smoothers[hand_type].smooth(np.array([raw_x, raw_y]))
            sm_x, sm_y = smoothed_coords[0], smoothed_coords[1]
            
            # Map normalized [0.1, 0.9] of camera frame to full virtual screen
            norm_x = (sm_x - 0.1) / 0.8
            norm_y = (sm_y - 0.1) / 0.8
            
            # Clamp to [0.0, 1.0]
            norm_x = np.clip(norm_x, 0.0, 1.0)
            norm_y = np.clip(norm_y, 0.0, 1.0)
            
            # Calculate pixel position on the virtual desktop
            screen_x = int(norm_x * self.virtual_width)
            screen_y = int(norm_y * self.virtual_height)
            
            # 5. Swipe gesture tracking via wrist velocity
            wrist_x = 1.0 - pts[self.WRIST][0]
            wrist_y = pts[self.WRIST][1]
            
            # Convert wrist coords to virtual screen pixels for velocity calculation
            pixel_wrist_x = wrist_x * self.virtual_width
            pixel_wrist_y = wrist_y * self.virtual_height
            
            self.velocity_trackers[hand_type].add(pixel_wrist_x, pixel_wrist_y)
            vx, vy = self.velocity_trackers[hand_type].get_velocity()
            
            swipe_dir = None
            swipe_threshold = self.gesture_config.get("swipe_velocity_threshold", 800)
            
            cooldown_ms = self.gesture_config.get("gesture_cooldown_ms", 800)
            if not self.cooldown_timer.is_cooling_down(f"swipe_{hand_type}", cooldown_ms):
                if abs(vx) > swipe_threshold and abs(vx) > abs(vy):
                    swipe_dir = "RIGHT" if vx > 0 else "LEFT"
                    self.cooldown_timer.fire(f"swipe_{hand_type}")
                elif abs(vy) > swipe_threshold and abs(vy) > abs(vx):
                    swipe_dir = "DOWN" if vy > 0 else "UP"
                    self.cooldown_timer.fire(f"swipe_{hand_type}")

            results[hand_type] = GestureResult(
                active_gesture=gesture,
                hand_screen_x=screen_x,
                hand_screen_y=screen_y,
                pinch_distance=pinch_dist,
                is_pinching=is_pinching,
                swipe_direction=swipe_dir,
                hand_velocity_x=vx,
                hand_velocity_y=vy,
                hand_type=hand_type
            )
            
        return results

    def __repr__(self) -> str:
        return f"GestureEngine(virtual_w={self.virtual_width}, virtual_h={self.virtual_height})"
