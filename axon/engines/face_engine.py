import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from ..utils.logger import logger
from ..utils.timer import CooldownTimer, DwellTimer, VelocityTracker

@dataclass
class FaceAction:
    scroll_direction: Optional[str]
    scroll_speed: int
    action_triggered: Optional[str]
    dominant_emotion: Optional[str] = None


class FaceEngine:
    """Classifies facial actions like head nodding, head shaking, smiling, and eyebrow raises into OS commands."""
    def __init__(self, config: dict):
        self.config = config
        self.face_config = config.get("face", {})
        self.gaze_config = config.get("gaze", {})
        
        self.cooldown_timer = CooldownTimer()
        self.dwell_timer = DwellTimer()
        
        # Track velocity of head rotation for nods and shakes
        self.yaw_tracker = VelocityTracker(history_size=6)
        self.pitch_tracker = VelocityTracker(history_size=6)
        
        # For tracking state sequences (e.g., nod = pitch down then pitch up)
        self.last_pitch_state = "neutral"
        self.last_yaw_state = "neutral"
        self.shake_direction_changes = 0
        self.last_shake_time = 0.0

    def process(self, gaze_result: Any, face_result: Any) -> FaceAction:
        """Analyzes GazeResult and face blendshapes to determine face-driven actions."""
        scroll_dir = None
        scroll_speed = 0
        action = None
        
        if gaze_result is None:
            return FaceAction(scroll_direction=None, scroll_speed=0, action_triggered=None)
            
        now = time.perf_counter()
        cooldown_ms = self.face_config.get("action_cooldown_ms", 1500)
        
        # 1. Head Tilt (Roll) -> Scrolling
        roll = gaze_result.head_roll
        tilt_threshold = self.face_config.get("head_tilt_scroll_threshold_deg", 15)
        
        # Positive roll is tilting left, negative is right
        if roll > tilt_threshold:
            scroll_dir = "UP"
            scroll_speed = int(min((roll - tilt_threshold) / 3.0 + 1.0, 10.0))
        elif roll < -tilt_threshold:
            scroll_dir = "DOWN"
            scroll_speed = int(min((abs(roll) - tilt_threshold) / 3.0 + 1.0, 10.0))
            
        # 2. Head Shake -> Close Window (Yaw velocity)
        yaw = gaze_result.head_yaw
        self.yaw_tracker.add(yaw, 0.0)
        vy, _ = self.yaw_tracker.get_velocity()
        
        shake_threshold = self.face_config.get("head_shake_velocity_threshold", 150)
        if abs(vy) > shake_threshold:
            current_yaw_dir = "right" if vy > 0 else "left"
            if self.last_yaw_state != "neutral" and self.last_yaw_state != current_yaw_dir:
                # Direction reversal detected during fast yaw movement
                self.shake_direction_changes += 1
                self.last_shake_time = now
            self.last_yaw_state = current_yaw_dir
        else:
            if now - self.last_shake_time > 0.5:
                # Reset shake counter after 500ms of quiet yaw
                self.shake_direction_changes = 0
                self.last_yaw_state = "neutral"
                
        # Trigger CLOSE_WINDOW if there are at least 3 direction reversals in a quick shake
        if self.shake_direction_changes >= 3:
            if not self.cooldown_timer.is_cooling_down("CLOSE_WINDOW", cooldown_ms):
                action = "CLOSE_WINDOW"
                self.cooldown_timer.fire("CLOSE_WINDOW")
                self.shake_direction_changes = 0
                
        # 3. Head Nod -> Confirm/Enter (Pitch velocity)
        pitch = gaze_result.head_pitch
        self.pitch_tracker.add(pitch, 0.0)
        vp, _ = self.pitch_tracker.get_velocity()
        
        nod_threshold = self.face_config.get("nod_velocity_threshold", 120)
        if vp < -nod_threshold:  # Pitching downwards quickly
            self.last_pitch_state = "down"
        elif vp > nod_threshold and self.last_pitch_state == "down":  # Rebounding back up
            if not self.cooldown_timer.is_cooling_down("CONFIRM", cooldown_ms):
                action = "CONFIRM"
                self.cooldown_timer.fire("CONFIRM")
            self.last_pitch_state = "neutral"
        else:
            if abs(vp) < 30:  # Back to stationary
                self.last_pitch_state = "neutral"

        # 4. Eyebrow Raise actions
        if gaze_result.eyebrows_raised:
            # Check gaze zone
            # Combine with look left/right to trigger browser actions
            # Or default to right click context menu if gaze is neutral
            if not self.cooldown_timer.is_cooling_down("EYEBROW_ACTION", cooldown_ms):
                # We can approximate horizontal gaze relative center: normal center around 0.5
                # Look hard left: relative_x < 0.4. Look hard right: relative_x > 0.6.
                rel_x = gaze_result.iris_relative_x
                if rel_x < 0.43:
                    action = "BROWSER_BACK"
                elif rel_x > 0.57:
                    action = "BROWSER_FORWARD"
                else:
                    action = "RIGHT_CLICK"
                self.cooldown_timer.fire("EYEBROW_ACTION")
                
        # 5. Mouth Open -> Voice Activation
        if gaze_result.mouth_open:
            if self.dwell_timer.update("voice_act", True):
                if self.dwell_timer.get_elapsed("voice_act") > 300.0:
                    if not self.cooldown_timer.is_cooling_down("VOICE_ACTIVATE", cooldown_ms):
                        action = "VOICE_ACTIVATE"
                        self.cooldown_timer.fire("VOICE_ACTIVATE")
        else:
            self.dwell_timer.update("voice_act", False)
            
        # 6. Smile -> Screenshot
        # We can extract smile score from face blendshapes if available
        smile_score = 0.0
        if face_result and face_result.face_blendshapes:
            blendshapes = face_result.face_blendshapes[0]
            smile_left = 0.0
            smile_right = 0.0
            for b in blendshapes:
                if b.category_name == "mouthSmileLeft":
                    smile_left = b.score
                elif b.category_name == "mouthSmileRight":
                    smile_right = b.score
            smile_score = (smile_left + smile_right) / 2.0
            
        smile_active = smile_score > self.face_config.get("smile_threshold", 0.48)
        if smile_active:
            if self.dwell_timer.update("smile", True):
                if self.dwell_timer.get_elapsed("smile") > self.face_config.get("smile_duration_ms", 1500):
                    if not self.cooldown_timer.is_cooling_down("SCREENSHOT", cooldown_ms):
                        action = "SCREENSHOT"
                        self.cooldown_timer.fire("SCREENSHOT")
                        self.dwell_timer.reset("smile")  # Trigger once per hold
        else:
            self.dwell_timer.update("smile", False)

        return FaceAction(
            scroll_direction=scroll_dir,
            scroll_speed=scroll_speed,
            action_triggered=action
        )

    def __repr__(self) -> str:
        return "FaceEngine()"
