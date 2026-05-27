import queue
import threading
import time
import subprocess
import os
from typing import Optional, Any, Dict
from ..utils.logger import logger
from ..utils.timer import CooldownTimer, DwellTimer
from ..engines.gaze_engine import GazeEngine, GazeResult
from ..engines.gesture_engine import GestureEngine, GestureResult
from ..engines.face_engine import FaceEngine, FaceAction
from ..engines.depth_engine import DepthEngine
from ..control.mouse_controller import MouseController
from ..control.keyboard_controller import KeyboardController
from ..control.window_manager import WindowManager
from ..control.audio_controller import AudioController

class ActionRouter:
    """Orchestrates incoming MediaPipe results, processes them through the engines, and routes them to OS controls."""
    def __init__(self, config: dict, result_queue: queue.Queue, state_callback=None):
        self.config = config
        self.result_queue = result_queue
        self.state_callback = state_callback
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Engines
        self.gaze_engine = GazeEngine(config)
        self.gesture_engine = GestureEngine(config)
        self.face_engine = FaceEngine(config)
        self.depth_engine = DepthEngine(config)
        
        # Controllers
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.window = WindowManager(config)
        self.audio = AudioController()
        
        # Global states
        self.freeze_mode = False
        self.drawing_mode = False
        
        # Dwell timers
        self.dwell_timer = DwellTimer()
        self.cooldown_timer = CooldownTimer()
        
        # Air drawing parameters
        self.drawing_points = []
        self.current_drawing_color = "red"
        
        # Last states for overlay HUD feedback
        self.last_gesture = "NONE"
        self.last_action = "NONE"
        self.fps_inference = 0.0
        self.fps_display = 0.0
        self.gaze_dot = (0, 0)
        self.camera_status = {"laptop": True, "phone": False}

    def start(self):
        """Starts the action router thread."""
        self.running = True
        self.thread = threading.Thread(target=self._route_actions, daemon=True, name="ActionRouter")
        self.thread.start()

    def stop(self):
        """Stops the action router thread and cleans up controllers."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.audio.cleanup()

    def _route_actions(self):
        last_frame_time = time.perf_counter()
        
        while self.running:
            try:
                # Retrieve inference result, blocking up to 100ms
                res = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            now = time.perf_counter()
            # Calculate inference FPS
            self.fps_inference = 1.0 / max(now - last_frame_time, 0.001)
            last_frame_time = now
            
            # Extract resolution
            h, w, _ = res.laptop_frame.shape
            
            # Update camera status
            self.camera_status["laptop"] = res.laptop_frame is not None
            self.camera_status["phone"] = res.phone_frame is not None
            
            # 1. RUN ENGINES
            gaze_res = self.gaze_engine.process(res.face_result, w, h)
            gesture_res_dict = self.gesture_engine.process(res.hand_result, w, h)
            face_act = self.face_engine.process(gaze_res, res.face_result)
            
            # Detect primary hand results (prefer Right hand for mouse navigation)
            primary_hand = "Right" if "Right" in gesture_res_dict else ("Left" if "Left" in gesture_res_dict else None)
            hand_res = gesture_res_dict.get(primary_hand) if primary_hand else None
            
            # Triangulate hand depth
            laptop_hand_lms = res.hand_result.hand_landmarks[0] if res.hand_result and res.hand_result.hand_landmarks else None
            hand_depth = self.depth_engine.process(laptop_hand_lms, res.phone_frame)
            
            # Gaze pointer coordinates
            if gaze_res:
                self.gaze_dot = (gaze_res.gaze_screen_x, gaze_res.gaze_screen_y)
                
            # 2. PRIORITY CASCADE AND DISPATCH
            
            # A. FREEZE MODE TOGGLE (Open palm held for 2s)
            palm_active = (hand_res is not None and hand_res.active_gesture == "OPEN_PALM")
            if palm_active:
                elapsed = self.dwell_timer.get_elapsed("freeze_toggle")
                # Update dwell status
                self.dwell_timer.update("freeze_toggle", True)
                if elapsed > self.config.get("gestures", {}).get("freeze_hold_duration_ms", 2000):
                    if not self.cooldown_timer.is_cooling_down("FREEZE_TOGGLE", 1500):
                        self.freeze_mode = not self.freeze_mode
                        logger.info(f"Freeze mode toggled: {self.freeze_mode}")
                        self.last_action = "TOGGLE_FREEZE"
                        self.cooldown_timer.fire("FREEZE_TOGGLE")
                        self.dwell_timer.reset("freeze_toggle")
            else:
                self.dwell_timer.update("freeze_toggle", False)
                
            # If Freeze Mode is active, bypass all subsequent OS triggers
            if self.freeze_mode:
                self.last_gesture = "FREEZE"
                self._update_hud_state()
                continue
                
            # B. DRAWING MODE TOGGLE (Peace sign held for 1.5s)
            peace_active = (hand_res is not None and hand_res.active_gesture == "PEACE")
            if peace_active:
                # Cycle colors or enter whiteboard drawing mode
                if self.dwell_timer.update("drawing_toggle", True):
                    if self.dwell_timer.get_elapsed("drawing_toggle") > 1500:
                        if not self.cooldown_timer.is_cooling_down("DRAWING_TOGGLE", 1500):
                            self.drawing_mode = not self.drawing_mode
                            self.drawing_points.clear()
                            logger.info(f"Drawing mode toggled: {self.drawing_mode}")
                            self.last_action = "TOGGLE_DRAWING"
                            self.cooldown_timer.fire("DRAWING_TOGGLE")
                            self.dwell_timer.reset("drawing_toggle")
            else:
                self.dwell_timer.update("drawing_toggle", False)

            # C. HAND GESTURE ACTIONS
            if hand_res:
                self.last_gesture = hand_res.active_gesture
                
                # Check swipe desk gestures
                if hand_res.swipe_direction:
                    dir_name = hand_res.swipe_direction
                    self.last_action = f"SWIPE_{dir_name}"
                    if dir_name == "LEFT":
                        self.keyboard.hotkey("ctrl", "win", "left")  # Switch virtual desktops
                    elif dir_name == "RIGHT":
                        self.keyboard.hotkey("ctrl", "win", "right")
                        
                # Drawing on Air Whiteboard
                elif self.drawing_mode and hand_res.active_gesture in ["AIR_MOUSE", "PINCH"]:
                    # In drawing mode, track index tip and add to canvas path
                    self.drawing_points.append((hand_res.hand_screen_x, hand_res.hand_screen_y))
                    # Clear drawing if two-hand spread is detected (which maps to OPEN_PALM on both hands)
                    if len(gesture_res_dict) == 2 and all(g.active_gesture == "OPEN_PALM" for g in gesture_res_dict.values()):
                        self.drawing_points.clear()
                        self.last_action = "CLEAR_CANVAS"
                        
                # Air Mouse cursor control
                elif hand_res.active_gesture == "AIR_MOUSE":
                    self.mouse.move_to(hand_res.hand_screen_x, hand_res.hand_screen_y)
                    
                # Pinch resizing window
                elif hand_res.active_gesture == "PINCH":
                    self.mouse.move_to(hand_res.hand_screen_x, hand_res.hand_screen_y)
                    # Window pinch resizing trigger
                    # If pinch-drag is sustained, resize active window based on movement
                    # For simplicity: drag moves mouse with left click down
                    # We can click and drag, or resize window. Let's do click and drag.
                    self.mouse.press("left")
                    self.last_action = "DRAG"
                else:
                    self.mouse.release("left")
                    
                # Rock gesture: Open Terminal
                if hand_res.active_gesture == "ROCK":
                    if not self.cooldown_timer.is_cooling_down("OPEN_TERMINAL", 2000):
                        subprocess.Popen("cmd.exe")  # Open windows command prompt
                        self.last_action = "OPEN_TERMINAL"
                        self.cooldown_timer.fire("OPEN_TERMINAL")
                        
                # Thumbs Up / Down: Volume Up / Down
                elif hand_res.active_gesture == "THUMBS_UP":
                    if not self.cooldown_timer.is_cooling_down("VOLUME_UP", 200):
                        self.audio.volume_up(0.05)
                        self.last_action = "VOLUME_UP"
                        self.cooldown_timer.fire("VOLUME_UP")
                elif hand_res.active_gesture == "THUMBS_DOWN":
                    if not self.cooldown_timer.is_cooling_down("VOLUME_DOWN", 200):
                        self.audio.volume_down(0.05)
                        self.last_action = "VOLUME_DOWN"
                        self.cooldown_timer.fire("VOLUME_DOWN")
                        
                # Fist -> Open palm: Play / Pause media
                elif hand_res.active_gesture == "FIST":
                    # Detect transition to OPEN_PALM
                    self.dwell_timer.update("fist_dwell", True)
                elif hand_res.active_gesture == "OPEN_PALM" and self.dwell_timer.get_elapsed("fist_dwell") > 200:
                    if not self.cooldown_timer.is_cooling_down("PLAY_PAUSE", 1000):
                        self.keyboard.press("space")
                        self.last_action = "PLAY_PAUSE"
                        self.cooldown_timer.fire("PLAY_PAUSE")
                        self.dwell_timer.reset("fist_dwell")
                        
                # Three fingers spread -> Pinch: minimize all windows
                elif hand_res.active_gesture == "THREE_FINGERS":
                    if not self.cooldown_timer.is_cooling_down("MINIMIZE_ALL", 2000):
                        self.window.minimize_all_windows()
                        self.last_action = "SHOW_DESKTOP"
                        self.cooldown_timer.fire("MINIMIZE_ALL")
            else:
                self.mouse.release("left")
                
            # D. EYE ACTIONS (Wink Clicking)
            if gaze_res and not hand_res:  # Only wink-click if hand is not performing gestures (safety fallback)
                if gaze_res.left_wink:
                    if not self.cooldown_timer.is_cooling_down("LEFT_CLICK", 500):
                        # Gaze position is where we click
                        self.mouse.move_to(gaze_res.gaze_screen_x, gaze_res.gaze_screen_y)
                        self.mouse.click("left")
                        self.last_action = "LEFT_WINK_CLICK"
                        self.cooldown_timer.fire("LEFT_CLICK")
                elif gaze_res.right_wink:
                    if not self.cooldown_timer.is_cooling_down("RIGHT_CLICK", 500):
                        self.mouse.move_to(gaze_res.gaze_screen_x, gaze_res.gaze_screen_y)
                        self.mouse.click("right")
                        self.last_action = "RIGHT_WINK_CLICK"
                        self.cooldown_timer.fire("RIGHT_CLICK")
                        
            # E. FACE ACTIONS (Head Tilt Scrolling & Actions)
            if face_act:
                if face_act.scroll_direction:
                    # Tilt roll scrolling
                    direction = 1 if face_act.scroll_direction == "UP" else -1
                    self.mouse.scroll(direction * face_act.scroll_speed)
                    self.last_action = f"SCROLL_{face_act.scroll_direction}"
                    
                if face_act.action_triggered:
                    act_name = face_act.action_triggered
                    self.last_action = act_name
                    if act_name == "CLOSE_WINDOW":
                        self.window.close_active_window()
                    elif act_name == "CONFIRM":
                        self.keyboard.press("enter")
                    elif act_name == "RIGHT_CLICK":
                        self.mouse.click("right")
                    elif act_name == "BROWSER_BACK":
                        self.keyboard.hotkey("alt", "left")
                    elif act_name == "BROWSER_FORWARD":
                        self.keyboard.hotkey("alt", "right")
                    elif act_name == "VOICE_ACTIVATE":
                        # Windows speech activation key combination
                        self.keyboard.hotkey("win", "h")
                    elif act_name == "SCREENSHOT":
                        # Taking a screenshot and saving to Desktop
                        self.keyboard.hotkey("win", "prtsc")
                        
            # F. MULTI-MONITOR TRANSFER WINDOW GESTURE
            # If the user gazes hard at the other monitor and does eyebrow raise (or nods head),
            # move active window there.
            if gaze_res and face_act.action_triggered == "CONFIRM":
                # nodules/confirmation while looking at the other screen
                cur_win = self.window.get_active_window()
                if cur_win:
                    win_monitor = self.window.get_window_monitor_index(cur_win)
                    gaze_monitor = 1 if gaze_res.is_looking_at_monitor else 0
                    if win_monitor != gaze_monitor:
                        self.window.move_active_window_to_other_monitor()
                        self.last_action = "MOVE_WINDOW"
                        
            # 3. UPDATE CALLBACK STATE FOR HUD / DEBUG DISPLAY
            self._update_hud_state()

    def _update_hud_state(self):
        """Sends current state info to display thread."""
        if self.state_callback:
            mode = "NORMAL"
            if self.freeze_mode:
                mode = "FREEZE"
            elif self.drawing_mode:
                mode = "DRAWING"
                
            state = {
                "mode": mode,
                "gesture": self.last_gesture,
                "action": self.last_action,
                "fps_inference": self.fps_inference,
                "fps_display": self.fps_display,
                "gaze_dot": self.gaze_dot,
                "camera_status": self.camera_status,
                "drawing_points": list(self.drawing_points),
                "drawing_color": self.current_drawing_color
            }
            self.state_callback(state)

    def __repr__(self) -> str:
        return f"ActionRouter(running={self.running}, freeze={self.freeze_mode}, drawing={self.drawing_mode})"
