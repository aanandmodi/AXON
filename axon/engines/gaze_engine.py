import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict
from ..utils.logger import logger
from ..utils.timer import DwellTimer
from ..utils.smoother import OneEuroFilter

@dataclass
class GazeResult:
    gaze_screen_x: int
    gaze_screen_y: int
    left_blink_score: float
    right_blink_score: float
    left_wink: bool
    right_wink: bool
    head_pitch: float
    head_yaw: float
    head_roll: float
    mouth_open: bool
    eyebrows_raised: bool
    is_looking_at_monitor: bool
    iris_relative_x: float
    iris_relative_y: float


class GazeEngine:
    """Processes face landmarks and blendshapes to track gaze direction, winks, eyebrows, and head pose."""
    def __init__(self, config: dict):
        self.config = config
        self.gaze_config = config.get("gaze", {})
        self.face_config = config.get("face", {})
        self.monitors_config = config.get("monitors", {})
        
        self.dwell_timer = DwellTimer()
        self.gaze_filter = None  # OneEuroFilter initialized on first point
        
        # Virtual desktop dimensions (combined monitor dimensions)
        self.laptop_w = self.monitors_config.get("laptop_width", 1920)
        self.laptop_h = self.monitors_config.get("laptop_height", 1080)
        self.monitor_w = self.monitors_config.get("monitor_width", 1920)
        self.monitor_h = self.monitors_config.get("monitor_height", 1080)
        
        # Virtual width = laptop_width + monitor_width (Since laptop is on LEFT and monitor is on RIGHT)
        self.virtual_width = self.laptop_w + self.monitor_w
        self.virtual_height = max(self.laptop_h, self.monitor_h)
        
        # 3D Standard Face Model Points for Pose Estimation
        # (Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),           # Nose tip (landmark 1)
            (0.0, -330.0, -65.0),      # Chin (landmark 152)
            (-225.0, 170.0, -135.0),   # Left eye corner (landmark 33)
            (225.0, 170.0, -135.0),    # Right eye corner (landmark 263)
            (-150.0, -150.0, -125.0),  # Left mouth corner (landmark 61)
            (150.0, -150.0, -125.0)    # Right mouth corner (landmark 291)
        ], dtype=np.float32)

    def process(self, face_result: Any, frame_width: int, frame_height: int) -> Optional[GazeResult]:
        """Processes face landmarker results and maps gaze coordinates."""
        if not face_result or not face_result.face_landmarks:
            return None
            
        landmarks = face_result.face_landmarks[0]  # Take first face detected
        blendshapes = face_result.face_blendshapes[0] if face_result.face_blendshapes else None
        
        # Extract blink/wink scores from blendshapes (0.0 to 1.0)
        # In MediaPipe: 9: eyeBlinkLeft, 10: eyeBlinkRight
        left_blink_score = 0.0
        right_blink_score = 0.0
        brow_outer_up_left = 0.0
        brow_outer_up_right = 0.0
        jaw_open = 0.0
        
        if blendshapes:
            for b in blendshapes:
                if b.category_name == "eyeBlinkLeft":
                    left_blink_score = b.score
                elif b.category_name == "eyeBlinkRight":
                    right_blink_score = b.score
                elif b.category_name == "browOuterUpLeft":
                    brow_outer_up_left = b.score
                elif b.category_name == "browOuterUpRight":
                    brow_outer_up_right = b.score
                elif b.category_name == "jawOpen":
                    jaw_open = b.score
                    
        # Left wink = left eye closed, right eye open
        wink_threshold = self.gaze_config.get("ear_wink_threshold", 0.21)  # Or adapt to blendshapes
        left_wink_active = left_blink_score > 0.65 and right_blink_score < 0.35
        right_wink_active = right_blink_score > 0.65 and left_blink_score < 0.35
        
        # Dwell winks (must be held for > 80ms and < 400ms to count as clicks)
        left_wink = self.dwell_timer.update("left_wink", left_wink_active) and self.dwell_timer.get_elapsed("left_wink") < 400.0
        right_wink = self.dwell_timer.update("right_wink", right_wink_active) and self.dwell_timer.get_elapsed("right_wink") < 400.0
        
        # Eyebrows raised
        eyebrows_raised = (brow_outer_up_left > 0.4) and (brow_outer_up_right > 0.4)
        
        # Mouth open
        mouth_open = jaw_open > 0.4
        
        # Gaze Iris tracking
        # Left Iris landmarks (indices 468, 474, 475, 476, 477 in 478-landmark model)
        # Right Iris landmarks (indices 473, 469, 470, 471, 472)
        # Eye boundaries: Left corner = 33, Right corner = 133 for left eye; 362, 263 for right eye
        left_eye_l = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_r = np.array([landmarks[133].x, landmarks[133].y])
        right_eye_l = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_r = np.array([landmarks[263].x, landmarks[263].y])
        
        # Iris centers
        left_iris_center = np.array([landmarks[468].x, landmarks[468].y])
        right_iris_center = np.array([landmarks[473].x, landmarks[473].y])
        
        # Compute horizontal relative positions
        left_relative_x = (left_iris_center[0] - left_eye_l[0]) / (left_eye_r[0] - left_eye_l[0])
        right_relative_x = (right_iris_center[0] - right_eye_l[0]) / (right_eye_r[0] - right_eye_l[0])
        
        # Average relative horizontal position
        iris_rel_x = (left_relative_x + right_relative_x) / 2.0
        
        # Vertical relative positions (top lid = 159, bottom lid = 145 for left; 386, 374 for right)
        left_eye_t = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_b = np.array([landmarks[145].x, landmarks[145].y])
        right_eye_t = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_b = np.array([landmarks[374].x, landmarks[374].y])
        
        left_relative_y = (left_iris_center[1] - left_eye_t[1]) / (left_eye_b[1] - left_eye_t[1])
        right_relative_y = (right_iris_center[1] - right_eye_t[1]) / (right_eye_b[1] - right_eye_t[1])
        
        # Average relative vertical position
        iris_rel_y = (left_relative_y + right_relative_y) / 2.0
        
        # Head Pose Estimation via SolvePnP
        # Coordinates in pixels
        image_points = np.array([
            (landmarks[1].x * frame_width, landmarks[1].y * frame_height),      # Nose tip
            (landmarks[152].x * frame_width, landmarks[152].y * frame_height),  # Chin
            (landmarks[33].x * frame_width, landmarks[33].y * frame_height),    # Left eye corner
            (landmarks[263].x * frame_width, landmarks[263].y * frame_height),  # Right eye corner
            (landmarks[61].x * frame_width, landmarks[61].y * frame_height),    # Left mouth corner
            (landmarks[291].x * frame_width, landmarks[291].y * frame_height)   # Right mouth corner
        ], dtype=np.float32)
        
        # Camera matrix approximation
        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3d, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        pitch, yaw, roll = 0.0, 0.0, 0.0
        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            # Euler angles
            sy = np.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
            singular = sy < 1e-6
            if not singular:
                pitch = np.arctan2(rmat[2,1] , rmat[2,2]) * 180.0 / np.pi
                yaw = np.arctan2(-rmat[2,0], sy) * 180.0 / np.pi
                roll = np.arctan2(rmat[1,0], rmat[0,0]) * 180.0 / np.pi
            else:
                pitch = np.arctan2(-rmat[1,2], rmat[1,1]) * 180.0 / np.pi
                yaw = np.arctan2(-rmat[2,0], sy) * 180.0 / np.pi
                roll = 0.0
                
        # Map gaze vector to screen coordinates
        # Apply head yaw and pitch compensation to the relative iris offsets
        # This keeps the gaze stable when the head turns but eyes stay fixed on target
        k_yaw = self.gaze_config.get("gaze_yaw_compensation", 0.004)
        k_pitch = self.gaze_config.get("gaze_pitch_compensation", 0.006)
        
        # Compensate: yaw turns right (positive yaw) -> iris moves left relative to eye socket -> we add compensation
        iris_rel_x_corr = iris_rel_x + k_yaw * yaw
        iris_rel_y_corr = iris_rel_y + k_pitch * pitch
        
        # Check if calibrated
        calib = self.gaze_config.get("gaze_calibration")
        if calib and "poly_x" in calib and "poly_y" in calib:
            # Polynomial degree 2 regression mapping:
            # X = c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*x*y
            cx = calib["poly_x"]
            cy = calib["poly_y"]
            x, y = iris_rel_x_corr, iris_rel_y_corr
            
            # Form the 2nd degree polynomial terms
            terms = [1.0, x, y, x**2, y**2, x*y]
            
            gaze_x = float(np.dot(cx, terms))
            gaze_y = float(np.dot(cy, terms))
        else:
            # Linear fallback
            # We map [0.35, 0.65] relative iris position to full screen bounds
            x_min, x_max = 0.35, 0.65
            y_min, y_max = 0.35, 0.65
            
            # Map horizontally across combined laptop + monitor width
            gaze_x = ((iris_rel_x_corr - x_min) / (x_max - x_min)) * self.virtual_width
            gaze_y = ((iris_rel_y_corr - y_min) / (y_max - y_min)) * self.virtual_height
            
        # Apply OneEuroFilter for low-latency, jitter-free gaze pointing
        t = time.time()
        gaze_min_cutoff = self.gaze_config.get("one_euro_min_cutoff", 0.45)
        gaze_beta = self.gaze_config.get("one_euro_beta", 0.04)
        
        if self.gaze_filter is None:
            self.gaze_filter = OneEuroFilter(t0=t, x0=np.array([gaze_x, gaze_y]), mincutoff=gaze_min_cutoff, beta=gaze_beta, dcutoff=1.0)
            smoothed_gaze = np.array([gaze_x, gaze_y])
        else:
            smoothed_gaze = self.gaze_filter(t, np.array([gaze_x, gaze_y]))
            
        # Clamp to screen bounds
        gaze_x = int(np.clip(smoothed_gaze[0], 0, self.virtual_width - 1))
        gaze_y = int(np.clip(smoothed_gaze[1], 0, self.virtual_height - 1))
        
        # Check if looking at monitor (laptop is on left, monitor is on right)
        is_looking_at_monitor = gaze_x >= self.laptop_w
        
        return GazeResult(
            gaze_screen_x=gaze_x,
            gaze_screen_y=gaze_y,
            left_blink_score=left_blink_score,
            right_blink_score=right_blink_score,
            left_wink=left_wink,
            right_wink=right_wink,
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            mouth_open=mouth_open,
            eyebrows_raised=eyebrows_raised,
            is_looking_at_monitor=is_looking_at_monitor,
            iris_relative_x=iris_rel_x_corr,
            iris_relative_y=iris_rel_y_corr
        )

    def __repr__(self) -> str:
        calibrated = self.gaze_config.get("gaze_calibration") is not None
        return f"GazeEngine(calibrated={calibrated}, virtual_w={self.virtual_width}, virtual_h={self.virtual_height})"
