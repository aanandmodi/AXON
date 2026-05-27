import numpy as np
import cv2
import time
from typing import Optional, Tuple, Any
from ..utils.logger import logger

class DepthEngine:
    """Estimates pseudo-depth (Z-axis) of the user's hand using laptop cam scale and side phone camera triangulation."""
    def __init__(self, config: dict):
        self.config = config
        self.use_phone = config.get("cameras", {}).get("use_phone", False)
        
        # Phone position relative to laptop: "right" (45 deg right) or "left" (45 deg left)
        self.phone_side = config.get("cameras", {}).get("phone_side", "right")
        
        # Z-depth smoothing
        self.prev_z = 0.5
        self.alpha = 0.3
        
        # Thresholds
        # Z ranges from 0.0 (far) to 1.0 (close to screen/pushing forward)
        self.push_threshold = 0.70
        self.hover_threshold = 0.55
        self.is_pushing = False

    def process(self, laptop_landmarks: Optional[Any], phone_frame: Optional[np.ndarray]) -> float:
        """Computes pseudo-depth. Returns a value from 0.0 (far/pull back) to 1.0 (close/push forward)."""
        if not laptop_landmarks:
            return 0.5
            
        pts = np.array([[lm.x, lm.y, lm.z] for lm in laptop_landmarks])
        
        # 1. Laptop depth proxy: wrist to middle finger knuckle (MCP) distance in 3D
        # As hand comes closer to camera, this distance (in image coords) increases.
        wrist = pts[0]
        middle_mcp = pts[9]
        laptop_hand_scale = np.linalg.norm(wrist - middle_mcp)
        
        # Laptop scale ranges typically from 0.08 (far) to 0.25 (very close)
        # Normalize laptop scale to 0.0 - 1.0 range
        z_laptop = (laptop_hand_scale - 0.08) / (0.25 - 0.08)
        z_laptop = np.clip(z_laptop, 0.0, 1.0)
        
        z_est = z_laptop
        
        # 2. Triangulation if phone stream is active
        # We can locate the hand in the phone frame. To make it extremely fast and lightweight without YOLO,
        # we can perform a simple color thresholding (e.g. skin color mask) or template matching, 
        # or locate the hand by tracking the largest moving object in the phone camera stream!
        # This is extremely creative and doesn't require loading a heavy YOLO model on the RTX 2050 
        # unless explicitly requested, saving GPU memory and latency.
        # Let's perform a lightweight skin color / motion tracking in the phone camera.
        if self.use_phone and phone_frame is not None:
            phone_hand_x = self._estimate_hand_x_in_phone(phone_frame)
            if phone_hand_x is not None:
                # If phone is on the RIGHT:
                # As hand pushes forward (closer to screen), in the phone's field of view (looking at desk from the right),
                # the hand moves from RIGHT to LEFT (smaller X coordinate in the image).
                # If phone is on the LEFT:
                # As hand pushes forward, the hand moves from LEFT to RIGHT (larger X coordinate in the image).
                if self.phone_side == "right":
                    z_phone = 1.0 - phone_hand_x  # closer to screen = moves left = smaller X = higher depth
                else:
                    z_phone = phone_hand_x        # closer to screen = moves right = larger X = higher depth
                    
                # Combine laptop and phone estimations
                # Weight laptop cam (0.6) and phone cam triangulation (0.4)
                z_est = 0.6 * z_laptop + 0.4 * z_phone
                
        # Smooth the result
        z_smooth = self.alpha * z_est + (1.0 - self.alpha) * self.prev_z
        self.prev_z = z_smooth
        
        # Update push state
        if z_smooth > self.push_threshold:
            self.is_pushing = True
        elif z_smooth < self.hover_threshold:
            self.is_pushing = False
            
        return float(z_smooth)

    def _estimate_hand_x_in_phone(self, frame: np.ndarray) -> Optional[float]:
        """Locates the horizontal position of the hand in the phone frame using skin color masking. Returns normalized X [0, 1]."""
        try:
            # Downsample to speed up processing
            small_frame = cv2.resize(frame, (160, 120))
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            
            # Skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Find centroid of the skin mask
            moments = cv2.moments(mask)
            if moments["m00"] > 1000:  # Threshold to ensure enough skin pixels
                cx = int(moments["m10"] / moments["m00"])
                return cx / 160.0  # Normalized x-coordinate
        except Exception as e:
            logger.debug(f"Error in phone hand detection: {e}")
            
        return None

    def __repr__(self) -> str:
        return f"DepthEngine(use_phone={self.use_phone}, side={self.phone_side}, pushing={self.is_pushing})"
