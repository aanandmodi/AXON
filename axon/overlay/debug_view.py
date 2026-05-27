import cv2
import numpy as np
from typing import Optional, Any
from ..utils.logger import logger

class DebugViewer:
    """Manages the OpenCV debug window, drawing facial and hand landmarks onto the captured camera feeds."""
    def __init__(self, window_name: str = "AXON Debug Feed"):
        self.window_name = window_name
        self.active = False

    def show(self, frame: np.ndarray, face_result: Any, hand_result: Any, 
             active_gesture: str, active_action: str, fps: float):
        """Annotates the frame with landmarks and text, and displays it in a window."""
        if frame is None:
            return
            
        self.active = True
        annotated_frame = frame.copy()
        h, w, _ = annotated_frame.shape
        
        # 1. DRAW FACE LANDMARKS (subset for clean view)
        # We can draw the outline, eyes, eyebrows, and iris centers
        if face_result and face_result.face_landmarks:
            landmarks = face_result.face_landmarks[0]
            
            # Draw eyes & iris centers
            # Left eye corners: 33, 133. Left Iris center: 468
            # Right eye corners: 362, 263. Right Iris center: 473
            for idx in [33, 133, 362, 263]:
                if idx < len(landmarks):
                    cx = int(landmarks[idx].x * w)
                    cy = int(landmarks[idx].y * h)
                    cv2.circle(annotated_frame, (cx, cy), 2, (0, 255, 255), -1)
                    
            # Draw Iris centers in green
            for idx in [468, 473]:
                if idx < len(landmarks):
                    cx = int(landmarks[idx].x * w)
                    cy = int(landmarks[idx].y * h)
                    cv2.circle(annotated_frame, (cx, cy), 3, (0, 255, 0), -1)
                    
            # Draw chin and nose tip for head pose reference
            for idx in [1, 152]:
                if idx < len(landmarks):
                    cx = int(landmarks[idx].x * w)
                    cy = int(landmarks[idx].y * h)
                    cv2.circle(annotated_frame, (cx, cy), 3, (255, 0, 0), -1)
                    
        # 2. DRAW HAND LANDMARKS AND BONES
        if hand_result and hand_result.hand_landmarks:
            for landmarks in hand_result.hand_landmarks:
                # Draw connections (bones)
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
                    (9, 10), (10, 11), (11, 12),           # Middle
                    (13, 14), (14, 15), (15, 16),          # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
                    (5, 9), (9, 13), (13, 17)              # Palm base
                ]
                
                # Convert points to pixel coordinates
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                
                # Draw lines
                for p1, p2 in connections:
                    if p1 < len(pts) and p2 < len(pts):
                        cv2.line(annotated_frame, pts[p1], pts[p2], (0, 255, 0), 1)
                        
                # Draw joints as points
                for pt in pts:
                    cv2.circle(annotated_frame, pt, 3, (0, 0, 255), -1)
                    
                # Highlight Index Tip in yellow
                cv2.circle(annotated_frame, pts[8], 6, (0, 255, 255), -1)
                
        # 3. TEXT OVERLAYS
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Gesture: {active_gesture}", (10, 60), font, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Action: {active_action}", (10, 90), font, 0.7, (0, 0, 255), 2)
        
        # Display window
        cv2.imshow(self.window_name, annotated_frame)
        cv2.waitKey(1)  # Required to refresh the frame

    def close(self):
        """Destroys the OpenCV debug window."""
        if self.active:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self.active = False
