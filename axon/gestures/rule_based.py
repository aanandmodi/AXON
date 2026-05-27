import numpy as np
from typing import List, Tuple, Optional

# Landmark index definitions
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_PIP = 10
MIDDLE_MCP = 9
RING_TIP = 16
RING_PIP = 14
RING_MCP = 13
PINKY_TIP = 20
PINKY_PIP = 18
PINKY_MCP = 17

def get_finger_states(pts: np.ndarray, hand_type: str) -> Tuple[bool, bool, bool, bool, bool]:
    """Returns whether each finger is extended: (thumb, index, middle, ring, pinky).
    Uses rotation-invariant distance ratios relative to the wrist to ensure reliability at all hand angles.
    """
    hand_scale = np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP])
    if hand_scale == 0:
        hand_scale = 1.0
        
    # Calculate Euclidean distances from Wrist (0) to Tips and PIPs
    # A finger is extended if its tip is significantly farther from the wrist than its PIP joint
    index_extended = np.linalg.norm(pts[INDEX_TIP] - pts[WRIST]) > np.linalg.norm(pts[INDEX_PIP] - pts[WRIST]) * 1.12
    middle_extended = np.linalg.norm(pts[MIDDLE_TIP] - pts[WRIST]) > np.linalg.norm(pts[MIDDLE_PIP] - pts[WRIST]) * 1.12
    ring_extended = np.linalg.norm(pts[RING_TIP] - pts[WRIST]) > np.linalg.norm(pts[RING_PIP] - pts[WRIST]) * 1.12
    pinky_extended = np.linalg.norm(pts[PINKY_TIP] - pts[WRIST]) > np.linalg.norm(pts[PINKY_PIP] - pts[WRIST]) * 1.12
    
    # Thumb: Check distance from thumb tip to index MCP and middle MCP
    # If it is far away, it is extended
    thumb_extended = np.linalg.norm(pts[THUMB_TIP] - pts[MIDDLE_MCP]) > hand_scale * 0.75
        
    return thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended


def classify_gesture(pts: np.ndarray, hand_type: str, pinch_threshold: float = 0.05) -> Tuple[str, float, bool]:
    """Classifies a gesture based on landmark points. Returns (gesture_name, pinch_distance, is_pinching)."""
    thumb, index, middle, ring, pinky = get_finger_states(pts, hand_type)
    
    # Scale normalization factor (wrist to middle MCP)
    hand_scale = np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP])
    if hand_scale == 0:
        hand_scale = 1.0
        
    # Calculate normalized pinch distances
    index_pinch_dist = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / hand_scale
    middle_pinch_dist = np.linalg.norm(pts[THUMB_TIP] - pts[MIDDLE_TIP]) / hand_scale
    
    is_index_pinching = index_pinch_dist < pinch_threshold
    is_middle_pinching = middle_pinch_dist < pinch_threshold
    
    # 1. Pinch Gestures (Must have ring and pinky folded to avoid false positives)
    if is_index_pinching and not ring and not pinky:
        return "INDEX_PINCH", index_pinch_dist, True
        
    if is_middle_pinching and not ring and not pinky:
        return "MIDDLE_PINCH", middle_pinch_dist, True
        
    # 2. Air Mouse Pointer Mode: Index extended, middle/ring/pinky folded (thumb can be open or closed)
    if index and not middle and not ring and not pinky:
        return "AIR_MOUSE", index_pinch_dist, False
        
    # 3. Scroll Mode: Both Index and Middle extended, ring/pinky folded
    if index and middle and not ring and not pinky:
        return "SCROLL", index_pinch_dist, False
        
    # 4. Standard Hand Postures
    if not index and not middle and not ring and not pinky and not thumb:
        return "FIST", index_pinch_dist, False
        
    if index and middle and ring and pinky:
        return "OPEN_PALM", index_pinch_dist, False
        
    if index and pinky and not middle and not ring:
        return "ROCK", index_pinch_dist, False
        
    if index and middle and ring and not pinky:
        return "THREE_FINGERS", index_pinch_dist, False
        
    if thumb and not index and not middle and not ring and not pinky:
        # Check simple relative vertical displacement of thumb vs wrist
        if pts[THUMB_TIP][1] < pts[WRIST][1] - 0.08:
            return "THUMBS_UP", index_pinch_dist, False
        elif pts[THUMB_TIP][1] > pts[WRIST][1] + 0.08:
            return "THUMBS_DOWN", index_pinch_dist, False
            
    return "UNKNOWN", index_pinch_dist, False
