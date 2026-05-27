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
    """Returns whether each finger is extended: (thumb, index, middle, ring, pinky)."""
    # Check simple vertical extensions (y is inverted in MediaPipe coords: smaller y is higher up)
    index_extended = pts[INDEX_TIP][1] < pts[INDEX_PIP][1]
    middle_extended = pts[MIDDLE_TIP][1] < pts[MIDDLE_PIP][1]
    ring_extended = pts[RING_TIP][1] < pts[RING_PIP][1]
    pinky_extended = pts[PINKY_TIP][1] < pts[PINKY_PIP][1]
    
    # Thumb: Check x displacement relative to index MCP
    if hand_type == "Right":
        thumb_extended = pts[THUMB_TIP][0] < pts[INDEX_MCP][0] - 0.02
    else:
        thumb_extended = pts[THUMB_TIP][0] > pts[INDEX_MCP][0] + 0.02
        
    return thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended


def classify_gesture(pts: np.ndarray, hand_type: str, pinch_threshold: float = 0.05) -> Tuple[str, float, bool]:
    """Classifies a gesture based on landmark points. Returns (gesture_name, pinch_distance, is_pinching)."""
    thumb, index, middle, ring, pinky = get_finger_states(pts, hand_type)
    
    # Scale normalization
    hand_scale = np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP])
    if hand_scale == 0:
        hand_scale = 1.0
        
    # Thumb-to-Index Tip distance
    pinch_dist = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / hand_scale
    is_pinching = pinch_dist < pinch_threshold
    
    # Classification rules
    if not index and not middle and not ring and not pinky and not thumb:
        return "FIST", pinch_dist, is_pinching
    elif index and middle and ring and pinky:
        return "OPEN_PALM", pinch_dist, is_pinching
    elif index and not middle and not ring and not pinky:
        return "AIR_MOUSE", pinch_dist, is_pinching
    elif is_pinching and not middle and not ring and not pinky:
        return "PINCH", pinch_dist, is_pinching
    elif index and middle and not ring and not pinky:
        return "PEACE", pinch_dist, is_pinching
    elif index and pinky and not middle and not ring:
        return "ROCK", pinch_dist, is_pinching
    elif index and middle and ring and not pinky:
        return "THREE_FINGERS", pinch_dist, is_pinching
    elif thumb and not index and not middle and not ring and not pinky:
        if pts[THUMB_TIP][1] < pts[WRIST][1] - 0.1:
            return "THUMBS_UP", pinch_dist, is_pinching
        elif pts[THUMB_TIP][1] > pts[WRIST][1] + 0.1:
            return "THUMBS_DOWN", pinch_dist, is_pinching
            
    return "UNKNOWN", pinch_dist, is_pinching
