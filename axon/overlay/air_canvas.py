import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..utils.logger import logger

class AirCanvas:
    """Manages the status, colors, coordinates, and rendering of the transparent whiteboard drawing layer."""
    def __init__(self):
        self.points: List[Tuple[int, int]] = []
        self.color = "red"
        self.thickness = 5
        self.colors_cycle = ["red", "blue", "green", "yellow", "purple"]
        self.color_idx = 0

    def add_point(self, x: int, y: int):
        """Adds a coordinate point to the active drawing trail."""
        self.points.append((x, y))

    def clear(self):
        """Clears all drawn paths."""
        self.points.clear()
        logger.info("Whiteboard canvas cleared.")

    def cycle_color(self) -> str:
        """Cycles to the next color in the palette."""
        self.color_idx = (self.color_idx + 1) % len(self.colors_cycle)
        self.color = self.colors_cycle[self.color_idx]
        logger.info(f"Whiteboard color changed to: {self.color}")
        return self.color

    def draw_on_tkinter_canvas(self, tk_canvas):
        """Draws the whiteboard trail onto a Tkinter Canvas widget."""
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                p1 = self.points[i]
                p2 = self.points[i+1]
                tk_canvas.create_line(
                    p1[0], p1[1], p2[0], p2[1],
                    fill=self.color, width=self.thickness, capstyle="round"
                )

    def save_to_png(self, filepath: str, width: int, height: int):
        """Exports the whiteboard drawing to a transparent PNG file."""
        if not self.points:
            return
            
        try:
            # Create a transparent RGBA image
            img = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Map color names to BGR
            color_map = {
                "red": (0, 0, 255, 255),
                "blue": (255, 0, 0, 255),
                "green": (0, 255, 0, 255),
                "yellow": (0, 255, 255, 255),
                "purple": (255, 0, 255, 255)
            }
            c = color_map.get(self.color, (0, 0, 255, 255))
            
            for i in range(len(self.points) - 1):
                p1 = self.points[i]
                p2 = self.points[i+1]
                cv2.line(img, p1, p2, c, self.thickness)
                
            cv2.imwrite(filepath, img)
            logger.info(f"Saved whiteboard drawing to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save whiteboard image: {e}")

    def __repr__(self) -> str:
        return f"AirCanvas(points={len(self.points)}, color={self.color})"
