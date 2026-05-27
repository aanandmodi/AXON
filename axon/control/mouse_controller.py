import pyautogui
from pynput.mouse import Controller, Button
from typing import Tuple
from ..utils.logger import logger

class MouseController:
    """Controls OS mouse cursor movements, clicks, and scrolling."""
    def __init__(self):
        # Configure PyAutoGUI for minimum latency
        pyautogui.PAUSE = 0.0
        pyautogui.FAILSAFE = True  # Move cursor to corner to abort
        
        self.pynput_mouse = Controller()
        self.screen_w, self.screen_h = pyautogui.size()

    def move_to(self, x: int, y: int):
        """Moves mouse cursor to absolute pixel coordinates (x, y)."""
        try:
            # Using pynput is faster and bypasses pyautogui's safety pauses completely
            self.pynput_mouse.position = (x, y)
        except Exception as e:
            logger.error(f"Error in mouse move_to: {e}")
            # Fallback to pyautogui
            pyautogui.moveTo(x, y, _pause=False)

    def move_relative(self, dx: int, dy: int):
        """Moves mouse cursor relative to current position."""
        try:
            self.pynput_mouse.move(dx, dy)
        except Exception as e:
            logger.error(f"Error in mouse move_relative: {e}")
            pyautogui.moveRel(dx, dy, _pause=False)

    def click(self, button: str = "left"):
        """Clicks the specified mouse button ('left', 'right', 'middle')."""
        try:
            if button == "left":
                self.pynput_mouse.click(Button.left)
            elif button == "right":
                self.pynput_mouse.click(Button.right)
            elif button == "middle":
                self.pynput_mouse.click(Button.middle)
            logger.info(f"Mouse click: {button}")
        except Exception as e:
            logger.error(f"Error in mouse click: {e}")
            pyautogui.click(button=button)

    def press(self, button: str = "left"):
        """Holds down the specified mouse button."""
        try:
            if button == "left":
                self.pynput_mouse.press(Button.left)
            elif button == "right":
                self.pynput_mouse.press(Button.right)
        except Exception as e:
            logger.error(f"Error in mouse press: {e}")
            pyautogui.mouseDown(button=button)

    def release(self, button: str = "left"):
        """Releases the specified mouse button."""
        try:
            if button == "left":
                self.pynput_mouse.release(Button.left)
            elif button == "right":
                self.pynput_mouse.release(Button.right)
        except Exception as e:
            logger.error(f"Error in mouse release: {e}")
            pyautogui.mouseUp(button=button)

    def scroll(self, amount: int):
        """Scrolls vertically. Positive for up, negative for down."""
        try:
            # pynput scroll uses (dx, dy)
            self.pynput_mouse.scroll(0, amount)
        except Exception as e:
            logger.error(f"Error in mouse scroll: {e}")
            pyautogui.scroll(amount * 50)  # pyautogui scroll units differ

    def get_position(self) -> Tuple[int, int]:
        """Returns current mouse position (x, y)."""
        return self.pynput_mouse.position

    def __repr__(self) -> str:
        return f"MouseController(screen_res={self.screen_w}x{self.screen_h})"


if __name__ == "__main__":
    print("Testing MouseController...")
    mouse = MouseController()
    pos = mouse.get_position()
    print("Current position:", pos)
    mouse.move_to(pos[0] + 50, pos[1] + 50)
    print("Moved position:", mouse.get_position())
