import pyautogui
from typing import Sequence
from ..utils.logger import logger

class KeyboardController:
    """Controls OS keyboard simulation using PyAutoGUI."""
    def __init__(self):
        pyautogui.PAUSE = 0.0

    def press(self, key: str):
        """Simulates pressing and releasing a key (e.g. 'enter', 'space', 'a')."""
        try:
            pyautogui.press(key, _pause=False)
            logger.info(f"Keyboard press: {key}")
        except Exception as e:
            logger.error(f"Error in key press: {e}")

    def hotkey(self, *keys: str):
        """Simulates pressing a combination of keys (e.g. 'ctrl', 'alt', 'del')."""
        try:
            pyautogui.hotkey(*keys, _pause=False)
            logger.info(f"Keyboard hotkey: {keys}")
        except Exception as e:
            logger.error(f"Error in hotkey: {e}")

    def key_down(self, key: str):
        """Holds down a key."""
        try:
            pyautogui.keyDown(key, _pause=False)
        except Exception as e:
            logger.error(f"Error in key_down: {e}")

    def key_up(self, key: str):
        """Releases a key."""
        try:
            pyautogui.keyUp(key, _pause=False)
        except Exception as e:
            logger.error(f"Error in key_up: {e}")

    def type_string(self, text: str):
        """Types out a string of characters."""
        try:
            pyautogui.write(text, interval=0.01)
            logger.info(f"Keyboard type text length: {len(text)}")
        except Exception as e:
            logger.error(f"Error typing string: {e}")

    def __repr__(self) -> str:
        return "KeyboardController()"
