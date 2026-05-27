import pygetwindow as gw
from screeninfo import get_monitors
from typing import Optional, List, Tuple
from ..utils.logger import logger

class WindowManager:
    """Manages window movements, resizing, monitor transfer, and visibility on Windows."""
    def __init__(self, config: dict):
        self.config = config
        self.monitors = []
        self._detect_monitors()

    def _detect_monitors(self):
        """Loads connected monitors and sorts them horizontally by X offset."""
        try:
            self.monitors = get_monitors()
            # Sort monitors from left to right based on their x coordinates
            self.monitors.sort(key=lambda m: m.x)
            for i, m in enumerate(self.monitors):
                logger.info(f"Monitor {i}: {m.name} ({m.width}x{m.height}) at ({m.x}, {m.y}) Primary: {m.is_primary}")
        except Exception as e:
            logger.error(f"Failed to detect monitors: {e}")
            # Fallback to single monitor config values
            mc = self.config.get("monitors", {})
            class DummyMonitor:
                def __init__(self, x, y, w, h, name, primary):
                    self.x = x
                    self.y = y
                    self.width = w
                    self.height = h
                    self.name = name
                    self.is_primary = primary
            self.monitors = [
                DummyMonitor(0, 0, mc.get("laptop_width", 1920), mc.get("laptop_height", 1080), "Laptop", True),
                DummyMonitor(mc.get("laptop_width", 1920), 0, mc.get("monitor_width", 1920), mc.get("monitor_height", 1080), "Monitor", False)
            ]

    def get_active_window(self) -> Optional[gw.Win32Window]:
        """Returns the currently focused window."""
        try:
            return gw.getActiveWindow()
        except Exception as e:
            logger.debug(f"Failed to get active window: {e}")
            return None

    def get_window_monitor_index(self, window: gw.Win32Window) -> int:
        """Determines which monitor index the window center is on."""
        if not window:
            return 0
            
        win_cx = window.left + (window.width / 2)
        win_cy = window.top + (window.height / 2)
        
        for i, m in enumerate(self.monitors):
            if (m.x <= win_cx < m.x + m.width) and (m.y <= win_cy < m.y + m.height):
                return i
        return 0

    def move_active_window_to_other_monitor(self):
        """Moves the active window to the alternate monitor."""
        window = self.get_active_window()
        if not window:
            logger.warning("No active window to move.")
            return
            
        if len(self.monitors) < 2:
            logger.warning("Only one monitor detected. Cannot move window.")
            return

        current_idx = self.get_window_monitor_index(window)
        target_idx = 1 if current_idx == 0 else 0
        
        current_monitor = self.monitors[current_idx]
        target_monitor = self.monitors[target_idx]
        
        # Calculate relative coordinates inside current monitor
        rel_x = window.left - current_monitor.x
        rel_y = window.top - current_monitor.y
        
        # Scale width and height if monitors have different resolutions
        scale_w = target_monitor.width / current_monitor.width
        scale_h = target_monitor.height / current_monitor.height
        
        new_w = int(window.width * scale_w)
        new_h = int(window.height * scale_h)
        
        new_x = int(target_monitor.x + (rel_x * scale_w))
        new_y = int(target_monitor.y + (rel_y * scale_h))
        
        try:
            # If window is maximized, restore it first before moving
            was_maximized = window.isMaximized
            if was_maximized:
                window.restore()
                
            window.resizeTo(new_w, new_h)
            window.moveTo(new_x, new_y)
            
            if was_maximized:
                window.maximize()
                
            logger.info(f"Moved active window '{window.title}' from Monitor {current_idx} to Monitor {target_idx}")
        except Exception as e:
            logger.error(f"Failed to move window: {e}")

    def resize_active_window(self, scale_x: float, scale_y: float):
        """Resizes the active window based on scaling factors."""
        window = self.get_active_window()
        if not window:
            return
            
        try:
            if window.isMaximized or window.isMinimized:
                window.restore()
            new_w = int(window.width * scale_x)
            new_h = int(window.height * scale_y)
            window.resizeTo(new_w, new_h)
        except Exception as e:
            logger.error(f"Failed to resize window: {e}")

    def minimize_all_windows(self):
        """Minimizes all windows to show the desktop."""
        try:
            # We can use PyAutoGUI shortcut to show desktop (Win + D)
            import pyautogui
            pyautogui.hotkey('win', 'd', _pause=False)
            logger.info("Minimizing all windows (Show Desktop)")
        except Exception as e:
            logger.error(f"Failed to minimize all windows: {e}")

    def close_active_window(self):
        """Closes the currently active window."""
        window = self.get_active_window()
        if window:
            try:
                window.close()
                logger.info(f"Closed window: {window.title}")
            except Exception as e:
                logger.error(f"Failed to close window: {e}")

    def __repr__(self) -> str:
        return f"WindowManager(monitors={len(self.monitors)})"
