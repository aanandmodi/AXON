import tkinter as tk
import time
from typing import Dict, Any, Optional
from screeninfo import get_monitors
from ..utils.logger import logger

class HUDOverlay:
    """A transparent, topmost, click-through Tkinter overlay showing system status, gaze pointer, and air whiteboard."""
    def __init__(self, config: dict):
        self.config = config
        self.monitors_config = config.get("monitors", {})
        
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        
        # Calculate full virtual desktop dimensions
        self.laptop_w = self.monitors_config.get("laptop_width", 1920)
        self.laptop_h = self.monitors_config.get("laptop_height", 1080)
        self.monitor_w = self.monitors_config.get("monitor_width", 1920)
        self.monitor_h = self.monitors_config.get("monitor_height", 1080)
        
        # Laptop is on left (0 to laptop_w), Monitor on right (laptop_w to laptop_w + monitor_w)
        self.virtual_width = self.laptop_w + self.monitor_w
        self.virtual_height = max(self.laptop_h, self.monitor_h)
        
        self.state: Dict[str, Any] = {
            "mode": "NORMAL",
            "gesture": "NONE",
            "action": "NONE",
            "fps_inference": 0.0,
            "fps_display": 0.0,
            "gaze_dot": (0, 0),
            "camera_status": {"laptop": True, "phone": False},
            "drawing_points": [],
            "drawing_color": "red"
        }
        
        self.last_display_time = time.perf_counter()
        self.fps_display = 0.0
        self.gesture_fade_time = 0.0
        self.action_fade_time = 0.0

    def start(self, ready_callback=None):
        """Initializes and runs the Tkinter window mainloop."""
        self.root = tk.Tk()
        
        # Windows-specific transparency and style properties
        # Make the background color 'white' transparent and click-through
        self.root.configure(bg='white')
        self.root.wm_attributes('-transparentcolor', 'white')
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        
        # Span across all displays
        geom = f"{self.virtual_width}x{self.virtual_height}+0+0"
        self.root.geometry(geom)
        logger.info(f"HUD Overlay geometry set to: {geom}")
        
        # Set click-through via Windows API for the entire window
        self._set_click_through()
        
        # Create full screen Canvas
        # Background is white (so it will be transparent)
        # bd=0, highlightthickness=0 removes borders
        self.canvas = tk.Canvas(self.root, width=self.virtual_width, height=self.virtual_height, 
                                bg='white', bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        if ready_callback:
            ready_callback()
            
        # Run drawing update loop
        self.update_hud()
        
        # Run Tkinter mainloop
        self.root.mainloop()

    def _set_click_through(self):
        """Uses Windows Win32 API to set extended window style WS_EX_TRANSPARENT and WS_EX_LAYERED for click-through."""
        try:
            import win32gui
            import win32con
            # Get window handle
            hwnd = win32gui.GetParent(self.root.winfo_id())
            # Get current styles
            styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            # Add transparent (click-through) style
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED)
            logger.info("HUD click-through styles applied via Win32 API.")
        except Exception as e:
            logger.error(f"Failed to set window click-through styles: {e}")

    def update_state(self, new_state: dict):
        """Receives state dictionary from ActionRouter thread."""
        old_gesture = self.state.get("gesture")
        old_action = self.state.get("action")
        
        self.state.update(new_state)
        
        now = time.perf_counter()
        if self.state["gesture"] != old_gesture and self.state["gesture"] != "NONE":
            self.gesture_fade_time = now + 1.0  # Show gesture for 1 second
        if self.state["action"] != old_action and self.state["action"] != "NONE":
            self.action_fade_time = now + 1.2  # Show action for 1.2 seconds

    def update_hud(self):
        """Redraws canvas elements at ~60fps."""
        if not self.root or not self.canvas:
            return
            
        now = time.perf_counter()
        dt = now - self.last_display_time
        self.fps_display = 1.0 / max(dt, 0.001)
        self.last_display_time = now
        
        # Clear canvas (re-fill with transparent white)
        self.canvas.delete("all")
        
        # 1. DRAW AIR CANVAS WHITEBOARD PATHS
        # (Draws line trails if whiteboard mode is active and we have coordinates)
        pts = self.state.get("drawing_points", [])
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i+1]
                # Draw lines
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=self.state.get("drawing_color", "red"), width=5, capstyle="round")
                
        # 2. DRAW GAZE DOT
        gaze_x, gaze_y = self.state.get("gaze_dot", (0, 0))
        # Only draw if not in FREEZE mode
        if self.state.get("mode") != "FREEZE":
            r = 10
            # Draw outer ring
            self.canvas.create_oval(gaze_x - r, gaze_y - r, gaze_x + r, gaze_y + r, outline="red", width=2)
            # Draw inner solid dot (use off-white for center if we want contrast, but standard color is fine)
            self.canvas.create_oval(gaze_x - 3, gaze_y - 3, gaze_x + 3, gaze_y + 3, fill="red")
            
        # 3. DRAW INFOBAR HUD BADGES
        # Let's draw HUD badge in top-right of both screens so it is always visible!
        # Top-right of laptop screen: (laptop_w - 200, 20)
        # Top-right of external monitor screen: (laptop_w + monitor_w - 200, 20)
        badge_positions = [
            (self.laptop_w - 280, 20),
            (self.virtual_width - 280, 20)
        ]
        
        for bx, by in badge_positions:
            # Mode Badge
            mode = self.state.get("mode", "NORMAL")
            bg_color = "#333333"
            if mode == "FREEZE":
                bg_color = "#f44336"  # Red
            elif mode == "DRAWING":
                bg_color = "#4caf50"  # Green
                
            self.canvas.create_rectangle(bx, by, bx + 120, by + 35, fill=bg_color, outline="#777777", width=1)
            self.canvas.create_text(bx + 60, by + 18, text=mode, fill="yellow", font=("Arial", 12, "bold"))
            
            # FPS Stats
            fps_inf = self.state.get("fps_inference", 0.0)
            fps_text = f"INF: {fps_inf:.1f} | HUD: {self.fps_display:.1f}"
            self.canvas.create_text(bx + 180, by + 18, text=fps_text, fill="black", font=("Courier", 10, "bold"))
            
            # Draw Camera Status indicators
            cam_stat = self.state.get("camera_status", {"laptop": True, "phone": False})
            laptop_color = "#4caf50" if cam_stat.get("laptop") else "#f44336"
            phone_color = "#4caf50" if cam_stat.get("phone") else "#f44336"
            
            # Laptop cam label & indicator
            self.canvas.create_oval(bx, by + 50, bx + 10, by + 60, fill=laptop_color, outline="")
            self.canvas.create_text(bx + 50, by + 55, text="L-CAM", fill="black", font=("Arial", 8, "bold"))
            
            # Phone cam label & indicator
            self.canvas.create_oval(bx + 90, by + 50, bx + 100, by + 60, fill=phone_color, outline="")
            self.canvas.create_text(bx + 140, by + 55, text="P-CAM", fill="black", font=("Arial", 8, "bold"))
            
            # Gesture name (with fade)
            if now < self.gesture_fade_time:
                gest_text = f"Gesture: {self.state['gesture']}"
                self.canvas.create_text(bx + 60, by + 80, text=gest_text, fill="#2196f3", font=("Arial", 11, "bold"), anchor="w")
                
            # Action fired (with fade)
            if now < self.action_fade_time:
                act_text = f"Action: {self.state['action']}"
                self.canvas.create_text(bx + 60, by + 105, text=act_text, fill="#e91e63", font=("Arial", 11, "bold"), anchor="w")

        # Schedule next update in ~16ms (60 FPS)
        self.root.after(16, self.update_hud)

    def stop(self):
        """Closes the Tkinter window."""
        if self.root:
            self.root.quit()
            self.root.destroy()
            self.root = None


if __name__ == "__main__":
    print("Testing HUDOverlay...")
    config = {
        "monitors": {
            "laptop_width": 1920,
            "laptop_height": 1080,
            "monitor_width": 1920,
            "monitor_height": 1080
        }
    }
    hud = HUDOverlay(config)
    
    def test_updates():
        import random
        # Simulate moving gaze
        time.sleep(1)
        for _ in range(100):
            hud.update_state({
                "mode": "NORMAL",
                "gaze_dot": (random.randint(200, 1600), random.randint(200, 800)),
                "fps_inference": 45.3,
                "gesture": "AIR_MOUSE" if random.random() > 0.5 else "PINCH",
                "action": "LEFT_CLICK" if random.random() > 0.8 else "NONE",
                "camera_status": {"laptop": True, "phone": False}
            })
            time.sleep(0.05)
            
    threading.Thread(target=test_updates, daemon=True).start()
    hud.start()
