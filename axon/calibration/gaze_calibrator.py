import tkinter as tk
import time
import queue
import threading
import numpy as np
import yaml
import os
from typing import Optional, List, Tuple
from ..utils.logger import logger
from ..core.camera import CameraReader
from ..core.inference import InferenceWorker
from ..engines.gaze_engine import GazeEngine

class GazeCalibrator:
    """Performs a 9-point gaze calibration routine by fitting a 2nd-degree polynomial regression."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.monitors_config = self.config.get("monitors", {})
        self.laptop_w = self.monitors_config.get("laptop_width", 1920)
        self.laptop_h = self.monitors_config.get("laptop_height", 1080)
        self.monitor_w = self.monitors_config.get("monitor_width", 1920)
        self.monitor_h = self.monitors_config.get("monitor_height", 1080)
        
        # Combined virtual desktop (Laptop LEFT, Monitor RIGHT)
        self.virtual_width = self.laptop_w + self.monitor_w
        self.virtual_height = max(self.laptop_h, self.monitor_h)
        
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.current_point_idx = 0
        self.calibration_active = False
        
        # Gaze coordinates targets (3x3 grid in combined desktop)
        # Margin of 10% from borders
        xs = [int(0.1 * self.virtual_width), int(0.5 * self.virtual_width), int(0.9 * self.virtual_width)]
        ys = [int(0.1 * self.virtual_height), int(0.5 * self.virtual_height), int(0.9 * self.virtual_height)]
        
        self.target_points = []
        for y in ys:
            for x in xs:
                self.target_points.append((x, y))
                
        # Collected samples: list of (iris_x, iris_y, target_x, target_y)
        self.samples: List[Tuple[float, float, int, int]] = []
        self.camera_reader: Optional[CameraReader] = None
        self.inference_worker: Optional[InferenceWorker] = None
        self.result_queue = queue.Queue(maxsize=5)
        self.gaze_engine = GazeEngine(self.config)
        self.collecting = False

    def run(self):
        """Launches the calibration GUI and starts camera/inference loops."""
        logger.info("Starting Gaze Calibration Routine...")
        
        # Start cameras and inference thread for calibration tracking
        self.camera_reader = CameraReader(laptop_idx=0)
        self.camera_reader.start()
        
        self.inference_worker = InferenceWorker(self.camera_reader, self.config, self.result_queue)
        self.inference_worker.start()
        
        # Start helper thread to collect iris samples in background
        threading.Thread(target=self._sample_collector, daemon=True).start()
        
        # Start GUI
        self.root = tk.Tk()
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.virtual_width}x{self.virtual_height}+0+0")
        
        self.canvas = tk.Canvas(self.root, bg="#222222", bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.create_text(
            self.virtual_width // 2, self.virtual_height // 2 - 50,
            text="AXON EYE GAZE CALIBRATION", fill="white", font=("Arial", 28, "bold")
        )
        self.canvas.create_text(
            self.virtual_width // 2, self.virtual_height // 2 + 20,
            text="Look at the red dots and keep your head steady.\nPress SPACEBAR to start.",
            fill="yellow", font=("Arial", 16), justify="center"
        )
        
        self.root.bind("<space>", lambda e: self._start_calibration())
        self.root.bind("<Escape>", lambda e: self._exit_calibration())
        
        self.root.mainloop()

    def _start_calibration(self):
        self.canvas.delete("all")
        self.current_point_idx = 0
        self.samples.clear()
        self.calibration_active = True
        self._show_next_point()

    def _show_next_point(self):
        self.canvas.delete("all")
        if self.current_point_idx >= len(self.target_points):
            self._finish_calibration()
            return
            
        tx, ty = self.target_points[self.current_point_idx]
        
        # Draw target dot
        self.canvas.create_oval(tx - 15, ty - 15, tx + 15, ty + 15, fill="red", outline="white", width=3)
        self.canvas.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, fill="white")
        
        # Dwell instructions
        self.canvas.create_text(
            self.virtual_width // 2, 50,
            text=f"Focus on the Dot ({self.current_point_idx + 1}/9)",
            fill="white", font=("Arial", 20, "bold")
        )
        
        # Start collecting samples after 1.0s dwell
        self.root.after(1000, self._start_sampling)

    def _start_sampling(self):
        self.collecting = True
        # Collect for 1.2 seconds (approx 35 frames)
        self.root.after(1200, self._stop_sampling)

    def _stop_sampling(self):
        self.collecting = False
        self.current_point_idx += 1
        self._show_next_point()

    def _sample_collector(self):
        """Thread that pops results and aggregates iris offsets if collecting is active."""
        while True:
            try:
                res = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            if not self.calibration_active:
                continue
                
            if self.collecting:
                h, w, _ = res.laptop_frame.shape
                # Process landmarks via GazeEngine
                gaze_res = self.gaze_engine.process(res.face_result, w, h)
                if gaze_res:
                    tx, ty = self.target_points[self.current_point_idx]
                    self.samples.append((
                        gaze_res.iris_relative_x,
                        gaze_res.iris_relative_y,
                        tx,
                        ty
                    ))

    def _finish_calibration(self):
        """Solves the second-degree polynomial mapping and updates config.yaml."""
        self.calibration_active = False
        self.collecting = False
        
        if len(self.samples) < 20:
            logger.error("Not enough calibration samples collected.")
            self.canvas.create_text(
                self.virtual_width // 2, self.virtual_height // 2,
                text="Calibration Failed! Not enough landmarks detected.\nPress SPACEBAR to try again or ESC to exit.",
                fill="red", font=("Arial", 20), justify="center"
            )
            return
            
        logger.info(f"Fitting polynomial model on {len(self.samples)} samples...")
        
        # Format design matrix A for polynomial degree 2:
        # A[i] = [1, x, y, x^2, y^2, x*y]
        samples_arr = np.array(self.samples)
        x_iris = samples_arr[:, 0]
        y_iris = samples_arr[:, 1]
        target_x = samples_arr[:, 2]
        target_y = samples_arr[:, 3]
        
        A = np.vstack([
            np.ones_like(x_iris),
            x_iris,
            y_iris,
            x_iris**2,
            y_iris**2,
            x_iris * y_iris
        ]).T
        
        # Solve least squares for X and Y mapping
        cx, _, _, _ = np.linalg.lstsq(A, target_x, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, target_y, rcond=None)
        
        # Save to config dict
        poly_coefficients = {
            "poly_x": [float(c) for c in cx],
            "poly_y": [float(c) for c in cy]
        }
        
        self.config["gaze"]["gaze_calibration"] = poly_coefficients
        
        # Save back to config.yaml file
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f)
            
        logger.info("Successfully saved gaze calibration coefficients to config.yaml!")
        
        self.canvas.create_text(
            self.virtual_width // 2, self.virtual_height // 2 - 50,
            text="CALIBRATION SUCCESSFUL!", fill="green", font=("Arial", 26, "bold")
        )
        self.canvas.create_text(
            self.virtual_width // 2, self.virtual_height // 2 + 20,
            text="Saved coefficients to config.yaml.\nPress ESCAPE to exit and start AXON.",
            fill="white", font=("Arial", 16), justify="center"
        )
        
        # Clean up threads
        self.inference_worker.stop()
        self.camera_reader.stop()

    def _exit_calibration(self):
        if self.inference_worker:
            self.inference_worker.stop()
        if self.camera_reader:
            self.camera_reader.stop()
        if self.root:
            self.root.destroy()


if __name__ == "__main__":
    # Test runner
    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")
    calib = GazeCalibrator(config_file)
    calib.run()
