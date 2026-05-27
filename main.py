import os
import sys
import argparse
import ctypes
import yaml
import queue
import time
import threading
from typing import Optional

# Set Process DPI Awareness early (critical for accurate multi-monitor and absolute coordinate controls on Windows)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

from axon.utils.logger import logger
from axon.core.camera import CameraReader
from axon.core.inference import InferenceWorker
from axon.core.action_router import ActionRouter
from axon.overlay.hud import HUDOverlay
from axon.overlay.debug_view import DebugViewer
from axon.calibration.gaze_calibrator import GazeCalibrator

def main():
    parser = argparse.ArgumentParser(description="AXON: Adaptive eXpression & Optical Navigation")
    parser.add_argument("--debug", action="store_true", help="Display local camera stream with overlaid landmarks")
    parser.add_argument("--no-phone", action="store_true", help="Disable second-angle phone RTSP camera feed")
    parser.add_argument("--calibrate", action="store_true", help="Run the 9-point eye gaze calibration routine")
    args = parser.parse_args()
    
    # Load config file
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if not os.path.exists(config_path):
        logger.critical(f"Config file not found at: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Check if models are downloaded
    face_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axon", "models", "face_landmarker.task")
    hand_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axon", "models", "hand_landmarker.task")
    if not os.path.exists(face_model) or not os.path.exists(hand_model):
        print("Required MediaPipe task model files are missing.")
        print("Running automatic downloader (setup_models.py)...")
        import setup_models
        setup_models.main()
        
    # 1. RUN GAZE CALIBRATION AND EXIT
    if args.calibrate:
        calibrator = GazeCalibrator(config_path)
        calibrator.run()
        sys.exit(0)
        
    # Override phone webcam status from CLI flag
    if args.no_phone:
        config["cameras"]["use_phone"] = False
        
    logger.info("AXON System Starting...")
    
    # Queues for inter-thread communication
    # maxsize=2 for cameras drops old frames automatically, maxsize=5 for inference results
    result_queue = queue.Queue(maxsize=5)
    
    # Initialize Camera Reader
    cam_config = config.get("cameras", {})
    phone_url = cam_config.get("phone_rtsp_url") if cam_config.get("use_phone", False) else None
    
    camera_reader = CameraReader(
        laptop_idx=cam_config.get("laptop_index", 0),
        phone_url=phone_url,
        laptop_res=tuple(cam_config.get("laptop_resolution", (1280, 720))),
        phone_res=tuple(cam_config.get("phone_resolution", (640, 480)))
    )
    
    # Initialize HUD Overlay (Tkinter must run on main thread)
    hud = HUDOverlay(config)
    
    # Initialize Action Router (which processes coordinates and fires clicks/hotkeys)
    router = ActionRouter(config, result_queue, state_callback=hud.update_state)
    
    # Initialize Inference Worker (runs MediaPipe on the frames)
    inference_worker = InferenceWorker(camera_reader, config, result_queue)
    
    # Initialize Debug window if requested
    debug_viewer = DebugViewer() if args.debug else None
    
    # Debug view polling thread (if debug option is enabled, display window using OpenCV in loop)
    def debug_polling_loop():
        logger.info("Started OpenCV Debug window thread.")
        # Create a separate copy of result queue to read from for rendering
        while router.running:
            # Check latest state from router
            state = hud.state
            # Fetch camera frames and run overlay draws
            laptop_data = camera_reader.get_laptop_frame()
            if laptop_data:
                frame, _ = laptop_data
                # Fetch dummy placeholder landmarks if queue empty, else pull from inference result
                try:
                    res = result_queue.queue[-1] if not result_queue.empty() else None
                    if res:
                        debug_viewer.show(
                            frame=res.laptop_frame,
                            face_result=res.face_result,
                            hand_result=res.hand_result,
                            active_gesture=state.get("gesture", "NONE"),
                            active_action=state.get("action", "NONE"),
                            fps=state.get("fps_inference", 0.0)
                        )
                except Exception:
                    pass
            time.sleep(0.02)
        debug_viewer.close()

    # Callback when HUD starts to kick off background worker loops
    def start_background_threads():
        logger.info("Tkinter HUD ready, booting background workers...")
        camera_reader.start()
        inference_worker.start()
        router.start()
        
        if args.debug:
            threading.Thread(target=debug_polling_loop, daemon=True).start()

    try:
        # Start Tkinter HUD mainloop (blocks until closed)
        hud.start(ready_callback=start_background_threads)
    except KeyboardInterrupt:
        logger.info("AXON System interrupted by user.")
    finally:
        logger.info("AXON System Shutting Down...")
        # Shut down threads in order
        router.stop()
        inference_worker.stop()
        camera_reader.stop()
        logger.info("AXON Shutdown Complete.")

if __name__ == "__main__":
    main()
