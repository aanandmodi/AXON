import os
import time
import queue
import threading
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, NamedTuple, Any
from ..utils.logger import logger

class InferenceResult(NamedTuple):
    laptop_frame: np.ndarray
    timestamp: float
    face_result: Any  # FaceLandmarkerResult
    hand_result: Any  # HandLandmarkerResult
    phone_frame: Optional[np.ndarray] = None


class InferenceWorker:
    """Runs MediaPipe FaceLandmarker and HandLandmarker models on a background thread."""
    def __init__(self, camera_reader: Any, config: dict, result_queue: queue.Queue):
        self.camera_reader = camera_reader
        self.config = config
        self.result_queue = result_queue
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Load model paths
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.face_model_path = os.path.join(current_dir, "models", "face_landmarker.task")
        self.hand_model_path = os.path.join(current_dir, "models", "hand_landmarker.task")
        
        self.use_gpu = config.get("advanced", {}).get("use_gpu", True)

    def start(self):
        """Starts the inference thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_inference, daemon=True, name="InferenceWorker")
        self.thread.start()

    def stop(self):
        """Stops the inference thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _run_inference(self):
        logger.info("Initializing MediaPipe models...")
        
        # Configure Delegate
        delegate = python.BaseOptions.Delegate.CPU
        if self.use_gpu:
            logger.info("Attempting to use GPU Delegate for MediaPipe...")
            delegate = python.BaseOptions.Delegate.GPU
            
        # Face Landmarker options
        face_base_options = python.BaseOptions(
            model_asset_path=self.face_model_path,
            delegate=delegate
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        
        # Hand Landmarker options
        hand_base_options = python.BaseOptions(
            model_asset_path=self.hand_model_path,
            delegate=delegate
        )
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        
        try:
            face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
            hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            logger.info("MediaPipe landmarker models successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe models with GPU delegate: {e}. Falling back to CPU...")
            # Fallback to CPU delegate
            face_base_options.delegate = python.BaseOptions.Delegate.CPU
            hand_base_options.delegate = python.BaseOptions.Delegate.CPU
            try:
                face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
                hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
                logger.info("MediaPipe models loaded successfully on CPU.")
            except Exception as cpu_err:
                logger.critical(f"Failed to initialize MediaPipe models even on CPU: {cpu_err}")
                self.running = False
                return

        start_time = time.perf_counter()
        
        while self.running:
            # Get latest frames
            laptop_data = self.camera_reader.get_laptop_frame()
            phone_data = self.camera_reader.get_phone_frame() if self.config.get("cameras", {}).get("use_phone", False) else None
            
            if laptop_data is None:
                # No new frame yet, sleep briefly
                time.sleep(0.005)
                continue
                
            frame, frame_time = laptop_data
            phone_frame = phone_data[0] if phone_data is not None else None
            
            # Convert frame to MediaPipe Image
            # MediaPipe expects RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # MediaPipe requires frame timestamp in milliseconds as integer
            timestamp_ms = int((frame_time - start_time) * 1000)
            
            try:
                # Run landmarker inference synchronously (for video mode)
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                result = InferenceResult(
                    laptop_frame=frame,
                    timestamp=frame_time,
                    face_result=face_result,
                    hand_result=hand_result,
                    phone_frame=phone_frame
                )
                
                # Push results to Action Router, drop old results if queue is full
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Error during MediaPipe inference: {e}")
                
        face_landmarker.close()
        hand_landmarker.close()
        logger.info("Inference worker stopped, landmarker instances closed.")


if __name__ == "__main__":
    print("Testing InferenceWorker...")
    # This requires camera and models downloaded
    from .camera import CameraReader
    import yaml
    
    config = {
        "cameras": {"use_phone": False},
        "advanced": {"use_gpu": True}
    }
    
    reader = CameraReader(laptop_idx=0)
    reader.start()
    
    result_q = queue.Queue(maxsize=5)
    worker = InferenceWorker(reader, config, result_q)
    worker.start()
    
    time.sleep(5.0)
    
    if not result_q.empty():
        res = result_q.get()
        print("Success! Got inference result. Face points:", len(res.face_result.face_landmarks))
    else:
        print("No results in queue yet.")
        
    worker.stop()
    reader.stop()
