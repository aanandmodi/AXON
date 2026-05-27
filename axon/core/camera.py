import cv2
import time
import queue
import threading
from typing import Optional, Tuple
from ..utils.logger import logger

class CameraReader:
    """Reads frames from the laptop camera and optionally an RTSP phone camera feed."""
    def __init__(self, laptop_idx: int = 0, phone_url: Optional[str] = None, 
                 laptop_res: Tuple[int, int] = (1280, 720), phone_res: Tuple[int, int] = (640, 480)):
        self.laptop_idx = laptop_idx
        self.phone_url = phone_url
        self.laptop_res = laptop_res
        self.phone_res = phone_res
        
        # Double-buffer style queues (keeps latest frames, drops older ones)
        self.laptop_queue = queue.Queue(maxsize=2)
        self.phone_queue = queue.Queue(maxsize=2)
        
        self.running = False
        self.laptop_thread: Optional[threading.Thread] = None
        self.phone_thread: Optional[threading.Thread] = None
        
        self.laptop_cap: Optional[cv2.VideoCapture] = None
        self.phone_cap: Optional[cv2.VideoCapture] = None
        
        self.laptop_active = False
        self.phone_active = False

    def start(self):
        """Starts camera reader threads."""
        self.running = True
        
        # Start laptop camera reader
        self.laptop_thread = threading.Thread(target=self._read_laptop, daemon=True, name="LaptopCamReader")
        self.laptop_thread.start()
        
        # Start phone camera reader if URL is provided and active
        if self.phone_url:
            self.phone_thread = threading.Thread(target=self._read_phone, daemon=True, name="PhoneCamReader")
            self.phone_thread.start()

    def stop(self):
        """Stops camera reader threads and releases resources."""
        self.running = False
        if self.laptop_thread:
            self.laptop_thread.join(timeout=1.0)
        if self.phone_thread:
            self.phone_thread.join(timeout=1.0)
            
        if self.laptop_cap:
            self.laptop_cap.release()
            logger.info("Laptop camera released.")
        if self.phone_cap:
            self.phone_cap.release()
            logger.info("Phone camera released.")

    def _read_laptop(self):
        logger.info(f"Opening laptop camera index {self.laptop_idx}...")
        self.laptop_cap = cv2.VideoCapture(self.laptop_idx, cv2.CAP_DSHOW)  # CAP_DSHOW is faster on Windows
        
        self.laptop_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.laptop_res[0])
        self.laptop_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.laptop_res[1])
        self.laptop_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer size 1 to get latest frames
        
        if not self.laptop_cap.isOpened():
            logger.error(f"Failed to open laptop camera at index {self.laptop_idx}")
            self.laptop_active = False
            return
            
        self.laptop_active = True
        logger.info("Laptop camera successfully opened.")
        
        while self.running:
            ret, frame = self.laptop_cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from laptop camera.")
                time.sleep(0.01)
                continue
                
            timestamp = time.perf_counter()
            # Push latest, drop old if full
            if self.laptop_queue.full():
                try:
                    self.laptop_queue.get_nowait()
                except queue.Empty:
                    pass
            self.laptop_queue.put((frame, timestamp))
            
        self.laptop_active = False

    def _read_phone(self):
        backoff = 1.0
        while self.running:
            logger.info(f"Connecting to phone RTSP: {self.phone_url}...")
            self.phone_cap = cv2.VideoCapture(self.phone_url)
            self.phone_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.phone_res[0])
            self.phone_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.phone_res[1])
            self.phone_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.phone_cap.isOpened():
                logger.warning(f"Failed to connect to phone camera. Reconnecting in {backoff:.1f}s...")
                self.phone_active = False
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
                continue
                
            self.phone_active = True
            logger.info("Successfully connected to phone camera stream.")
            backoff = 1.0  # Reset backoff
            
            while self.running:
                ret, frame = self.phone_cap.read()
                if not ret or frame is None:
                    logger.warning("Phone camera stream disconnected. Attempting reconnection...")
                    self.phone_active = False
                    break
                    
                timestamp = time.perf_counter()
                if self.phone_queue.full():
                    try:
                        self.phone_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.phone_queue.put((frame, timestamp))
            
            if self.phone_cap:
                self.phone_cap.release()

    def get_laptop_frame(self) -> Optional[Tuple[cv2.Mat, float]]:
        """Returns the latest laptop frame and its timestamp."""
        try:
            return self.laptop_queue.get_nowait()
        except queue.Empty:
            return None

    def get_phone_frame(self) -> Optional[Tuple[cv2.Mat, float]]:
        """Returns the latest phone frame and its timestamp."""
        try:
            return self.phone_queue.get_nowait()
        except queue.Empty:
            return None

    def __repr__(self) -> str:
        return f"CameraReader(laptop_active={self.laptop_active}, phone_active={self.phone_active})"


if __name__ == "__main__":
    # Small test
    print("Testing CameraReader...")
    reader = CameraReader(laptop_idx=0)
    reader.start()
    time.sleep(2.0)
    frame_data = reader.get_laptop_frame()
    if frame_data is not None:
        print("Success! Captured laptop frame.")
    else:
        print("Could not capture laptop frame.")
    reader.stop()
