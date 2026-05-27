import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "axon") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers if already initialized
    if logger.handlers:
        return logger
        
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "axon.log")
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Global logger
logger = setup_logger()
