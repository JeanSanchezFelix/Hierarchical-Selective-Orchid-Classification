import os
import logging
from typing import Optional

def configure_logging(logs: bool = False, save_dir: Optional[str] = None):
    """
    Configures logging to write to a file (if save_dir is provided) 
    and optionally to the console.

    Parameters:
        logs (bool): Whether to log messages to the console.
        save_dir (Optional[str]): Directory where the log file will be saved.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s')

    # File handler (if save_dir is provided)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Optional console stream handler
    if logs:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)