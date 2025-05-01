import os
import logging
from typing import Optional

def configure_logging(
    enable_console: bool = False,
    log_dir: Optional[str] = None
) -> None:
    """
    Configure the root logger to output INFO-level messages to a file and/or console.

    Args:
        enable_console: If True, adds a console (stdout) logging handler.
        log_dir: Optional directory path to save the log file. If provided,
                 a 'training.log' file will be created in this directory.

    Raises:
        RuntimeError: If the specified log directory cannot be created.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Define log message format
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
    )
    # File handler: write logs to file if log_dir is specified
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Could not create log directory '{log_dir}': {e}")
        log_file_path = os.path.join(log_dir, 'training.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Console handler: output logs to stdout if enabled
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)