import logging
import os
import sys
from datetime import datetime
from colorlog import ColoredFormatter

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

def setup_logger(log_level="INFO"):
    logger_name = "CustomLogger"
    logger = logging.getLogger(logger_name)

    # ✅ Prevent duplicate handlers
    if logger.hasHandlers():
        return logger  

    logger.setLevel(logging.DEBUG)

    LOG_FOLDER = datetime.now().strftime("%Y_%m_%d")
    LOG_FILE = f"{log_level}_{datetime.now().strftime('%H_%M_%S')}.log"

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    logs_dir = os.path.join(base_dir, "logs", LOG_FOLDER)
    os.makedirs(logs_dir, exist_ok=True)

    LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

    formatter = ColoredFormatter(
        "%(log_color)s[ %(asctime)s ] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
