"""Centralized structured logging module for the safety monitoring system."""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def setup_logger(
    log_file: str = "logs/system.log",
    log_level: str = "INFO",
    logger_name: str = "SafetySystem",
) -> logging.Logger:
    """Set up and configure application logger writing to console and log_file."""
    global _LOGGER

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (clean stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"[WARN] Failed to open log file {log_file}: {e}")

    _LOGGER = logger
    return logger


def get_logger() -> logging.Logger:
    """Return the global logger instance or initialize default."""
    global _LOGGER
    if _LOGGER is None:
        return setup_logger()
    return _LOGGER
