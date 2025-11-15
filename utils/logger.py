"""Centralized logging configuration utilities."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from utils import env


def configure_logger(name: str) -> logging.Logger:
    """Return a logger with console and optional rotating file handler."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file = os.getenv(env.APP_LOG)
    if log_file:
        try:
            file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            stream_handler.setLevel(logging.WARNING)
            logger.warning("Could not create log file handler at %s", log_file)

    logger.setLevel(logging.DEBUG)
    return logger
