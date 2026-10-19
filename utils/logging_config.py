"""Logging setup: console + rotating file handler in ``logs/``."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import config

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the ``metaopt`` logger once and return it."""
    global _CONFIGURED
    logger = logging.getLogger("metaopt")
    if _CONFIGURED:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.LOGS_DIR / "metaopt.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:  # read-only file system: console logging is still available
        logger.warning("File logging unavailable; continuing with console logging only.")
    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of ``metaopt``."""
    setup_logging()
    return logging.getLogger(f"metaopt.{name}")
