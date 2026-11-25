"""Structured logging setup for ChronoRisk modules."""

from __future__ import annotations

import logging
import os


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a configured logger without duplicate handlers.

    Args:
        name: Logger namespace (typically ``__name__``).
        level: Optional logging level override (defaults to ``CHRONORISK_LOG_LEVEL`` or ``INFO``).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    resolved_level = (level or os.getenv("CHRONORISK_LOG_LEVEL", "INFO")).upper()
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
