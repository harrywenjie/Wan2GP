from __future__ import annotations

import logging
import sys
from typing import Optional, TextIO


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
TIME_FORMAT = "%H:%M:%S"


def _coerce_level(level: str) -> int:
    candidate = getattr(logging, level.upper(), None)
    if isinstance(candidate, int):
        return candidate
    raise ValueError(f"Invalid log level: {level}")


def configure_logging(level: str = "INFO", stream: Optional[TextIO] = None) -> logging.Logger:
    """Configure a dedicated logger for CLI telemetry."""
    target_stream = stream or sys.stderr
    numeric_level = _coerce_level(level)

    logger = logging.getLogger("wan2gp.cli")
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Ensure deterministic handler setup across repeated invocations.
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(target_stream)
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, TIME_FORMAT))
    logger.addHandler(handler)

    return logger
