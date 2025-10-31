from __future__ import annotations

import logging
from logging import Logger
from typing import Optional

_LOGGER: Optional[Logger] = None


def configure_notifications(logger: Logger) -> None:
    """Route notification helpers to the provided logger."""
    global _LOGGER  # noqa: PLW0603 - intentional global configuration
    _LOGGER = logger


def _get_logger() -> Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    fallback_logger = logging.getLogger("wan2gp.notifications")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    return fallback_logger


def get_notifications_logger() -> Logger:
    """Expose the currently configured notifications logger."""
    return _get_logger()


def notify_debug(message: str) -> None:
    _get_logger().debug(message)


def notify_info(message: str) -> None:
    _get_logger().info(message)


def notify_warning(message: str) -> None:
    _get_logger().warning(message)


def notify_error(message: str) -> None:
    _get_logger().error(message)
