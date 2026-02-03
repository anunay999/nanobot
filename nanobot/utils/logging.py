"""Logging helpers for nanobot."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from nanobot.utils.helpers import ensure_dir, get_data_path


def setup_logging(
    *,
    verbose: bool = False,
    log_path: Path | None = None,
) -> Path:
    """
    Configure loguru sinks and formatting.

    Returns the resolved log file path.
    """
    log_dir = ensure_dir(get_data_path() / "logs")
    resolved_path = log_path or (log_dir / "nanobot.log")

    logger.remove()
    logger.configure(
        extra={
            "message_id": "-",
            "channel": "-",
            "chat_id": "-",
            "sender_id": "-",
            "session_key": "-",
        }
    )

    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "{extra[message_id]} | {extra[channel]}:{extra[chat_id]} | "
        "{message}"
    )
    file_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "{extra[message_id]} | {extra[channel]}:{extra[chat_id]} | "
        "{extra[session_key]} | {message}"
    )

    console_level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=console_level, format=fmt, backtrace=False, diagnose=False)
    logger.add(
        str(resolved_path),
        level="DEBUG",
        format=file_fmt,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    return resolved_path
