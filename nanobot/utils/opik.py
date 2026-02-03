"""Opik observability helpers."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.config.schema import OpikConfig


def init_opik(config: OpikConfig | None) -> Any | None:
    """Initialize Opik client if configured."""
    if not config:
        return None

    enabled = config.enabled or bool(config.host) or bool(config.api_key)
    if not enabled:
        return None

    try:
        from opik import Opik  # type: ignore
    except Exception as e:
        logger.warning(f"Opik not available: {e}")
        return None

    host = config.host or None
    api_key = config.api_key or None
    workspace = config.workspace or None
    project_name = config.project or None

    try:
        return Opik(
            project_name=project_name,
            workspace=workspace,
            host=host,
            api_key=api_key,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Opik: {e}")
        return None
