"""
Logging — Structured logging configuration
============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Unified logging across the orchestrator using structlog.
Provides configured loggers for both development (console, colored)
and production (JSON) environments.

Usage:
    from orchestrator.logging import get_logger

    log = get_logger(__name__)
    log.info("task_started", task_id="t_001", project_id="p_42")
    log.warning("budget_exceeded", phase="generation", spent=12.5, cap=10.0)

Environment:
    LOG_FORMAT=json      — JSON output (production, default: console)
    LOG_LEVEL=DEBUG      — Override log level (default: INFO)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


def configure_logging(
    level: str | None = None,
    json_format: bool | None = None,
) -> None:
    """Configure structlog processors and handlers.

    Call once at application startup (or lazily on first use).
    Idempotent — calling multiple times is safe.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
               Falls back to LOG_LEVEL env var, then INFO.
        json_format: If True, emit JSON lines. Falls back to
                     LOG_FORMAT env var, then console mode.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if json_format is None:
        json_format = os.environ.get("LOG_FORMAT", "") == "json"

    # Configure standard logging for structlog to wrap
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
        force=True,
    )

    try:
        import structlog

        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(
                structlog.dev.ConsoleRenderer(
                    colors=sys.stderr.isatty(),
                    sort_keys=False,
                )
            )

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    except ImportError:
        # Fallback to stdlib logging if structlog is not installed
        _fallback_configure(level)


def _fallback_configure(level: str) -> None:
    """Configure stdlib logging when structlog is unavailable."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Get a structured logger.

    Returns structlog.BoundLogger if structlog is installed,
    falls back to stdlib logging.Logger otherwise.

    Args:
        name: Logger name (usually __name__). If None, uses root logger.

    Returns:
        A logger-compatible object with .info(), .debug(), .warning(), .error().
    """
    try:
        import structlog

        return structlog.get_logger(name) if name else structlog.get_logger()
    except ImportError:
        return logging.getLogger(name) if name else logging.getLogger()


__all__ = ["configure_logging", "get_logger"]
