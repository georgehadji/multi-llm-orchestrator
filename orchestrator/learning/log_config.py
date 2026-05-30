"""Logging helpers for the learning sub-package."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under orchestrator.learning."""
    return logging.getLogger(f"orchestrator.learning.{name}")
