"""Minimal shim to expose the orchestrator log_config module for legacy tests."""

from orchestrator.log_config import get_logger  # noqa: F401

__all__ = ["get_logger"]
