"""Shim that mirrors the orchestrator logging helpers for testing."""

from orchestrator import log_config  # noqa: F401

__all__ = ["log_config"]
