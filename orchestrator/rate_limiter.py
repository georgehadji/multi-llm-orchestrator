"""
RateLimiter — Sliding-window TPM/RPM rate limiting per tenant and model.
=========================================================================
Implements in-memory, per-(tenant, model) rate limiting using a 60-second
sliding window. No external dependencies required.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

from .log_config import get_logger

logger = get_logger(__name__)

_WINDOW_SECONDS: float = 60.0


class RateLimitExceeded(Exception):
    """Raised when a tenant/model would exceed its TPM or RPM limit."""

    def __init__(self, limit_type: str, retry_after: float, tenant: str, model: str) -> None:
        self.limit_type = limit_type      # "tpm" or "rpm"
        self.retry_after = retry_after    # seconds until oldest entry expires
        self.tenant = tenant
        self.model = model
        super().__init__(
            f"Rate limit exceeded ({limit_type}) for tenant={tenant!r} model={model!r}. "
            f"Retry after {retry_after:.1f}s"
        )


_Entry = Tuple[float, int]


class RateLimiter:
    """Per-(tenant, model) sliding-window rate limiter."""

    def __init__(self) -> None:
        self._windows: Dict[Tuple[str, str], deque] = defaultdict(deque)
        self._limits: Dict[Tuple[str, str], Dict[str, int]] = {}

    def set_limits(self, tenant: str, model: str, tpm: int, rpm: int) -> None:
        """Set TPM and RPM limits for a (tenant, model) pair."""
        if tpm <= 0 or rpm <= 0:
            raise ValueError("tpm and rpm must be positive integers")
        self._limits[(tenant, model)] = {"tpm": tpm, "rpm": rpm}
        logger.debug("Rate limits set: tenant=%r model=%r tpm=%d rpm=%d", tenant, model, tpm, rpm)

    def check(self, tenant: str, model: str, tokens: int) -> None:
        """
        Check whether a request of `tokens` tokens is within limits.
        Raises RateLimitExceeded if the request would exceed TPM or RPM.
        No-ops if no limits have been set for this (tenant, model) pair.
        """
        limits = self._limits.get((tenant, model))
        if limits is None:
            return

        window = self._windows[(tenant, model)]
        self._evict(window)

        if len(window) >= limits["rpm"]:
            retry_after = self._retry_after(window)
            raise RateLimitExceeded("rpm", retry_after, tenant, model)

        tokens_used = sum(t for _, t in window)
        if tokens_used + tokens > limits["tpm"]:
            retry_after = self._retry_after(window)
            raise RateLimitExceeded("tpm", retry_after, tenant, model)

    def record(self, tenant: str, model: str, tokens: int) -> None:
        """Record a completed request."""
        window = self._windows[(tenant, model)]
        self._evict(window)
        window.append((time.monotonic(), tokens))

    def get_usage(self, tenant: str, model: str) -> Dict[str, int]:
        """Return current sliding-window usage for (tenant, model)."""
        window = self._windows[(tenant, model)]
        self._evict(window)
        return {
            "tokens_used": sum(t for _, t in window),
            "requests": len(window),
        }

    @staticmethod
    def _evict(window: deque) -> None:
        cutoff = time.monotonic() - _WINDOW_SECONDS
        while window and window[0][0] < cutoff:
            window.popleft()

    @staticmethod
    def _retry_after(window: deque) -> float:
        if not window:
            return 0.0
        oldest_ts = window[0][0]
        return max(0.0, oldest_ts + _WINDOW_SECONDS - time.monotonic())
