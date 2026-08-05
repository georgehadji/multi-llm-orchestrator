"""Sandbox isolation adapters — subprocess and Docker tiers (F-3).

Tier selection ladder, resolved at composition time:

| Tier       | Selected when                                          | Protections                        |
|------------|--------------------------------------------------------|------------------------------------|
| ``DOCKER`` | daemon reachable and ``ORCH_SANDBOX_TIER != subprocess``| network=none, read-only, caps drop |
| ``SUBPROCESS`` | Docker unavailable or forced                    | rlimits, process-group kill, scrubbed env, temp HOME |

There is **no** ``NONE`` tier: :func:`resolve_sandbox` always returns a
real isolation boundary, and :func:`require_sandbox` raises if neither
backend is available rather than executing on the host.
"""

from __future__ import annotations

import logging
import os

from .docker_sandbox import DockerSandbox, _docker_available
from .subprocess_sandbox import SubprocessSandbox

logger = logging.getLogger(__name__)

__all__ = ["DockerSandbox", "SubprocessSandbox", "resolve_sandbox", "require_sandbox"]


def resolve_sandbox(*, tier: str | None = None) -> SubprocessSandbox | DockerSandbox:
    """Resolve the sandbox tier for this process.

    Args:
        tier: One of ``"docker"``, ``"subprocess"``, or ``"auto"``. Defaults
            to ``ORCH_SANDBOX_TIER`` env var, then ``"auto"``.

    Returns:
        A concrete sandbox. Never returns a ``NONE`` tier.

    Raises:
        RuntimeError: If ``tier="docker"`` is requested but the Docker
            daemon is unreachable.
    """
    selected = (tier or os.environ.get("ORCH_SANDBOX_TIER", "auto")).lower()
    if selected == "subprocess":
        logger.info("Sandbox tier: SUBPROCESS (forced by config)")
        return SubprocessSandbox()
    if selected == "docker":
        if not _docker_available():
            raise RuntimeError(
                "ORCH_SANDBOX_TIER=docker but the Docker daemon is unreachable "
                "(docker info failed). Set ORCH_SANDBOX_TIER=subprocess or start Docker."
            )
        logger.info("Sandbox tier: DOCKER (forced by config)")
        return DockerSandbox()
    # auto
    if _docker_available():
        logger.info("Sandbox tier: DOCKER (auto-detected)")
        return DockerSandbox()
    logger.info("Sandbox tier: SUBPROCESS (Docker unavailable)")
    return SubprocessSandbox()


def require_sandbox(*, tier: str | None = None) -> SubprocessSandbox | DockerSandbox:
    """Like :func:`resolve_sandbox` but never silently downgrades.

    If ``ORCH_SANDBOX_TIER=docker`` is set and the daemon is unreachable,
    this raises — the caller (TestingService) then refuses to execute rather
    than running unisolated on the host.
    """
    return resolve_sandbox(tier=tier)
