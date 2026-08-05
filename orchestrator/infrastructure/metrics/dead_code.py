"""Dead-code collector — vulture adapter with graceful absence (E-9).

Runs ``vulture`` over the workspace's source files when the tool is
installed; a missing vulture degrades to ``dead_symbols=None`` on the
snapshot rather than crashing. The result feeds ``MetricSnapshot``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def vulture_available() -> bool:
    """Return True when the ``vulture`` CLI is on PATH."""
    return shutil.which("vulture") is not None


async def count_dead_symbols(workspace_root: Path, timeout_s: float = 30.0) -> int | None:
    """Count dead symbols in a workspace via vulture (async, best-effort).

    Args:
        workspace_root: Workspace root to scan (source files only — tests
            are excluded because vulture flags test helpers as unused).
        timeout_s: Wall-clock bound.

    Returns:
        Dead-symbol count, or ``None`` when vulture is unavailable or the
        scan fails (graceful degradation — never blocks refinement).
    """
    if not vulture_available():
        logger.debug("vulture not installed — dead_symbols unavailable")
        return None

    sources = sorted(workspace_root.rglob("*.py"))
    sources = [p for p in sources if "test" not in p.name.lower() and "__pycache__" not in str(p)]
    if not sources:
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            "vulture",
            *[str(p) for p in sources],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        output = stdout_bytes.decode("utf-8", errors="replace")
    except (asyncio.TimeoutError, OSError) as exc:
        logger.debug("vulture scan failed: %s", exc)
        return None

    # vulture prints one finding per line; count them.
    count = 0
    for line in output.splitlines():
        if "unused" in line or "unreachable" in line or ":" in line:
            if line.strip():
                count += 1
    return count
