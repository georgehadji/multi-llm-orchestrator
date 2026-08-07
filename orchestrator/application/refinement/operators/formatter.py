"""Formatter operator — post-test formatting pass (E-10).

Re-invokes ``ruff --fix`` and ``black`` as a refinement Command so the
pre-test format pass stays (tests validate formatted code) and the
post-test pass is verified by the suite and revertible — unlike the
swallowed ``except Exception`` at output_organizer.py:231.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from ....domain.refinement import (
    MetricSnapshot,
    RefinementCandidate,
    RefinementTier,
)
from ....domain.testing_models import Workspace
from ..ledger import RefinementCommand

logger = logging.getLogger(__name__)


class FormatCommand(RefinementCommand):
    """Command: run ruff --fix then black over the workspace."""

    name = "mechanical.format"

    def __init__(self, target_files: Iterable[Path] | None = None) -> None:
        """Initialize.

        Args:
            target_files: Optional explicit files; defaults to all *.py.
        """
        self._target_files = [Path(f) for f in target_files] if target_files else None

    def apply(self, workspace_root: Path) -> list[str]:
        """Format the workspace; return the changed file paths (best-effort).

        Test files are excluded: tests are hash-locked and read-only for the
        entire refinement pass (E-11 invariant 2).
        """
        if self._target_files is None:
            sources = sorted(workspace_root.rglob("*.py"))
        else:
            sources = [f if f.is_absolute() else workspace_root / f for f in self._target_files]
        sources = [
            p
            for p in sources
            if "__pycache__" not in str(p)
            and not (p.name.startswith("test_") or p.name.endswith("_test.py"))
        ]

        changed: list[str] = []

        def _norm(data: bytes) -> bytes:
            # Compare modulo line endings: black/ruff may normalize EOLs on
            # Windows; a pure EOL change is not a meaningful edit.
            return data.replace(b"\r\n", b"\n")

        before = {str(p): _norm(p.read_bytes()) for p in sources if p.exists()}
        cache_dir = workspace_root / ".orch-ruff-cache"
        for tool in ("ruff", "black"):
            executable = shutil.which(tool)
            if executable is None:
                logger.debug("formatter %s not installed — skipped", tool)
                continue
            if tool == "ruff":
                args = [
                    executable,
                    "check",
                    "--fix",
                    "--cache-dir",
                    str(cache_dir),
                    *[str(p) for p in sources],
                ]
            else:
                args = [executable, *[str(p) for p in sources]]
            try:
                subprocess.run(  # nosec B603 - fixed arg list, no shell
                    args, cwd=str(workspace_root), capture_output=True, timeout=60
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.debug("formatter %s failed: %s", tool, exc)

        # Never leave transient tooling state in the workspace (revert and
        # byte-exactness must not be polluted by a cache directory).
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        for path in sources:
            if not path.exists():
                continue
            if before.get(str(path)) != _norm(path.read_bytes()):
                changed.append(str(path.relative_to(workspace_root)))
        return changed


class FormatterOperator:
    """Strategy operator: proposes a formatting pass when formatting tools exist."""

    name = "formatter"
    tier = RefinementTier.MECHANICAL

    def applicable(self, snapshot: MetricSnapshot) -> bool:
        """Formatting is deterministic and cheap; always worth a pass."""
        return True

    def command_for(self, candidate: RefinementCandidate) -> FormatCommand | None:
        """Build the Command for one of this operator's own candidates."""
        if candidate.operator != self.name:
            return None
        return FormatCommand()

    async def propose(
        self, workspace: Workspace, snapshot: MetricSnapshot
    ) -> list[RefinementCandidate]:
        """Propose a single formatting candidate (no diff preview needed)."""
        return [
            RefinementCandidate(
                operator=self.name,
                tier=self.tier,
                target_file="*",
                rationale="post-test formatting pass (ruff --fix + black)",
                diff="<formatting>",
                predicted_metric="total_lines",
            )
        ]
