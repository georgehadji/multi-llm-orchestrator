"""Dead-code operator — deterministic unused-import removal (E-10).

Zero-LLM, AST-based: removes top-level imports whose names are never
referenced in the module. The Command applies the transformation and the
service re-verifies with the suite (tests are hash-locked and immutable).
"""

from __future__ import annotations

import ast
import logging
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


def remove_unused_imports(source: str) -> str:
    """Remove top-level imports whose names are never referenced.

    Args:
        source: Python module source.

    Returns:
        The module with unused imports removed (best-effort; unchanged on
        any AST failure).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Collect the set of names actually used outside import statements.
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Attribute chains: the leaf attribute name matters only if the
            # whole expression is referenced; keep the root name.
            root: ast.expr = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)

    # Names bound by imports that are never used elsewhere.
    changed = False
    new_body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            kept = [a for a in node.names if (a.asname or a.name.split(".")[0]) in used]
            if not kept:
                changed = True
                continue  # drop the whole import
            if len(kept) != len(node.names):
                node.names = kept
                changed = True
        elif isinstance(node, ast.ImportFrom):
            kept = [a for a in node.names if (a.asname or a.name) in used]
            if not kept:
                changed = True
                continue  # drop the whole import
            if len(kept) != len(node.names):
                node.names = kept
                changed = True
        new_body.append(node)

    if not changed:
        return source

    tree.body = new_body
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:  # pragma: no cover - unparse is robust
        return source


class RemoveUnusedImportsCommand(RefinementCommand):
    """Command: strip unused imports from a set of workspace files."""

    name = "dead_code.unused_imports"

    def __init__(self, target_files: Iterable[Path]) -> None:
        """Initialize with the files to process (relative or absolute)."""
        self._target_files = [Path(f) for f in target_files]

    def apply(self, workspace_root: Path) -> list[str]:
        """Apply unused-import removal; return changed file paths."""
        changed: list[str] = []
        for target in self._target_files:
            path = target if target.is_absolute() else workspace_root / target
            if not path.exists():
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - best-effort
                continue
            cleaned = remove_unused_imports(source)
            if cleaned != source:
                path.write_text(cleaned, encoding="utf-8")
                changed.append(str(target))
        return changed


class DeadCodeOperator:
    """Strategy operator: proposes unused-import removal candidates."""

    name = "dead_code"
    tier = RefinementTier.MECHANICAL

    #: Only propose when there is something to remove (unused imports
    #: cannot be predicted from the snapshot, so this always proposes —
    #: the acceptance chain decides. Keep it cheap: one candidate max.)
    MAX_DEAD_CODE_CANDIDATES = 1

    def applicable(self, snapshot: MetricSnapshot) -> bool:
        """Dead-code pass is always worth attempting (deterministic, free)."""
        return True

    def command_for(self, candidate: RefinementCandidate) -> RemoveUnusedImportsCommand | None:
        """Build the Command for one of this operator's own candidates."""
        if candidate.operator != self.name:
            return None
        return RemoveUnusedImportsCommand([Path(candidate.target_file)])

    async def propose(
        self, workspace: Workspace, snapshot: MetricSnapshot
    ) -> list[RefinementCandidate]:
        """Propose removing unused imports across workspace source files.

        Uses the AST-based detector to find files with removable imports
        before proposing (a candidate with no diff is never proposed).
        """
        candidates: list[RefinementCandidate] = []
        for py_file in sorted(workspace.root.rglob("*.py")):
            if "__pycache__" in str(py_file) or "test" in py_file.name.lower():
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover
                continue
            cleaned = remove_unused_imports(source)
            if cleaned == source:
                continue
            rel = str(py_file.relative_to(workspace.root))
            candidates.append(
                RefinementCandidate(
                    operator=self.name,
                    tier=self.tier,
                    target_file=rel,
                    rationale="remove unused imports",
                    diff="<unused imports removed>",
                    predicted_metric="total_lines",
                )
            )
            if len(candidates) >= self.MAX_DEAD_CODE_CANDIDATES:
                break
        return candidates
