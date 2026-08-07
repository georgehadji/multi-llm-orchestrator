"""Flatten-nesting operator — structural refinement tier (E-11).

Targets ``max_nesting_depth``. Proposes flattening the single most deeply
nested function via early returns and guard clauses, above the repo's own
review-standard threshold of 4 levels (see
``~/.claude/rules/common/coding-style.md``: "Deep Nesting ... Prefer early
returns over nested conditionals").
"""

from __future__ import annotations

import logging

from ....domain.refinement import MetricSnapshot, RefinementCandidate, RefinementTier
from ....domain.testing_models import Workspace
from ._finders import find_deepest_nesting_function
from .llm_refactor import LLMRefactorCommand, RefactorClientPort, propose_llm_candidate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert Python engineer performing a targeted, behavior-preserving "
    "refactor. Flatten the named function's control flow using early returns and guard "
    "clauses instead of deeply nested conditionals. Do not change the function's public "
    "signature, do not change any other function's public API, and do not alter "
    "behavior. Output the complete file only."
)


class FlattenNestingOperator:
    """Strategy operator: proposes flattening the workspace's deepest nesting."""

    name = "flatten_nesting"
    tier = RefinementTier.STRUCTURAL

    #: Matches the repo's own coding-style standard for nesting depth.
    NESTING_THRESHOLD = 4

    def __init__(self, client: RefactorClientPort, model: object) -> None:
        """Initialize with the injected LLM client and the model to use."""
        self._client = client
        self._model = model

    def applicable(self, snapshot: MetricSnapshot) -> bool:
        """Only propose when the workspace has nesting past the threshold."""
        return snapshot.max_nesting_depth > self.NESTING_THRESHOLD

    def command_for(self, candidate: RefinementCandidate) -> LLMRefactorCommand | None:
        """Build the Command from the candidate's pre-generated payload."""
        if candidate.operator != self.name or not candidate.payload:
            return None
        return LLMRefactorCommand(candidate.target_file, candidate.payload)

    async def propose(
        self, workspace: Workspace, snapshot: MetricSnapshot
    ) -> list[RefinementCandidate]:
        """Locate the most deeply nested function and propose flattening it."""
        located = self._find_worst(workspace)
        if located is None:
            return []
        rel, source, finding = located

        instruction = (
            f"Function `{finding.name}()` in this file reaches {finding.metric_value} levels "
            f"of nesting (lines {finding.lineno}-{finding.end_lineno}), above the "
            f"{self.NESTING_THRESHOLD}-level review threshold. Rewrite it using early returns "
            "and guard clauses so nesting stays shallow."
        )
        return await propose_llm_candidate(
            client=self._client,
            model=self._model,
            operator_name=self.name,
            tier=self.tier,
            target_file=rel,
            source=source,
            predicted_metric="max_nesting_depth",
            rationale=(
                f"flatten `{finding.name}()` at {rel}:{finding.lineno} "
                f"({finding.metric_value} levels, threshold {self.NESTING_THRESHOLD})"
            ),
            instruction=instruction,
            system=_SYSTEM_PROMPT,
        )

    @staticmethod
    def _find_worst(workspace: Workspace):
        """Scan workspace source files for the single most-nested function."""
        best = None
        for py_file in sorted(workspace.root.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            if py_file.name.startswith("test_") or py_file.name.endswith("_test.py"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - best-effort
                continue
            finding = find_deepest_nesting_function(source)
            if finding is None:
                continue
            if best is None or finding.metric_value > best[2].metric_value:
                rel = str(py_file.relative_to(workspace.root))
                best = (rel, source, finding)
        return best
