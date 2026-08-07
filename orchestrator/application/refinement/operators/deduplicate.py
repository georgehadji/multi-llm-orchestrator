"""Deduplicate operator — structural refinement tier (E-11).

Targets ``duplicated_blocks``. Proposes extracting a shared helper for the
first pair of structurally-identical functions found in a single file.
Scoped to same-file pairs only: cross-file deduplication would need to
update call sites and imports in a second file, which a single-target-file
Command cannot express safely (see ``_finders.find_duplicate_pair``).
"""

from __future__ import annotations

import logging

from ....domain.refinement import MetricSnapshot, RefinementCandidate, RefinementTier
from ....domain.testing_models import Workspace
from ._finders import find_duplicate_pair
from .llm_refactor import LLMRefactorCommand, RefactorClientPort, propose_llm_candidate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert Python engineer performing a targeted, behavior-preserving "
    "refactor. Two functions in this file are structurally identical (same logic, "
    "different names/literals). Extract their shared logic into one private helper "
    "and have both call it. Do not change either function's public signature, do not "
    "change any other function's public API, and do not alter behavior. Output the "
    "complete file only."
)


class DeduplicateOperator:
    """Strategy operator: proposes extracting a shared helper for duplicate functions."""

    name = "deduplicate"
    tier = RefinementTier.STRUCTURAL

    def __init__(self, client: RefactorClientPort, model: object) -> None:
        """Initialize with the injected LLM client and the model to use."""
        self._client = client
        self._model = model

    def applicable(self, snapshot: MetricSnapshot) -> bool:
        """Only propose when the workspace has at least one duplicated block."""
        return snapshot.duplicated_blocks > 0

    def command_for(self, candidate: RefinementCandidate) -> LLMRefactorCommand | None:
        """Build the Command from the candidate's pre-generated payload."""
        if candidate.operator != self.name or not candidate.payload:
            return None
        return LLMRefactorCommand(candidate.target_file, candidate.payload)

    async def propose(
        self, workspace: Workspace, snapshot: MetricSnapshot
    ) -> list[RefinementCandidate]:
        """Find the first same-file duplicate function pair and propose merging them."""
        located = self._find_first_pair(workspace)
        if located is None:
            return []
        rel, source, first, second = located

        instruction = (
            f"Functions `{first.name}()` (lines {first.lineno}-{first.end_lineno}) and "
            f"`{second.name}()` (lines {second.lineno}-{second.end_lineno}) in this file "
            "have the same structure. Extract their shared logic into one private helper "
            "that both call."
        )
        return await propose_llm_candidate(
            client=self._client,
            model=self._model,
            operator_name=self.name,
            tier=self.tier,
            target_file=rel,
            source=source,
            predicted_metric="duplicated_blocks",
            rationale=(
                f"deduplicate `{first.name}()`/`{second.name}()` in {rel} "
                f"(lines {first.lineno} and {second.lineno})"
            ),
            instruction=instruction,
            system=_SYSTEM_PROMPT,
        )

    @staticmethod
    def _find_first_pair(workspace: Workspace):
        """Scan workspace source files for the first same-file duplicate pair."""
        for py_file in sorted(workspace.root.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            if py_file.name.startswith("test_") or py_file.name.endswith("_test.py"):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - best-effort
                continue
            pair = find_duplicate_pair(source)
            if pair is None:
                continue
            rel = str(py_file.relative_to(workspace.root))
            return rel, source, pair[0], pair[1]
        return None
