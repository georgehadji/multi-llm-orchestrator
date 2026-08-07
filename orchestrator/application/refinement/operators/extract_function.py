"""Extract-function operator — structural refinement tier (E-11).

Targets ``longest_function_lines``. Proposes extracting one or more
helpers from the single longest function in the workspace, above the
repo's own review-standard threshold of 50 lines (see
``~/.claude/rules/common/code-review.md``: "Functions are focused
(<50 lines)").
"""

from __future__ import annotations

import logging

from ....domain.refinement import MetricSnapshot, RefinementCandidate, RefinementTier
from ....domain.testing_models import Workspace
from ._finders import find_longest_function
from .llm_refactor import LLMRefactorCommand, RefactorClientPort, propose_llm_candidate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert Python engineer performing a targeted, behavior-preserving "
    "refactor. Extract cohesive pieces of the named function into well-named private "
    "helper functions. Do not change the function's public signature, do not change "
    "any other function's public API, and do not alter behavior. Output the complete "
    "file only."
)


class ExtractFunctionOperator:
    """Strategy operator: proposes extracting the workspace's longest function."""

    name = "extract_function"
    tier = RefinementTier.STRUCTURAL

    #: Matches the repo's own code-review standard for function length.
    LENGTH_THRESHOLD = 50

    def __init__(self, client: RefactorClientPort, model: object) -> None:
        """Initialize with the injected LLM client and the model to use.

        Args:
            client: Duck-typed LLM client (``RefactorClientPort``).
            model: The model identifier/enum passed through to the client.
        """
        self._client = client
        self._model = model

    def applicable(self, snapshot: MetricSnapshot) -> bool:
        """Only propose when the workspace has a function past the threshold."""
        return snapshot.longest_function_lines > self.LENGTH_THRESHOLD

    def command_for(self, candidate: RefinementCandidate) -> LLMRefactorCommand | None:
        """Build the Command from the candidate's pre-generated payload."""
        if candidate.operator != self.name or not candidate.payload:
            return None
        return LLMRefactorCommand(candidate.target_file, candidate.payload)

    async def propose(
        self, workspace: Workspace, snapshot: MetricSnapshot
    ) -> list[RefinementCandidate]:
        """Locate the longest function in the workspace and propose one extraction."""
        located = self._find_worst(workspace)
        if located is None:
            return []
        rel, source, finding = located

        instruction = (
            f"Function `{finding.name}()` in this file spans {finding.line_count} lines "
            f"(lines {finding.lineno}-{finding.end_lineno}), above the {self.LENGTH_THRESHOLD}-line "
            "review threshold. Extract cohesive sub-steps into private helper functions "
            "so the body of `{name}` becomes a short sequence of calls.".format(name=finding.name)
        )
        return await propose_llm_candidate(
            client=self._client,
            model=self._model,
            operator_name=self.name,
            tier=self.tier,
            target_file=rel,
            source=source,
            predicted_metric="longest_function_lines",
            rationale=(
                f"extract helper(s) from `{finding.name}()` at {rel}:{finding.lineno} "
                f"({finding.line_count} lines, threshold {self.LENGTH_THRESHOLD})"
            ),
            instruction=instruction,
            system=_SYSTEM_PROMPT,
        )

    @staticmethod
    def _find_worst(workspace: Workspace):
        """Scan workspace source files for the single longest function."""
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
            finding = find_longest_function(source)
            if finding is None:
                continue
            if best is None or finding.line_count > best[2].line_count:
                rel = str(py_file.relative_to(workspace.root))
                best = (rel, source, finding)
        return best
