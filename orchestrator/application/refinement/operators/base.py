"""Refinement operators — Strategy Protocol (Phase 6, E-10/E-11).

Operators propose candidates; the service applies them through the
acceptance chain. Operators are Protocol implementations (composition over
inheritance — no Template Method hierarchy), mirroring the repo's
``ImageOptimizerStrategy`` idiom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ....domain.refinement import MetricSnapshot, RefinementCandidate, RefinementTier
from ....domain.testing_models import Workspace

if TYPE_CHECKING:
    from ..ledger import RefinementCommand


@runtime_checkable
class RefinementOperator(Protocol):
    """Proposes candidates. Reads a snapshot and workspace; returns proposals."""

    name: str
    tier: RefinementTier

    def applicable(self, snapshot: MetricSnapshot) -> bool:
        """Whether this operator has work to do for the measured snapshot.

        Gates entry so no LLM is called when the corresponding metric is
        already healthy (plan §3.4.5 / R-14).
        """
        ...

    async def propose(
        self, workspace: Workspace, snapshot: MetricSnapshot
    ) -> list[RefinementCandidate]:
        """Propose candidates ranked by predicted gain."""
        ...

    def command_for(self, candidate: RefinementCandidate) -> "RefinementCommand | None":
        """Build the reversible Command for one of this operator's own candidates.

        Keeps the service Open/Closed (plan §3.4.2): a new operator plugs
        in without the service ever branching on its name.
        """
        ...
