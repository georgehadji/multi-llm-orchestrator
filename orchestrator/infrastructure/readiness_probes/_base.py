"""Template method shared by every static probe (Phase 7, P-2).

A probe never raises for an ordinary observation failure (domain/ports.py
``ProbePort`` contract). Centralizing the try/except here means individual
probes stay pure "what does satisfied look like" logic — the wrapping is
never re-implemented per probe, and it can never accidentally be forgotten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...domain.readiness import Evidence, ProbeResult, RequirementOutcome

if TYPE_CHECKING:
    from ...domain.ports import ProbeContext
    from ...domain.testing_models import Workspace


class StaticProbe:
    requirement_id: str

    def check(self, workspace: "Workspace") -> RequirementOutcome:
        raise NotImplementedError

    async def probe(self, workspace: "Workspace", ctx: "ProbeContext") -> RequirementOutcome:
        try:
            return self.check(workspace)
        except Exception as exc:  # noqa: BLE001 - contract: a probe never raises
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.INDETERMINATE,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail=f"probe raised {type(exc).__name__}: {exc}",
                    ),
                ),
            )


__all__ = ["StaticProbe"]
