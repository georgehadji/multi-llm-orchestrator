"""StaticProbe template method never lets an exception escape (Phase 7, P-2)."""

from __future__ import annotations

import pytest

from orchestrator.domain.ports import ProbeContext
from orchestrator.domain.readiness import AppArchetype, ProbeResult
from orchestrator.infrastructure.readiness_probes._base import StaticProbe


class _ExplodingProbe(StaticProbe):
    requirement_id = "test.exploding"

    def check(self, workspace):
        raise RuntimeError("boom")


@pytest.mark.unit
async def test_probe_never_raises_degrades_to_indeterminate():
    outcome = await _ExplodingProbe().probe(
        workspace=None, ctx=ProbeContext(archetype=AppArchetype.PYTHON_CLI)
    )
    assert outcome.result == ProbeResult.INDETERMINATE
    assert "boom" in outcome.evidence[0].detail
