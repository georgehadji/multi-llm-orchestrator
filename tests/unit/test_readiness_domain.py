"""
Tests for the readiness domain model (P-1).
=============================================
Pure roll-up scoring: a level is achieved only when every blocking
requirement at that level AND all levels below it is satisfied.
"""

from __future__ import annotations

import pytest

from orchestrator.domain.readiness import (
    AppArchetype,
    Evidence,
    ProbeResult,
    ReadinessLevel,
    ReadinessReport,
    Requirement,
    RequirementOutcome,
    achieved_level,
)

CLI = AppArchetype.PYTHON_CLI
SERVICE = AppArchetype.PYTHON_SERVICE


def _req(id: str, level: ReadinessLevel, *, blocking: bool = True, archetypes=frozenset({CLI})):
    return Requirement(
        id=id,
        title=id,
        derives_from="test fixture",
        level=level,
        archetypes=frozenset(archetypes),
        blocking=blocking,
    )


def _outcome(req_id: str, result: ProbeResult) -> RequirementOutcome:
    return RequirementOutcome(requirement_id=req_id, result=result)


@pytest.mark.unit
class TestAchievedLevel:
    def test_all_satisfied_reaches_top_level(self) -> None:
        requirements = (
            _req("l0", ReadinessLevel.PROTOTYPE),
            _req("l1", ReadinessLevel.DEPLOYABLE),
            _req("l2", ReadinessLevel.OPERABLE),
        )
        outcomes = (
            _outcome("l0", ProbeResult.SATISFIED),
            _outcome("l1", ProbeResult.SATISFIED),
            _outcome("l2", ProbeResult.SATISFIED),
        )
        assert achieved_level(outcomes, requirements, CLI) == ReadinessLevel.OPERABLE

    def test_satisfied_l2_with_violated_l1_yields_l0(self) -> None:
        """The classic roll-up error: a higher level can't rescue a lower failure."""
        requirements = (
            _req("l1", ReadinessLevel.DEPLOYABLE),
            _req("l2", ReadinessLevel.OPERABLE),
        )
        outcomes = (
            _outcome("l1", ProbeResult.VIOLATED),
            _outcome("l2", ProbeResult.SATISFIED),
        )
        assert achieved_level(outcomes, requirements, CLI) == ReadinessLevel.PROTOTYPE

    def test_indeterminate_never_counts_as_satisfied(self) -> None:
        requirements = (_req("l1", ReadinessLevel.DEPLOYABLE),)
        outcomes = (_outcome("l1", ProbeResult.INDETERMINATE),)
        assert achieved_level(outcomes, requirements, CLI) == ReadinessLevel.PROTOTYPE

    def test_not_applicable_does_not_block_level(self) -> None:
        requirements = (
            _req("l1", ReadinessLevel.DEPLOYABLE),
            _req("l2", ReadinessLevel.OPERABLE),
        )
        outcomes = (
            _outcome("l1", ProbeResult.NOT_APPLICABLE),
            _outcome("l2", ProbeResult.SATISFIED),
        )
        assert achieved_level(outcomes, requirements, CLI) == ReadinessLevel.OPERABLE

    def test_missing_outcome_blocks_level(self) -> None:
        requirements = (_req("l1", ReadinessLevel.DEPLOYABLE),)
        assert achieved_level((), requirements, CLI) == ReadinessLevel.PROTOTYPE

    def test_non_blocking_violation_does_not_block_level(self) -> None:
        requirements = (_req("advisory", ReadinessLevel.DEPLOYABLE, blocking=False),)
        outcomes = (_outcome("advisory", ProbeResult.VIOLATED),)
        assert achieved_level(outcomes, requirements, CLI) == ReadinessLevel.DEPLOYABLE

    def test_requirement_for_a_different_archetype_is_ignored(self) -> None:
        requirements = (_req("service-only", ReadinessLevel.DEPLOYABLE, archetypes={SERVICE}),)
        # No requirement at all applies to CLI — nothing was characterized,
        # so the roll-up stays at the floor rather than vacuously advancing.
        assert achieved_level((), requirements, CLI) == ReadinessLevel.PROTOTYPE


@pytest.mark.unit
class TestReadinessReport:
    def test_passed_true_when_achieved_meets_required(self) -> None:
        report = ReadinessReport(
            archetype=CLI,
            achieved_level=ReadinessLevel.DEPLOYABLE,
            required_level=ReadinessLevel.DEPLOYABLE,
        )
        assert report.passed is True

    def test_passed_false_when_below_required(self) -> None:
        report = ReadinessReport(
            archetype=CLI,
            achieved_level=ReadinessLevel.PROTOTYPE,
            required_level=ReadinessLevel.DEPLOYABLE,
        )
        assert report.passed is False

    def test_blocking_violations_filters_to_violated_only(self) -> None:
        outcomes = (
            _outcome("a", ProbeResult.SATISFIED),
            _outcome("b", ProbeResult.VIOLATED),
            _outcome("c", ProbeResult.NOT_APPLICABLE),
            _outcome("d", ProbeResult.INDETERMINATE),
        )
        report = ReadinessReport(
            archetype=CLI,
            achieved_level=ReadinessLevel.PROTOTYPE,
            required_level=ReadinessLevel.DEPLOYABLE,
            outcomes=outcomes,
        )
        assert [o.requirement_id for o in report.blocking_violations] == ["b"]


@pytest.mark.unit
def test_evidence_is_frozen_value_object() -> None:
    evidence = Evidence(probe="p", detail="d", location="file.py:1")
    with pytest.raises(Exception):
        evidence.detail = "changed"  # type: ignore[misc]
