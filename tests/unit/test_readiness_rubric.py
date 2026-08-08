"""Tests for the Rubric registry and the archetype adapter (P-1)."""

from __future__ import annotations

import pytest

from orchestrator.application.readiness.archetype import archetype_for
from orchestrator.application.readiness.rubric import Rubric
from orchestrator.domain.readiness import (
    AppArchetype,
    ProbeResult,
    ReadinessLevel,
    Requirement,
    RequirementOutcome,
)


def _req(id: str, level: ReadinessLevel, archetypes) -> Requirement:
    return Requirement(
        id=id,
        title=id,
        derives_from="fixture",
        level=level,
        archetypes=frozenset(archetypes),
    )


@pytest.mark.unit
class TestRubric:
    def test_register_and_list(self) -> None:
        rubric = Rubric()
        req = _req("a", ReadinessLevel.DEPLOYABLE, {AppArchetype.PYTHON_CLI})
        rubric.register(req)
        assert rubric.requirements == (req,)

    def test_applicable_filters_by_archetype(self) -> None:
        rubric = Rubric()
        rubric.register(_req("cli-only", ReadinessLevel.DEPLOYABLE, {AppArchetype.PYTHON_CLI}))
        rubric.register(
            _req("service-only", ReadinessLevel.OPERABLE, {AppArchetype.PYTHON_SERVICE})
        )
        applicable = rubric.applicable(AppArchetype.PYTHON_CLI)
        assert [r.id for r in applicable] == ["cli-only"]

    def test_achieved_level_delegates_to_domain_rollup(self) -> None:
        rubric = Rubric()
        rubric.register(_req("l1", ReadinessLevel.DEPLOYABLE, {AppArchetype.PYTHON_CLI}))
        outcomes = (RequirementOutcome(requirement_id="l1", result=ProbeResult.SATISFIED),)
        assert rubric.achieved_level(outcomes, AppArchetype.PYTHON_CLI) == ReadinessLevel.DEPLOYABLE

    def test_explain_renders_a_tree_grouped_by_level(self) -> None:
        rubric = Rubric()
        rubric.register(_req("l1-req", ReadinessLevel.DEPLOYABLE, {AppArchetype.PYTHON_CLI}))
        tree = rubric.explain()
        assert "DEPLOYABLE" in tree
        assert "l1-req" in tree

    def test_every_registered_requirement_has_non_empty_derives_from(self) -> None:
        rubric = Rubric()
        rubric.register(_req("a", ReadinessLevel.DEPLOYABLE, {AppArchetype.PYTHON_CLI}))
        assert all(r.derives_from for r in rubric.requirements)


@pytest.mark.unit
class TestArchetypeAdapter:
    @pytest.mark.parametrize(
        "app_type,expected",
        [
            ("fastapi", AppArchetype.PYTHON_SERVICE),
            ("flask", AppArchetype.PYTHON_SERVICE),
            ("react-fastapi", AppArchetype.FULLSTACK),
            ("nextjs", AppArchetype.FULLSTACK),
            ("cli", AppArchetype.PYTHON_CLI),
            ("script", AppArchetype.PYTHON_CLI),
            ("generic", AppArchetype.PYTHON_CLI),
            ("library", AppArchetype.LIBRARY),
        ],
    )
    def test_known_app_types_map_correctly(self, app_type: str, expected: AppArchetype) -> None:
        assert archetype_for(app_type) == expected

    def test_unknown_app_type_falls_back_to_cli(self) -> None:
        assert archetype_for("some-unknown-type") == AppArchetype.PYTHON_CLI
