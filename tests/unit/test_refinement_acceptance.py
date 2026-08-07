"""
Tests for the refinement acceptance chain (E-11) — six rules.
==============================================================
Each rule in isolation, plus the composition: suite green but a metric
regressed => reject; API rename => reject; mutation drop => reject.
"""

from __future__ import annotations

import pytest

from orchestrator.application.refinement.acceptance import (
    AcceptanceContext,
    evaluate_acceptance,
)
from orchestrator.domain.refinement import (
    MetricSnapshot,
    RefinementCandidate,
    RefinementTier,
)


def _snap(**overrides) -> MetricSnapshot:
    base = {
        "cyclomatic_mean": 5.0,
        "cyclomatic_max": 12,
        "max_nesting_depth": 4,
        "longest_function_lines": 60,
        "duplicated_blocks": 3,
        "dead_symbols": 2,
        "total_lines": 500,
    }
    base.update(overrides)
    return MetricSnapshot(**base)


def _candidate(predicted_metric: str = "cyclomatic_max", operator: str = "extract_function"):
    return RefinementCandidate(
        operator=operator,
        tier=RefinementTier.STRUCTURAL,
        target_file="svc.py",
        rationale="test",
        diff="-",
        predicted_metric=predicted_metric,
    )


def _ctx(**overrides) -> AcceptanceContext:
    base = {
        "suite_passed": True,
        "before": _snap(),
        "after": _snap(cyclomatic_max=9),
        "candidate": _candidate(),
        "mutation_before": 0.7,
        "mutation_after": 0.7,
        "new_findings": False,
        "api_surface_changed": False,
    }
    base.update(overrides)
    return AcceptanceContext(**base)


@pytest.mark.unit
class TestAcceptanceRules:
    """Each rule in isolation."""

    def test_all_green_accepts(self) -> None:
        verdict = evaluate_acceptance(_ctx())
        assert verdict.accepted is True
        assert len(verdict.rule_results) == 6
        assert verdict.rejection_reasons == ()

    def test_suite_not_green_rejects(self) -> None:
        verdict = evaluate_acceptance(_ctx(suite_passed=False))
        assert verdict.accepted is False
        assert "suite not green" in verdict.rejection_reasons[0]

    def test_target_metric_not_improved_rejects(self) -> None:
        verdict = evaluate_acceptance(_ctx(after=_snap(cyclomatic_max=12)))  # equal
        assert verdict.accepted is False
        assert any("did not improve" in r for r in verdict.rejection_reasons)

    def test_other_metric_regressed_rejects(self) -> None:
        # targeted metric improved, but total_lines regressed
        verdict = evaluate_acceptance(_ctx(after=_snap(cyclomatic_max=9, total_lines=700)))
        assert verdict.accepted is False
        assert any("regressed" in r for r in verdict.rejection_reasons)

    def test_mutation_drop_rejects(self) -> None:
        verdict = evaluate_acceptance(_ctx(mutation_after=0.5))
        assert verdict.accepted is False
        assert any("mutation" in r for r in verdict.rejection_reasons)

    def test_new_findings_reject(self) -> None:
        verdict = evaluate_acceptance(_ctx(new_findings=True))
        assert verdict.accepted is False
        assert any("finding" in r for r in verdict.rejection_reasons)

    def test_api_surface_change_rejects(self) -> None:
        verdict = evaluate_acceptance(_ctx(api_surface_changed=True))
        assert verdict.accepted is False
        assert any("API" in r for r in verdict.rejection_reasons)

    def test_rejection_reasons_give_full_diagnosis(self) -> None:
        verdict = evaluate_acceptance(_ctx(suite_passed=False, new_findings=True))
        assert len(verdict.rejection_reasons) == 2  # not first-fail

    def test_none_mutation_fields_neutral(self) -> None:
        verdict = evaluate_acceptance(_ctx(mutation_before=None, mutation_after=None))
        assert verdict.accepted is True  # rule 4 skipped when unmeasured
