"""
Tests for the refinement domain model (E-9) — pure functions, no I/O.
=====================================================================
Total-function coverage: every metric field pair, including None
benchmark fields, must be comparable without crashing.
"""

from __future__ import annotations

import pytest

from orchestrator.domain.refinement import (
    AcceptanceVerdict,
    MetricSnapshot,
    RefinementCandidate,
    RefinementTier,
    metric_improved,
    no_metric_regressed,
)


@pytest.mark.unit
class TestMetricImproved:
    """Targeted metric strictly improves (lower is better)."""

    def _snap(self, **overrides) -> MetricSnapshot:
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

    def test_improvement_true(self) -> None:
        before = self._snap(cyclomatic_max=12)
        after = self._snap(cyclomatic_max=9)
        assert metric_improved(before, after, "cyclomatic_max") is True

    def test_equal_is_not_improvement(self) -> None:
        before = self._snap(cyclomatic_max=12)
        after = self._snap(cyclomatic_max=12)
        assert metric_improved(before, after, "cyclomatic_max") is False

    def test_regression_is_false(self) -> None:
        before = self._snap(max_nesting_depth=3)
        after = self._snap(max_nesting_depth=5)
        assert metric_improved(before, after, "max_nesting_depth") is False

    def test_none_fields_never_improve(self) -> None:
        before = self._snap(dead_symbols=2)
        after = self._snap(dead_symbols=None)  # type: ignore[arg-type]
        assert metric_improved(before, after, "dead_symbols") is False
        before_none = self._snap(dead_symbols=None)  # type: ignore[arg-type]
        assert metric_improved(before_none, self._snap(dead_symbols=1), "dead_symbols") is False

    def test_unknown_field_raises(self) -> None:
        before = self._snap()
        with pytest.raises(ValueError):
            metric_improved(before, before, "not_a_metric")

    def test_total_functions_cover_all_tracked_fields(self) -> None:
        """Every tracked field is comparable in both directions (total)."""
        fields = [
            "cyclomatic_mean",
            "cyclomatic_max",
            "max_nesting_depth",
            "longest_function_lines",
            "duplicated_blocks",
            "dead_symbols",
            "total_lines",
        ]
        better = self._snap(
            cyclomatic_mean=3.0,
            cyclomatic_max=8,
            max_nesting_depth=2,
            longest_function_lines=30,
            duplicated_blocks=1,
            dead_symbols=1,
            total_lines=400,
        )
        for f in fields:
            assert metric_improved(self._snap(), better, f) is True, f
            assert no_metric_regressed(self._snap(), better) is True


@pytest.mark.unit
class TestNoMetricRegressed:
    """No non-targeted metric may regress beyond tolerance."""

    def _snap(self, **overrides) -> MetricSnapshot:
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

    def test_identical_snapshots_pass(self) -> None:
        s = self._snap()
        assert no_metric_regressed(s, s) is True

    def test_improvement_passes(self) -> None:
        assert no_metric_regressed(self._snap(), self._snap(cyclomatic_max=9)) is True

    def test_regression_fails(self) -> None:
        assert no_metric_regressed(self._snap(), self._snap(total_lines=600)) is False

    def test_tolerance_allows_small_regression(self) -> None:
        before = self._snap(cyclomatic_max=12)
        after = self._snap(cyclomatic_max=13)
        assert no_metric_regressed(before, after, tolerance=1.0) is True
        assert no_metric_regressed(before, after, tolerance=0.0) is False

    def test_none_fields_ignored(self) -> None:
        before = self._snap(dead_symbols=None)  # type: ignore[arg-type]
        after = self._snap(dead_symbols=5)  # became measurable — neutral
        assert no_metric_regressed(before, after) is True

    def test_benchmark_none_fields_comparable(self) -> None:
        before = self._snap(benchmark_ns=None)
        after = self._snap(benchmark_ns={"bench_x": 1.5})
        assert no_metric_regressed(before, after) is True  # benchmark not tracked


@pytest.mark.unit
class TestValueObjects:
    """Frozen dataclasses behave as value objects."""

    def test_snapshot_is_frozen(self) -> None:
        s = MetricSnapshot(
            cyclomatic_mean=1.0,
            cyclomatic_max=2,
            max_nesting_depth=1,
            longest_function_lines=5,
            duplicated_blocks=0,
        )
        with pytest.raises(Exception):
            s.cyclomatic_mean = 9.0  # type: ignore[misc]

    def test_candidate_is_frozen_and_tiered(self) -> None:
        c = RefinementCandidate(
            operator="extract_function",
            tier=RefinementTier.STRUCTURAL,
            target_file="svc.py",
            rationale="reduce nesting",
            diff="-",
            predicted_metric="max_nesting_depth",
        )
        assert c.tier is RefinementTier.STRUCTURAL
        assert c.predicted_metric == "max_nesting_depth"

    def test_acceptance_verdict_rejection_reasons(self) -> None:
        v = AcceptanceVerdict(
            accepted=False,
            rule_results=(("green", True, "ok"), ("ratchet", False, "cyclomatic_max regressed")),
        )
        assert v.rejection_reasons == ("cyclomatic_max regressed",)
