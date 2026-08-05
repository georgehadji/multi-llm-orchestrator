"""Refinement domain model — pure data and decision functions (Phase 6).

Functional core of green-anchored refinement: frozen value objects plus
total, side-effect-free decision functions. No I/O, no subprocess, no
clock (Contract 1 — stdlib only). Every acceptance rule is a pure function
so accept/reject logic is unit-testable with no filesystem and auditable
by construction.

Invariants (plan §3.4.3):
* Every tracked metric is *lower-is-better*.
* A candidate is accepted only if the targeted metric strictly improves
  AND no other tracked metric regresses beyond tolerance.
* ``None`` benchmark fields are comparable-total: missing measurements
  never crash a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: All MetricSnapshot fields that participate in ratchet comparisons.
#: Lower is better for every one of them.
_TRACKED_METRICS = (
    "cyclomatic_mean",
    "cyclomatic_max",
    "max_nesting_depth",
    "longest_function_lines",
    "duplicated_blocks",
    "dead_symbols",
    "total_lines",
)


class RefinementTier(str, Enum):
    """Staged refinement tiers (plan §3.4.4)."""

    MECHANICAL = "mechanical"  # deterministic, no LLM
    STRUCTURAL = "structural"  # LLM-proposed, suite-verified
    PERFORMANCE = "performance"  # benchmark-gated


@dataclass(frozen=True)
class MetricSnapshot:
    """One measurement of a workspace. Comparable, not mutable."""

    cyclomatic_mean: float
    cyclomatic_max: int
    max_nesting_depth: int
    longest_function_lines: int
    duplicated_blocks: int
    dead_symbols: int = 0
    total_lines: int = 0
    bundle_bytes: int | None = None
    benchmark_ns: dict[str, float] | None = None

    def value_of(self, field_name: str) -> float | int | None:
        """Read a metric field by name (None-safe)."""
        value = getattr(self, field_name, None)
        if isinstance(value, (float, int)):
            return value
        return None


@dataclass(frozen=True)
class RefinementCandidate:
    """A single proposed change. Immutable description; a Command applies it."""

    operator: str  # "extract_function", "dead_code", ...
    tier: RefinementTier
    target_file: str
    rationale: str
    diff: str
    predicted_metric: str  # which MetricSnapshot field this must improve


@dataclass(frozen=True)
class AcceptanceVerdict:
    """Outcome of the acceptance chain (Chain of Responsibility)."""

    accepted: bool
    rule_results: tuple[tuple[str, bool, str], ...] = ()  # (rule, passed, reason)

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        """Reasons from every rule that failed (full diagnosis, not first-fail)."""
        return tuple(reason for _, ok, reason in self.rule_results if not ok)


@dataclass(frozen=True)
class RefinementOutcome:
    """Result of one refinement pass over a workspace."""

    tier: RefinementTier
    accepted_candidates: int = 0
    rejected_candidates: int = 0
    model_calls: int = 0
    started: bool = False
    reason: str = ""


# ── pure decision core — no I/O, total functions ────────────────────────────


def metric_improved(before: MetricSnapshot, after: MetricSnapshot, field: str) -> bool:
    """Return True when *field* strictly improved (lower is better).

    A field that is ``None`` in either snapshot is *not* an improvement
    (missing measurement cannot license restructuring) unless it became
    measurable: None -> a value is treated as improved only when the new
    value is strictly lower than the old one would have been — which we
    cannot know, so None in either side returns False.

    Args:
        before: Baseline snapshot.
        after: Post-candidate snapshot.
        field: MetricSnapshot field name (must be a tracked metric).

    Returns:
        True iff the metric strictly decreased.
    """
    if field not in _TRACKED_METRICS:
        raise ValueError(f"'{field}' is not a tracked refinement metric")
    before_val = getattr(before, field, None)
    after_val = getattr(after, field, None)
    if before_val is None or after_val is None:
        return False
    if isinstance(before_val, bool) or isinstance(after_val, bool):
        return False
    return float(after_val) < float(before_val)


def no_metric_regressed(
    before: MetricSnapshot, after: MetricSnapshot, tolerance: float = 0.0
) -> bool:
    """Return True when no *other* tracked metric got worse beyond tolerance.

    ``None`` fields are ignored (missing measurements cannot regress);
    a metric that became measurable from None is treated as neutral.

    Args:
        before: Baseline snapshot.
        after: Post-candidate snapshot.
        tolerance: Allowed absolute regression per metric (ratchet floor).

    Returns:
        True iff every tracked metric stayed within tolerance (or improved).
    """
    for metric_field in _TRACKED_METRICS:
        before_val = getattr(before, metric_field, None)
        after_val = getattr(after, metric_field, None)
        if before_val is None or after_val is None:
            continue
        if float(after_val) > float(before_val) + tolerance:
            return False
    return True


__all__ = [
    "AcceptanceVerdict",
    "MetricSnapshot",
    "RefinementCandidate",
    "RefinementOutcome",
    "RefinementTier",
    "metric_improved",
    "no_metric_regressed",
]
