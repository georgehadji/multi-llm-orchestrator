"""Refinement acceptance chain (Phase 6, E-10/E-11).

Chain of Responsibility — every rule runs and reports, so a rejected
candidate yields a full diagnosis rather than a single first failure
(plan §3.4.2, §3.4.3 invariants 3-5). Pure functions: no I/O.

Rules, in order:
1. green        — the SAME suite (hash-locked, immutable) still passes
2. ratchet      — the targeted metric strictly improved (lower is better)
3. no-regress   — no other tracked metric regressed beyond tolerance
4. mutation-held— the suite's mutation score did not drop
5. no-findings  — no new lint/type/security finding appeared
6. api-surface  — the public API surface is unchanged (freeze)
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.refinement import (
    AcceptanceVerdict as AcceptanceVerdict,
    MetricSnapshot,
    RefinementCandidate,
    metric_improved,
    no_metric_regressed,
)


@dataclass(frozen=True)
class AcceptanceContext:
    """Everything the acceptance chain needs to judge one candidate."""

    suite_passed: bool
    before: MetricSnapshot
    after: MetricSnapshot
    candidate: RefinementCandidate
    mutation_before: float | None = None
    mutation_after: float | None = None
    new_findings: bool = False
    api_surface_changed: bool = False
    ratchet_tolerance: float = 0.0


def evaluate_acceptance(ctx: AcceptanceContext) -> AcceptanceVerdict:
    """Run all six rules and aggregate the verdict.

    Args:
        ctx: Pre-computed candidate context.

    Returns:
        An AcceptanceVerdict with one result per rule.
    """
    results: list[tuple[str, bool, str]] = []

    # 1. green
    results.append(("suite_green", ctx.suite_passed, "" if ctx.suite_passed else "suite not green"))

    # 2. ratchet — targeted metric must strictly improve
    improved = metric_improved(ctx.before, ctx.after, ctx.candidate.predicted_metric)
    results.append(
        (
            "target_improved",
            improved,
            "" if improved else f"{ctx.candidate.predicted_metric} did not improve",
        )
    )

    # 3. no-regress — all other tracked metrics within tolerance
    no_regress = no_metric_regressed(ctx.before, ctx.after, ctx.ratchet_tolerance)
    results.append(("no_metric_regressed", no_regress, "" if no_regress else "a metric regressed"))

    # 4. mutation-held — score must not drop
    mutation_held = True
    mutation_reason = ""
    if ctx.mutation_before is not None and ctx.mutation_after is not None:
        mutation_held = ctx.mutation_after >= ctx.mutation_before - 1e-9
        mutation_reason = "" if mutation_held else "mutation score dropped"
    results.append(("mutation_held", mutation_held, mutation_reason))

    # 5. no new findings
    results.append(
        ("no_new_findings", not ctx.new_findings, "" if not ctx.new_findings else "new finding(s)")
    )

    # 6. api-surface freeze
    results.append(
        (
            "api_surface_frozen",
            not ctx.api_surface_changed,
            "" if not ctx.api_surface_changed else "public API surface changed",
        )
    )

    accepted = all(ok for _, ok, _ in results)
    return AcceptanceVerdict(accepted=accepted, rule_results=tuple(results))
