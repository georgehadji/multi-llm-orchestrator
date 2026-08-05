"""RefinementService — green-anchored refinement (Phase 6, E-10).

Imperative shell around the pure decision core (plan §3.4.1). Flow:

    entry gate (green ∧ mutation ≥ θ)  → fail: exit, receipt records why
    MEASURE                            → MetricSnapshot (deterministic, no LLM)
    no findings                        → exit (zero model calls)
    for each applicable operator       → propose → for each candidate:
        snapshot (Memento) → apply (Command) → re-run SAME suite
        → re-measure → acceptance chain → accept | revert

Entry gate is a HARD gate: refinement never runs against a suite that has
not passed the RED-gate and mutation threshold (plan §1.5, R-13).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from ...domain.refinement import (
    MetricSnapshot,
    RefinementCandidate,
    RefinementOutcome,
    RefinementTier,
)
from ...domain.ports import TestExecutorPort
from ...domain.testing_models import SuiteReport, Workspace
from .acceptance import AcceptanceContext, AcceptanceVerdict, evaluate_acceptance
from .api_surface import api_surface_equal, extract_api_surface
from .ledger import RefinementCommand, RefinementLedger
from .operators.base import RefinementOperator

logger = logging.getLogger(__name__)


@dataclass
class RefinementReceipt:
    """Structured outcome of one refinement pass (observability, E-8)."""

    tier: RefinementTier
    started: bool
    reason: str = ""
    accepted: int = 0
    rejected: int = 0
    model_calls: int = 0
    before: MetricSnapshot | None = None
    after: MetricSnapshot | None = None
    rejection_reasons: list[str] = field(default_factory=list)


class RefinementService:
    """Orchestrates measure -> propose -> apply -> verify -> keep-or-revert."""

    def __init__(
        self,
        *,
        runner: "TestExecutorPort",
        materializer: object | None = None,
        collect_snapshot: "Callable[[Workspace], Awaitable[MetricSnapshot]] | None" = None,
        event_bus: object | None = None,
        mutation_provider: "Callable[[Workspace], Awaitable[float | None]] | None" = None,
        findings_provider: "Callable[[Workspace], Awaitable[bool]] | None" = None,
    ) -> None:
        """Initialize the service (dependencies injected by the composition root).

        Args:
            runner: TestExecutorPort — the SAME suite runner used for the
                entry gate and per-candidate verification.
            materializer: WorkspaceMaterializer (to materialize the suite
                alongside candidate changes if the workspace is not pre-built).
            collect_snapshot: async callable(workspace) -> MetricSnapshot.
            event_bus: Optional unified event bus (E-8 refinement events).
            mutation_provider: async callable(workspace, source_files) ->
                mutation score float | None, or None to skip rule 4.
            findings_provider: async callable(workspace) -> bool (any new
                lint/type/security finding), or None to skip rule 5.
        """
        self._runner = runner
        self._materializer = materializer
        self._collect = collect_snapshot
        self._event_bus = event_bus
        self._mutation_provider = mutation_provider
        self._findings_provider = findings_provider
        self._ledger = RefinementLedger()
        self._operators: list[RefinementOperator] = []

    def register_operator(self, operator: RefinementOperator) -> None:
        """Register a refinement operator (Strategy registry)."""
        self._operators.append(operator)

    @property
    def operators(self) -> list[RefinementOperator]:
        """Registered operators, in registration order."""
        return list(self._operators)

    # ── configuration ─────────────────────────────────────────────────────

    @staticmethod
    def _mode() -> str:
        return os.environ.get("ORCH_REFINE", "mechanical").lower()

    @staticmethod
    def _min_mutation() -> float:
        return float(os.environ.get("ORCH_REFINE_MIN_MUTATION", "0.6"))

    @staticmethod
    def _max_candidates() -> int:
        return int(os.environ.get("ORCH_REFINE_MAX_CANDIDATES", "8"))

    # ── main entry ────────────────────────────────────────────────────────

    async def refine(
        self,
        workspace: Workspace,
        suite_report: SuiteReport | None = None,
        *,
        mutation_score: float | None = None,
        source_files: tuple[Path, ...] | None = None,
        test_files: tuple[Path, ...] | None = None,
    ) -> RefinementReceipt:
        """Run one refinement pass over *workspace*.

        Args:
            workspace: The materialized workspace (or its root as a
                Workspace value).
            suite_report: The suite result that established green. When
                None, the entry gate is evaluated from a fresh run.
            mutation_score: The suite's measured mutation score.
            source_files: Optional explicit source files to measure/refine.
            test_files: Optional explicit test files (hash-locked, read-only).

        Returns:
            A RefinementReceipt describing the pass.
        """
        mode = self._mode()
        if mode == "off":
            return RefinementReceipt(
                tier=RefinementTier.MECHANICAL, started=False, reason="ORCH_REFINE=off"
            )

        tier = RefinementTier.MECHANICAL if mode == "mechanical" else RefinementTier.STRUCTURAL

        # ── Entry gate: green ∧ mutation ≥ θ (hard gate) ───────────────────
        gate_pass, gate_reason = await self._entry_gate(workspace, suite_report, mutation_score)
        if not gate_pass:
            return RefinementReceipt(tier=tier, started=False, reason=gate_reason, before=None)

        # ── Measure ────────────────────────────────────────────────────────
        before = await self._collect(workspace)
        if before is None:
            return RefinementReceipt(tier=tier, started=False, reason="measurement failed")

        # ── Operators ──────────────────────────────────────────────────────
        receipt = RefinementReceipt(tier=tier, started=True, before=before)
        model_calls = 0

        for operator in self._operators:
            if operator.tier.value != "mechanical" and mode != "full":
                # structural/performance tiers require ORCH_REFINE=structural/full
                if operator.tier.value in ("structural", "performance") and mode != "full":
                    continue
            if not operator.applicable(before):
                continue
            try:
                candidates = await operator.propose(workspace, before)
            except Exception as exc:  # never let one operator kill the pass
                logger.warning("operator %s propose failed: %s", operator.name, exc)
                continue

            for candidate in candidates[: self._max_candidates()]:
                status, verdict, changed = await self._apply_candidate(
                    workspace, candidate, before, source_files, test_files
                )
                if status == "accepted":
                    receipt.accepted += 1
                    before = await self._collect(workspace)  # new baseline
                elif status == "rejected":
                    receipt.rejected += 1
                    receipt.rejection_reasons.extend(verdict.rejection_reasons)
                # "noop" (zero changed files) is neither — zero cost, zero noise.
                model_calls += 0  # mechanical tier: zero LLM calls (asserted by tests)

        receipt.model_calls = model_calls
        receipt.after = before if receipt.accepted else None
        if receipt.accepted == 0 and receipt.rejected == 0:
            receipt.reason = "no applicable candidates"
        return receipt

    # ── entry gate ─────────────────────────────────────────────────────────

    async def _entry_gate(
        self,
        workspace: Workspace,
        suite_report: SuiteReport | None,
        mutation_score: float | None,
    ) -> tuple[bool, str]:
        """Green AND mutation >= theta. Never skips (R-13)."""
        if suite_report is not None and not suite_report.passed:
            return False, "suite not green — refinement requires a passing suite"
        if suite_report is None:
            # Evaluate green from a fresh run of the SAME suite.
            try:
                report = await self._runner.run(workspace, timeout_s=120.0)
            except Exception as exc:
                return False, f"entry-gate suite run failed: {exc}"
            if not report.passed:
                return False, "suite not green — refinement requires a passing suite"
        if mutation_score is None and self._mutation_provider is not None:
            try:
                mutation_score = await self._mutation_provider(workspace)
            except Exception as exc:  # pragma: no cover
                logger.warning("mutation provider failed at entry gate: %s", exc)
        if mutation_score is not None and mutation_score < self._min_mutation():
            return (
                False,
                f"mutation score {mutation_score:.2f} below ORCH_REFINE_MIN_MUTATION "
                f"({self._min_mutation():.2f})",
            )
        return True, ""

    # ── per-candidate apply / verify / revert ──────────────────────────────

    async def _apply_candidate(
        self,
        workspace: Workspace,
        candidate: RefinementCandidate,
        before: MetricSnapshot,
        source_files: tuple[Path, ...] | None,
        test_files: tuple[Path, ...] | None,
    ) -> tuple[str, AcceptanceVerdict | None, tuple[str, ...]]:
        """Apply one candidate; return (status, verdict, changed_files).

        Status is one of ``"accepted"``, ``"rejected"``, ``"noop"`` (the
        command changed no files — nothing to judge, zero cost).
        """
        command = self._command_for(candidate)
        if command is None:
            return "rejected", None, ()

        # Capture the PRE-command API surface (freeze check, rule 6).
        before_surface: dict[str, str] = {}
        if source_files:
            for rel in source_files:
                before_surface.update(self._file_surface(workspace.root / rel))

        # Memento before applying.
        entry = self._ledger.apply(command, workspace.root)

        if not entry.changed_files:
            # No-op: command made no edits (e.g. formatter on clean code).
            self._ledger.revert(workspace.root, entry)
            return "noop", None, ()

        try:
            # Re-run the SAME suite (tests hash-locked upstream).
            report = await self._runner.run(workspace, timeout_s=120.0)
            suite_passed = report.passed

            after = await self._collect(workspace)

            mutation_after = None
            if self._mutation_provider is not None:
                try:
                    mutation_after = await self._mutation_provider(workspace)
                except Exception:  # pragma: no cover
                    mutation_after = None

            new_findings = False
            if self._findings_provider is not None:
                try:
                    new_findings = await self._findings_provider(workspace)
                except Exception:  # pragma: no cover
                    new_findings = False

            api_changed = False
            if source_files:
                after_surface: dict[str, str] = {}
                for rel in source_files:
                    after_surface.update(self._file_surface(workspace.root / rel))
                api_changed = not api_surface_equal(before_surface, after_surface)

            ctx = AcceptanceContext(
                suite_passed=suite_passed,
                before=before,
                after=after,
                candidate=candidate,
                mutation_before=None,  # rule 4 uses the provider result only
                mutation_after=mutation_after,
                new_findings=new_findings,
                api_surface_changed=api_changed,
            )
            verdict = evaluate_acceptance(ctx)
        except Exception as exc:
            self._ledger.revert(workspace.root, entry)
            logger.warning("candidate %s failed: %s — reverted", candidate.operator, exc)
            return "rejected", None, ()

        if verdict.accepted:
            logger.info(
                "refinement accepted: %s on %s (%s)",
                candidate.operator,
                candidate.target_file,
                candidate.rationale,
            )
            return "accepted", verdict, entry.changed_files

        # Reject → restore the memento (byte-exact revert).
        self._ledger.revert(workspace.root, entry)
        logger.info(
            "refinement rejected: %s (%s)",
            candidate.operator,
            "; ".join(verdict.rejection_reasons) or "unknown",
        )
        return "rejected", verdict, ()

    def _command_for(self, candidate: RefinementCandidate) -> RefinementCommand | None:
        """Map a candidate to its reversible Command."""
        command: RefinementCommand | None = None
        for operator in self._operators:
            if operator.name != candidate.operator:
                continue
            if operator.name == "dead_code":
                from pathlib import Path as _Path

                from .operators.dead_code import RemoveUnusedImportsCommand

                command = RemoveUnusedImportsCommand([_Path(candidate.target_file)])
            elif operator.name == "formatter":
                from .operators.formatter import FormatCommand

                command = FormatCommand()
        return command

    @staticmethod
    def _file_surface(path: Path) -> dict[str, str]:
        try:
            return extract_api_surface(path.read_text(encoding="utf-8"))
        except OSError:  # pragma: no cover
            return {}
