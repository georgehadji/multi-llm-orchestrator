"""Performance operator — benchmark-gated structural tier (E-12).

Inert unless the workspace ships a benchmark: detection happens inside
``propose()`` (workspace I/O, so it cannot live in the pure ``applicable``
gate), and a missing or failed benchmark yields zero candidates and zero
LLM calls (plan §E-12: "guessing at optimizations without measurement is
how correctness bugs enter under the banner of performance").

The acceptance chain's noise floor is fixed at the plan's stated default
(5%, ``domain.refinement._BENCHMARK_NOISE_FLOOR``); this operator adds an
*upstream* filter derived from observed variance — it refuses to target a
benchmark whose own two-run spread already exceeds ``MAX_BASELINE_CV``,
because no rewrite could be trusted to show a real improvement over noise
that large.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from ....domain.refinement import (
    MetricSnapshot,
    RefinementCandidate,
    RefinementTier,
    coefficient_of_variation,
)
from ....domain.ports import BenchmarkPort
from ....domain.testing_models import Workspace
from .llm_refactor import LLMRefactorCommand, RefactorClientPort, propose_llm_candidate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert Python engineer performing a targeted, behavior-preserving "
    "performance optimization. Improve the runtime of the hot path exercised by the "
    "named benchmark. Do not change any function's public signature, do not change "
    "any other function's public API, and do not alter observable behavior. Output "
    "the complete file only."
)


class PerformanceOperator:
    """Strategy operator: proposes an optimization for the workspace's hottest benchmark.

    Benchmark existence and the two-run noise estimate are workspace I/O,
    so — unlike the other structural operators — ``applicable()`` is
    unconditionally True here; the real gate runs inside ``propose()``.
    This mirrors the mechanical operators' own "always applicable, cost
    decided downstream" shape rather than inventing a second convention.
    """

    name = "performance"
    tier = RefinementTier.PERFORMANCE

    #: A benchmark whose two-run coefficient of variation exceeds this is
    #: too noisy to trust — skip it rather than chase measurement jitter.
    MAX_BASELINE_CV = 0.20

    def __init__(self, client: RefactorClientPort, model: object, benchmark: BenchmarkPort) -> None:
        """Initialize with the injected LLM client, model, and BenchmarkPort."""
        self._client = client
        self._model = model
        self._benchmark = benchmark

    def applicable(self, snapshot: MetricSnapshot) -> bool:
        """Always True — benchmark detection is workspace I/O (see class docstring)."""
        return True

    def command_for(self, candidate: RefinementCandidate) -> LLMRefactorCommand | None:
        """Build the Command from the candidate's pre-generated payload."""
        if candidate.operator != self.name or not candidate.payload:
            return None
        return LLMRefactorCommand(candidate.target_file, candidate.payload)

    async def propose(
        self, workspace: Workspace, snapshot: MetricSnapshot
    ) -> list[RefinementCandidate]:
        """Measure twice, pick the hottest low-noise benchmark, propose one optimization."""
        baseline_a = await self._benchmark.run(workspace)
        if not baseline_a:
            return []  # no benchmark, or measurement failed — zero cost
        baseline_b = await self._benchmark.run(workspace)
        if not baseline_b:
            return []

        target = self._pick_target(workspace, baseline_a, baseline_b)
        if target is None:
            return []
        bench_name, source_file = target
        median_ns = baseline_a[bench_name]

        try:
            source = source_file.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - best-effort
            return []
        rel = str(source_file.relative_to(workspace.root))

        instruction = (
            f"Benchmark `{bench_name}` currently measures {median_ns:,.0f} ns (median). "
            "Optimize this file's hot path for lower latency."
        )
        return await propose_llm_candidate(
            client=self._client,
            model=self._model,
            operator_name=self.name,
            tier=self.tier,
            target_file=rel,
            source=source,
            predicted_metric=f"benchmark_ns:{bench_name}",
            rationale=f"optimize hot path for `{bench_name}` ({median_ns:,.0f} ns median)",
            instruction=instruction,
            system=_SYSTEM_PROMPT,
        )

    def _pick_target(
        self,
        workspace: Workspace,
        baseline_a: dict[str, float],
        baseline_b: dict[str, float],
    ) -> tuple[str, Path] | None:
        """Pick the hottest benchmark whose baseline noise is trustworthy.

        Considered in descending order of measured cost so the biggest win
        is attempted first; skipped when noisy or when its source cannot
        be resolved.
        """
        common = sorted(
            set(baseline_a) & set(baseline_b), key=lambda name: baseline_a[name], reverse=True
        )
        for name in common:
            cv = coefficient_of_variation((baseline_a[name], baseline_b[name]))
            if cv > self.MAX_BASELINE_CV:
                continue
            source_file = self._resolve_source_file(workspace, name)
            if source_file is not None:
                return name, source_file
        return None

    @staticmethod
    def _resolve_source_file(workspace: Workspace, bench_name: str) -> Path | None:
        """Best-effort: the first workspace-local module the benchmark test imports.

        ``bench_name`` looks like ``tests/benchmarks/test_foo.py::test_bench_x``.
        Returns None when the test file is missing, unparseable, or imports
        nothing resolvable inside the workspace — the operator skips rather
        than guessing at a target.
        """
        test_path = bench_name.split("::", 1)[0]
        test_file = workspace.root / test_path
        if not test_file.exists():
            return None
        try:
            tree = ast.parse(test_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return None

        for node in ast.walk(tree):
            module: str | None = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import) and node.names:
                module = node.names[0].name
            if not module:
                continue
            candidate = workspace.root / (module.replace(".", "/") + ".py")
            if candidate.exists() and "test" not in candidate.name.lower():
                return candidate
        return None
