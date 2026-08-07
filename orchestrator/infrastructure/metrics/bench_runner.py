"""BenchmarkPort adapter — pytest-benchmark (E-12).

The performance tier is inert unless the workspace ships a benchmark:
detection is a filesystem check that runs before any subprocess is
spawned, so unbenchmarked projects pay zero cost for this module. When a
benchmark exists, a single ``pytest --benchmark-only`` invocation supplies
statistically rigorous medians (pytest-benchmark's own ``min_rounds``),
avoiding the need to orchestrate repeated subprocess runs by hand.

Median, not mean, matching the repo's existing outlier-resistant
aggregation choice in the evaluator (docs/... aggregate() decision).
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path

from ...domain.testing_models import Workspace

logger = logging.getLogger(__name__)

_BENCHMARK_DIR_NAMES = ("benchmarks", "benchmark")


def has_benchmarks(workspace: Workspace) -> bool:
    """Cheap, filesystem-only detection — no subprocess.

    A workspace "ships a benchmark" when it has a ``tests/benchmarks/`` (or
    ``tests/benchmark/``) directory with test files, or any ``test_*bench*.py``
    file under ``tests/``.
    """
    root = workspace.root
    for name in _BENCHMARK_DIR_NAMES:
        candidate = root / "tests" / name
        if candidate.is_dir() and any(candidate.glob("test_*.py")):
            return True
    tests_dir = root / "tests"
    if tests_dir.is_dir():
        for _ in tests_dir.rglob("test_*bench*.py"):
            return True
    return False


class PytestBenchmarkRunner:
    """BenchmarkPort adapter over pytest-benchmark's JSON export."""

    def __init__(self, *, min_rounds: int = 5) -> None:
        """Initialize.

        Args:
            min_rounds: Minimum measured rounds pytest-benchmark must run
                per benchmark (statistical floor, plan default 5).
        """
        self._min_rounds = min_rounds

    async def run(self, workspace: Workspace, *, timeout_s: float = 300.0) -> dict[str, float]:
        """Run the workspace's benchmark suite; return name -> median nanoseconds.

        Returns an empty dict — never raises — when no benchmark exists,
        the run fails, or the report cannot be parsed. A benchmark
        subprocess failure must never abort a refinement pass.
        """
        if not has_benchmarks(workspace):
            return {}
        target = self._benchmark_target(workspace)
        if target is None:
            return {}

        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "bench.json"
            argv = [
                "python",
                "-m",
                "pytest",
                str(target),
                "-q",
                "--benchmark-only",
                f"--benchmark-json={report_path}",
                f"--benchmark-min-rounds={self._min_rounds}",
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=str(workspace.root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    logger.warning("benchmark run timed out after %.0fs", timeout_s)
                    return {}
            except OSError as exc:  # pragma: no cover - pytest/python missing
                logger.warning("benchmark run failed to start: %s", exc)
                return {}

            if not report_path.exists():
                return {}
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("benchmark report unparseable: %s", exc)
                return {}

        return self._extract_medians(data)

    @staticmethod
    def _extract_medians(data: dict) -> dict[str, float]:
        results: dict[str, float] = {}
        for bench in data.get("benchmarks", []):
            name = bench.get("fullname") or bench.get("name")
            stats = bench.get("stats") or {}
            median_s = stats.get("median")
            if name and median_s is not None:
                results[str(name)] = float(median_s) * 1e9  # seconds -> ns
        return results

    @staticmethod
    def _benchmark_target(workspace: Workspace) -> Path | None:
        for name in _BENCHMARK_DIR_NAMES:
            candidate = workspace.root / "tests" / name
            if candidate.is_dir():
                return candidate
        tests_dir = workspace.root / "tests"
        return tests_dir if tests_dir.is_dir() else None


__all__ = [
    "PytestBenchmarkRunner",
    "has_benchmarks",
]
