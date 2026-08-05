"""Metric collectors — registry and concurrent fan-out (E-9).

Collects a :class:`MetricSnapshot` for a workspace from three sources:

1. **AST collector** — cyclomatic mean/max, nesting depth, longest
   function, duplicated blocks, total lines (four visitors, one parse).
2. **StaticAnalyzer adapter** — reuses the existing quality analyzer's
   per-file complexity (the repo's established measurement) and merges it.
3. **Vulture adapter** — dead-symbol count, gracefully None when absent.

Collectors run concurrently and merge into one snapshot.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ...domain.refinement import MetricSnapshot
from ...domain.testing_models import Workspace
from . import ast_metrics as _ast

logger = logging.getLogger(__name__)


@dataclass
class MetricCollector:
    """One collector callable plus a label (for diagnostics)."""

    name: str
    collect: Callable[[Workspace], object]


def _collect_ast(workspace: Workspace) -> dict[str, float | int]:
    """Merge per-file AST measurements into aggregate snapshot fields."""
    aggregated = {
        "cyclomatic_mean": 0.0,
        "cyclomatic_max": 1,
        "max_nesting_depth": 0,
        "longest_function_lines": 0,
        "duplicated_blocks": 0,
        "total_lines": 0,
    }
    files = 0
    for py_file in sorted(workspace.root.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - best-effort
            continue
        fields = _ast.snapshot_fields(source)
        aggregated["cyclomatic_mean"] += fields["cyclomatic_mean"]
        aggregated["cyclomatic_max"] = max(aggregated["cyclomatic_max"], fields["cyclomatic_max"])
        aggregated["max_nesting_depth"] = max(
            aggregated["max_nesting_depth"], fields["max_nesting_depth"]
        )
        aggregated["longest_function_lines"] = max(
            aggregated["longest_function_lines"], fields["longest_function_lines"]
        )
        aggregated["duplicated_blocks"] += fields["duplicated_blocks"]
        aggregated["total_lines"] += fields["total_lines"]
        files += 1
    if files:
        aggregated["cyclomatic_mean"] = aggregated["cyclomatic_mean"] / files
    return aggregated


def _collect_static_complexity(workspace: Workspace) -> dict[str, float]:
    """Reuse StaticAnalyzer's complexity measurement (E-9 reuse requirement).

    The existing StaticAnalyzer computes per-file cyclomatic complexity via
    ``quality/quality_control.py``; we map its ``complexity_score`` mean into
    the snapshot so the orchestrator's established metric remains the
    authority rather than a second re-derived value.
    """
    try:
        from ...quality.quality_control import StaticAnalyzer
    except Exception:  # pragma: no cover - analyzer import is stable
        return {}
    analyzer = StaticAnalyzer()
    values: list[float] = []
    for py_file in sorted(workspace.root.rglob("*.py")):
        if "__pycache__" in str(py_file) or "test" in py_file.name.lower():
            continue
        try:
            metrics = asyncio.run(analyzer.analyze_file(py_file))
        except Exception:  # pragma: no cover - best-effort per file
            continue
        values.append(metrics.complexity_score)
    if not values:
        return {}
    return {"cyclomatic_mean": sum(values) / len(values)}


async def collect_snapshot(
    workspace: Workspace,
    *,
    use_static: bool = True,
    extra_collectors: tuple[MetricCollector, ...] = (),
) -> MetricSnapshot:
    """Collect a MetricSnapshot for a workspace (concurrent fan-out).

    Args:
        workspace: The materialized workspace to measure.
        use_static: Merge StaticAnalyzer's complexity mean (default True).
        extra_collectors: Additional collectors wired by the composition
            root (e.g. bundle-size for web projects) — never a module-level
            singleton (plan §3.4.2).

    Returns:
        A frozen MetricSnapshot with all measured fields.
    """
    ast_fields = await asyncio.to_thread(_collect_ast, workspace)
    static_fields: dict[str, float] = {}
    if use_static:
        static_fields = await asyncio.to_thread(_collect_static_complexity, workspace)
    if static_fields.get("cyclomatic_mean") is not None:
        ast_fields["cyclomatic_mean"] = static_fields["cyclomatic_mean"]

    from .dead_code import count_dead_symbols

    dead_symbols = await count_dead_symbols(workspace.root)

    # Extra collectors registered by the composition root (no globals).
    for collector in extra_collectors:
        try:
            extra = collector.collect(workspace)
            if isinstance(extra, dict):
                ast_fields.update(extra)
        except Exception as exc:  # never block measurement on one collector
            logger.warning("metric collector %s failed: %s", collector.name, exc)

    return MetricSnapshot(
        cyclomatic_mean=float(ast_fields.get("cyclomatic_mean", 1.0)),
        cyclomatic_max=int(ast_fields.get("cyclomatic_max", 1)),
        max_nesting_depth=int(ast_fields.get("max_nesting_depth", 0)),
        longest_function_lines=int(ast_fields.get("longest_function_lines", 0)),
        duplicated_blocks=int(ast_fields.get("duplicated_blocks", 0)),
        dead_symbols=dead_symbols if dead_symbols is not None else 0,
        total_lines=int(ast_fields.get("total_lines", 0)),
        bundle_bytes=ast_fields.get("bundle_bytes"),
        benchmark_ns=None,
    )


def bundle_size_collector(extra_patterns: tuple[str, ...] = ()) -> MetricCollector:
    """Registry collector for web bundle sizes (E-12 bundle_bytes)."""

    def _collect(workspace: Workspace) -> dict[str, int | None]:
        total = 0
        found = False
        patterns = ("dist", "build") + extra_patterns
        for pattern in patterns:
            target = workspace.root / pattern
            if target.is_dir():
                for p in target.rglob("*"):
                    if p.is_file():
                        total += p.stat().st_size
                        found = True
            elif target.is_file():
                total += target.stat().st_size
                found = True
        return {"bundle_bytes": total if found else None}

    return MetricCollector(name="bundle_size", collect=_collect)


__all__ = [
    "MetricCollector",
    "MetricSnapshot",
    "bundle_size_collector",
    "collect_snapshot",
]
