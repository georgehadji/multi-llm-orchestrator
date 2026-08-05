"""AST metrics collector — four visitors over one parse (E-9).

Measures the metric fields that feed ``MetricSnapshot``:

* cyclomatic complexity (mean / max) — decision-point visitor
* maximum nesting depth — block-depth visitor
* longest function — source-line-span visitor
* duplicated blocks — normalized-subtree-hash visitor (same structure,
  different identifiers => detected as duplicates)

Pure AST, stdlib only. One parse per file, four visitors.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DECISION_NODES = (
    ast.If,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.IfExp,
    ast.ExceptHandler,
    ast.BoolOp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Match,
)

_BLOCK_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


@dataclass
class AstMetrics:
    """Per-file AST measurements."""

    cyclomatic_values: list[int] = field(default_factory=list)
    max_nesting_depth: int = 0
    longest_function_lines: int = 0
    duplicated_blocks: int = 0
    total_lines: int = 0

    @property
    def cyclomatic_mean(self) -> float:
        """Mean cyclomatic complexity across functions (1.0 when none)."""
        if not self.cyclomatic_values:
            return 1.0
        return sum(self.cyclomatic_values) / len(self.cyclomatic_values)

    @property
    def cyclomatic_max(self) -> int:
        """Worst-case cyclomatic complexity (1 when no functions)."""
        return max(self.cyclomatic_values) if self.cyclomatic_values else 1


class _CyclomaticVisitor(ast.NodeVisitor):
    """Count decision points inside one function (base 1 + branches)."""

    def __init__(self) -> None:
        self.count = 1  # base complexity

    def visit_If(self, node: ast.If) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # each `and`/`or` is an extra path
        self.count += len(node.values) - 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.count += len(node.cases)
        self.generic_visit(node)


class _NestingVisitor(ast.NodeVisitor):
    """Track maximum block-nesting depth."""

    def __init__(self) -> None:
        self.depth = 0
        self.max_depth = 0

    def _enter(self, node: ast.AST) -> None:
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.generic_visit(node)
        self.depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self._enter(node)

    def visit_For(self, node: ast.For) -> None:
        self._enter(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._enter(node)

    def visit_While(self, node: ast.While) -> None:
        self._enter(node)

    def visit_With(self, node: ast.With) -> None:
        self._enter(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._enter(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._enter(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._enter(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node)


class _NormalizingTransformer(ast.NodeTransformer):
    """Strip identifiers/constants so structurally-equal code hashes equal."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.Name(id="N", ctx=node.ctx)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        return ast.Attribute(value=self.visit(node.value), attr="A", ctx=node.ctx)

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return ast.Constant(value="C")

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return ast.arg(arg="a", annotation=self.visit(node.annotation) if node.annotation else None)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = "f"
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.name = "f"
        return self.generic_visit(node)


def _normalized_hash(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Hash of a function body with identifiers/constants normalized."""
    normalized = _NormalizingTransformer().visit(func_node)
    ast.fix_missing_locations(normalized)
    try:
        source = ast.unparse(normalized)
    except Exception:  # pragma: no cover - unparse is robust
        return ""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def analyze_ast(source: str) -> AstMetrics:
    """Run all four visitors over *source*; return the measurements.

    Args:
        source: Python source text for one file.

    Returns:
        AstMetrics with cyclomatic values, nesting, longest function,
        duplicated-block count, and total non-blank lines.
    """
    metrics = AstMetrics()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.debug("AST metrics skipped (syntax error): %s", exc)
        return metrics

    metrics.total_lines = len(
        [ln for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    )

    funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    for func in funcs:
        cyc = _CyclomaticVisitor()
        cyc.visit(func)
        metrics.cyclomatic_values.append(cyc.count)
        length = (func.end_lineno or func.lineno) - func.lineno + 1
        metrics.longest_function_lines = max(metrics.longest_function_lines, length)

    nesting = _NestingVisitor()
    nesting.visit(tree)
    metrics.max_nesting_depth = nesting.max_depth

    # Duplicated blocks: same normalized hash in 2+ functions.
    hash_counts: dict[str, int] = {}
    for func in funcs:
        h = _normalized_hash(func)
        if h:
            hash_counts[h] = hash_counts.get(h, 0) + 1
    metrics.duplicated_blocks = sum(c - 1 for c in hash_counts.values() if c > 1)

    return metrics


def snapshot_fields(source: str) -> dict[str, float | int]:
    """Return the MetricSnapshot-compatible fields for one file's source."""
    m = analyze_ast(source)
    return {
        "cyclomatic_mean": (
            (sum(m.cyclomatic_values) / len(m.cyclomatic_values)) if m.cyclomatic_values else 1.0
        ),
        "cyclomatic_max": max(m.cyclomatic_values) if m.cyclomatic_values else 1,
        "max_nesting_depth": m.max_nesting_depth,
        "longest_function_lines": m.longest_function_lines,
        "duplicated_blocks": m.duplicated_blocks,
        "total_lines": m.total_lines,
    }
