"""Pure AST locators for the structural refinement operators (E-11).

Each finder answers "where is the single worst offender for this metric?"
so a candidate can cite an exact, targeted finding — "function `handle()`
at `svc.py:88` has 5 nesting levels" — never a generic "improve this code"
instruction (plan §3.4.5).

Self-contained rather than importing
``infrastructure.metrics.ast_metrics``: Contract 2 forbids the application
layer from importing concrete infrastructure adapters, and these finders
answer a different question (*which* function, not the workspace
aggregate) from the ``MetricSnapshot`` collectors that gate entry. The
small duplication of the normalized-hash technique mirrors the existing
precedent in ``operators/dead_code.py``, which does its own self-contained
AST pass rather than reaching into infrastructure.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass
from typing import Iterator

_FuncNode = ast.FunctionDef | ast.AsyncFunctionDef
_NESTING_BLOCK_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
)


@dataclass(frozen=True)
class FunctionLocation:
    """A single located function, with the metric value that selected it."""

    name: str
    lineno: int
    end_lineno: int
    line_count: int
    metric_value: int


def _iter_functions(tree: ast.AST) -> Iterator[_FuncNode]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _location(node: _FuncNode, metric_value: int) -> FunctionLocation:
    end = node.end_lineno or node.lineno
    return FunctionLocation(
        name=node.name,
        lineno=node.lineno,
        end_lineno=end,
        line_count=end - node.lineno + 1,
        metric_value=metric_value,
    )


def find_longest_function(source: str) -> FunctionLocation | None:
    """Return the function with the most source lines, or None.

    Args:
        source: Python module source.

    Returns:
        The longest function's location, or None on a syntax error or a
        module with no functions.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    best: FunctionLocation | None = None
    for node in _iter_functions(tree):
        loc = _location(node, node.end_lineno and node.end_lineno - node.lineno + 1 or 1)
        if best is None or loc.line_count > best.line_count:
            best = loc
    return best


class _FunctionNestingVisitor(ast.NodeVisitor):
    """Max block-nesting depth strictly inside one function's own body."""

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


def find_deepest_nesting_function(source: str) -> FunctionLocation | None:
    """Return the function with the deepest control-flow nesting, or None.

    Args:
        source: Python module source.

    Returns:
        The most-nested function's location (``metric_value`` is its max
        nesting depth), or None when nothing is nested or on syntax error.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    best: FunctionLocation | None = None
    for node in _iter_functions(tree):
        visitor = _FunctionNestingVisitor()
        for child in node.body:
            visitor.visit(child)
        if visitor.max_depth == 0:
            continue
        loc = _location(node, visitor.max_depth)
        if best is None or loc.metric_value > best.metric_value:
            best = loc
    return best


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


def _normalized_hash(node: _FuncNode) -> str:
    # Deep-copy first: NodeTransformer mutates in place, and callers read
    # the original node's .name/.lineno *after* hashing (e.g. to report
    # which function was found) — hashing must never clobber the source tree.
    normalized = _NormalizingTransformer().visit(copy.deepcopy(node))
    ast.fix_missing_locations(normalized)
    try:
        text = ast.unparse(normalized)
    except Exception:  # pragma: no cover - unparse is robust
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_duplicate_pair(source: str) -> tuple[FunctionLocation, FunctionLocation] | None:
    """Return the first pair of structurally-identical functions, or None.

    Scoped to a single file/module — cross-file deduplication would need
    to update call sites and imports in a second file, which the
    single-target-file Command this operator uses cannot express safely.

    Args:
        source: Python module source.

    Returns:
        A ``(first, second)`` pair of locations sharing a normalized AST
        hash (same structure, different identifiers), or None.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    seen: dict[str, _FuncNode] = {}
    for node in _iter_functions(tree):
        digest = _normalized_hash(node)
        if not digest:
            continue
        if digest in seen:
            first_node = seen[digest]
            return _location(first_node, 1), _location(node, 1)
        seen[digest] = node
    return None


__all__ = [
    "FunctionLocation",
    "find_deepest_nesting_function",
    "find_duplicate_pair",
    "find_longest_function",
]
