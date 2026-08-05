"""
Tests for the AST metrics collector (E-9).
==========================================
Each visitor against hand-built fixtures with known complexity/nesting/
length; duplication detection on near-miss pairs (same structure, different
identifiers => detected; different structure => not).
"""

from __future__ import annotations

import pytest

from orchestrator.infrastructure.metrics.ast_metrics import analyze_ast, snapshot_fields


@pytest.mark.unit
class TestCyclomaticComplexity:
    """Decision-point counting."""

    def test_flat_function_base_one(self) -> None:
        m = analyze_ast("def f():\n    return 1\n")
        assert m.cyclomatic_values == [1]

    def test_if_else_adds_branches(self) -> None:
        m = analyze_ast("def f(x):\n    if x:\n        return 1\n    return 2\n")
        assert m.cyclomatic_values == [2]

    def test_loops_and_boolean_add(self) -> None:
        src = "def f(xs):\n    for x in xs:\n        if x and y:\n            pass\n"
        m = analyze_ast(src)
        # base 1 + for 1 + if 1 + and 1 = 4
        assert m.cyclomatic_values == [4]

    def test_mean_and_max(self) -> None:
        src = "def a():\n    return 1\ndef b(x):\n    if x:\n        return 2\n    return 3\n"
        m = analyze_ast(src)
        assert m.cyclomatic_values == [1, 2]
        assert m.cyclomatic_max == 2
        assert m.cyclomatic_mean == 1.5


@pytest.mark.unit
class TestNesting:
    """Max block nesting depth."""

    def test_flat_is_depth_one(self) -> None:
        m = analyze_ast("def f():\n    return 1\n")
        assert m.max_nesting_depth == 1

    def test_nested_blocks(self) -> None:
        src = "def f(xs):\n    for x in xs:\n        if x:\n            try:\n                pass\n            except Exception:\n                pass\n"
        m = analyze_ast(src)
        assert m.max_nesting_depth == 4  # def > for > if > try

    def test_sibling_blocks_do_not_stack(self) -> None:
        src = "def f(xs):\n    for x in xs:\n        pass\n    if xs:\n        pass\n"
        m = analyze_ast(src)
        assert m.max_nesting_depth == 2


@pytest.mark.unit
class TestLongestFunction:
    """Source-line span of the longest function."""

    def test_single_function_span(self) -> None:
        src = "def f():\n    a = 1\n    b = 2\n    return a + b\n"
        m = analyze_ast(src)
        assert m.longest_function_lines == 4

    def test_takes_max_across_functions(self) -> None:
        src = "def short():\n    return 1\ndef long():\n    a = 1\n    b = 2\n    return a\n"
        m = analyze_ast(src)
        assert m.longest_function_lines == 4


@pytest.mark.unit
class TestDuplication:
    """Normalized subtree hashing catches near-miss duplicates."""

    def test_identical_functions_detected(self) -> None:
        src = (
            "def add(a, b):\n    return a + b\n"
            "def sub(x, y):\n    return x + y\n"  # same structure, diff names
        )
        m = analyze_ast(src)
        assert m.duplicated_blocks == 1

    def test_different_structure_not_detected(self) -> None:
        src = "def add(a, b):\n    return a + b\n" "def mul(x, y):\n    return x * y\n"
        m = analyze_ast(src)
        assert m.duplicated_blocks == 0

    def test_three_identical_counts_two(self) -> None:
        src = (
            "def a(x):\n    return x + 1\n"
            "def b(y):\n    return y + 1\n"
            "def c(z):\n    return z + 1\n"
        )
        m = analyze_ast(src)
        assert m.duplicated_blocks == 2


@pytest.mark.unit
class TestSnapshotFields:
    """snapshot_fields produces MetricSnapshot-compatible values."""

    def test_fields_present(self) -> None:
        fields = snapshot_fields("def f(x):\n    if x:\n        return 1\n    return 2\n")
        assert set(fields) == {
            "cyclomatic_mean",
            "cyclomatic_max",
            "max_nesting_depth",
            "longest_function_lines",
            "duplicated_blocks",
            "total_lines",
        }
        assert fields["cyclomatic_max"] == 2
        assert fields["total_lines"] >= 4

    def test_empty_source_is_safe(self) -> None:
        fields = snapshot_fields("")
        assert fields["cyclomatic_max"] == 1
        assert fields["total_lines"] == 0

    def test_syntax_error_is_safe(self) -> None:
        fields = snapshot_fields("def broken(:\n")
        assert fields["cyclomatic_max"] == 1
