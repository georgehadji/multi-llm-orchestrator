"""
Tests for the RED-gate stub synthesizer (E-2).
===============================================
Pure-AST unit tests: stub synthesis for functions, classes, async
functions, and constants; degenerate suite detection.
"""

from __future__ import annotations

import ast

import pytest

from orchestrator.application.testing.stub_synthesizer import (
    synthesize_stub,
)


@pytest.mark.unit
class TestStubSynthesizer:
    """E-2: stub generation from test-referenced symbols."""

    def test_from_import_referenced_names(self) -> None:
        stub = synthesize_stub(
            "from main import add, subtract\ndef test_x():\n    assert add(1, 2) == 3\n"
        )
        tree = ast.parse(stub)
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert {"add", "subtract"} <= names

    def test_attribute_access_referenced(self) -> None:
        stub = synthesize_stub("import main\ndef test_x():\n    assert main.VERSION == '1.0'\n")
        assert "def VERSION(" in stub

    def test_async_function_referenced(self) -> None:
        stub = synthesize_stub(
            "from main import fetch\ndef test_x():\n    assert fetch() is not None\n"
        )
        assert "def fetch(" in stub

    def test_class_reference(self) -> None:
        stub = synthesize_stub(
            "from main import Calculator\ndef test_x():\n    c = Calculator()\n    assert c\n"
        )
        assert "def Calculator(" in stub

    def test_stub_raises_not_implemented(self) -> None:
        stub = synthesize_stub("from main import add\ndef test_x():\n    assert add(1, 2) == 3\n")
        ns: dict = {}
        exec(stub, ns)  # nosec B102 — executing our OWN synthesized stub, not model output
        try:
            ns["add"](1, 2)
            pytest.fail("stub add() must raise")
        except NotImplementedError:
            pass

    def test_no_references_still_importable(self) -> None:
        stub = synthesize_stub("def test_independent():\n    assert True\n")
        tree = ast.parse(stub)
        assert len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]) >= 1

    def test_stdlib_imports_not_synthesized(self) -> None:
        stub = synthesize_stub(
            "import os\nfrom main import add\ndef test_x():\n    assert add(1, 2) == 3\n"
        )
        assert "def os(" not in stub
        assert "def add(" in stub

    def test_sibling_module_import_synthesized(self) -> None:
        """Multi-file tasks: imports from sibling modules must be satisfiable."""
        stub = synthesize_stub(
            "from b import quadruple\ndef test_x():\n    assert quadruple(2) == 8\n",
            module="main",
        )
        assert "def quadruple(" in stub
