"""
Tests for the dead-code operator and API-surface extractor (E-10/E-11).
========================================================================
Unused imports are removed deterministically; the API surface freeze
rejects symbol renames/additions/removals.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.application.refinement.api_surface import (
    api_surface_equal,
    extract_api_surface,
)
from orchestrator.application.refinement.operators.dead_code import (
    DeadCodeOperator,
    RemoveUnusedImportsCommand,
    remove_unused_imports,
)
from orchestrator.domain.refinement import MetricSnapshot, RefinementTier
from orchestrator.domain.testing_models import Workspace


@pytest.mark.unit
class TestRemoveUnusedImports:
    """AST-based unused-import removal."""

    def test_removes_unused_import(self) -> None:
        src = "import os\nimport json\n\nx = json.dumps({'a': 1})\n"
        cleaned = remove_unused_imports(src)
        assert "import os" not in cleaned
        assert "import json" in cleaned

    def test_keeps_used_from_import(self) -> None:
        src = "from pathlib import Path\nfrom typing import Optional\n\ny: Optional[Path] = None\n"
        cleaned = remove_unused_imports(src)
        assert "Path" in cleaned and "Optional" in cleaned

    def test_removes_unused_from_import(self) -> None:
        src = "from collections import defaultdict, Counter\n\nx = defaultdict(int)\n"
        cleaned = remove_unused_imports(src)
        assert "Counter" not in cleaned
        assert "defaultdict" in cleaned

    def test_import_used_via_attribute(self) -> None:
        src = "import json\n\nprint(json.dumps(1))\n"
        assert "import json" in remove_unused_imports(src)

    def test_noop_on_clean_source(self) -> None:
        src = "import os\n\nprint(os.getcwd())\n"
        assert remove_unused_imports(src) == src

    def test_syntax_error_unchanged(self) -> None:
        src = "def broken(:\n"
        assert remove_unused_imports(src) == src


@pytest.mark.unit
class TestDeadCodeOperator:
    """Strategy operator proposes real candidates only."""

    def test_proposes_only_files_with_removable_imports(self, tmp_path: Path) -> None:
        (tmp_path / "svc.py").write_text(
            "import os\nimport json\n\nx = json.dumps({})\n", encoding="utf-8"
        )
        (tmp_path / "clean.py").write_text("import os\n\nprint(os.getcwd())\n", encoding="utf-8")
        ws = Workspace(root=tmp_path, framework="pytest")
        snapshot = MetricSnapshot(
            cyclomatic_mean=1.0,
            cyclomatic_max=1,
            max_nesting_depth=0,
            longest_function_lines=0,
            duplicated_blocks=0,
            total_lines=6,
        )
        op = DeadCodeOperator()
        assert op.tier is RefinementTier.MECHANICAL
        assert op.applicable(snapshot) is True
        candidates = asyncio.run(op.propose(ws, snapshot))
        assert len(candidates) == 1
        assert candidates[0].target_file == "svc.py"
        assert candidates[0].predicted_metric == "total_lines"

    def test_command_applies_and_reports_changed(self, tmp_path: Path) -> None:
        (tmp_path / "svc.py").write_text("import os\n\nx = 1\n", encoding="utf-8")
        cmd = RemoveUnusedImportsCommand(["svc.py"])
        changed = cmd.apply(tmp_path)
        assert changed == ["svc.py"]
        assert "import os" not in (tmp_path / "svc.py").read_text(encoding="utf-8")


@pytest.mark.unit
class TestApiSurface:
    """E-11 rule 6: public API surface freeze."""

    def test_extracts_public_functions(self) -> None:
        src = "def add(a, b):\n    return a + b\n\ndef _private(x):\n    return x\n"
        surface = extract_api_surface(src)
        assert "add" in surface
        assert "_private" not in surface

    def test_class_methods_and_constants(self) -> None:
        src = (
            "VERSION = '1.0'\n"
            "class Service:\n"
            "    def start(self, port: int = 80):\n        pass\n"
            "    def _helper(self):\n        pass\n"
        )
        surface = extract_api_surface(src)
        assert surface["VERSION"] == "const"
        assert surface["Service"] == "class"
        assert "start" in surface["Service.start"]
        assert "Service._helper" not in surface

    def test_signature_change_detected(self) -> None:
        before = extract_api_surface("def add(a, b):\n    return a + b\n")
        after = extract_api_surface("def add(a, b, c=0):\n    return a + b + c\n")
        assert api_surface_equal(before, after) is False

    def test_rename_detected(self) -> None:
        before = extract_api_surface("def add(a, b):\n    return a + b\n")
        after = extract_api_surface("def sum2(a, b):\n    return a + b\n")
        assert api_surface_equal(before, after) is False

    def test_identical_surface_equal(self) -> None:
        src = "def add(a, b):\n    return a + b\n"
        assert api_surface_equal(extract_api_surface(src), extract_api_surface(src)) is True

    def test_syntax_error_empty(self) -> None:
        assert extract_api_surface("def broken(:\n") == {}
