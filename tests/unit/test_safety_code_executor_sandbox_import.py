"""
orchestrator/safety/code_executor.py:183 imported DockerSandbox via a 3-dot
relative import (`from ...cost_optimization.docker_sandbox import DockerSandbox`)
from within the `orchestrator.safety` package — one dot too many, resolving
beyond the top-level `orchestrator` package. The comment directly above it
(line 182) shows the correct 2-dot form. `require_sandbox: bool = True` is
this class's own default, so under default config `_execute_in_sandbox()`
would raise ImportError unconditionally, before ever touching Docker.

This test exercises the actual import statement's resolution mechanism
directly (no Docker required — the bug fires before any Docker interaction).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import re
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "orchestrator" / "safety" / "code_executor.py"


def _relative_import_in(func_source: str) -> tuple[str, int]:
    """Return (module, level) for the first relative `from X import Y` in func_source."""
    tree = ast.parse(textwrap.dedent(func_source))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level > 0
            and "docker_sandbox" in (node.module or "")
        ):
            return node.module, node.level
    raise AssertionError("no relative docker_sandbox import found in _execute_in_sandbox")


def test_execute_in_sandbox_docker_sandbox_import_resolves():
    """The relative import inside _execute_in_sandbox must resolve to a real,
    importable module — not beyond the top-level package."""
    from orchestrator.safety.code_executor import CodeExecutor

    module, level = _relative_import_in(inspect.getsource(CodeExecutor._execute_in_sandbox))
    absolute_name = importlib.util.resolve_name("." * level + module, "orchestrator.safety")

    assert absolute_name == "orchestrator.cost_optimization.docker_sandbox"
    importlib.import_module(absolute_name)  # must not raise


def test_execute_in_sandbox_import_matches_its_own_fixed_comment():
    """No-regression guard: the '# FIXED:' comment directly above the import
    documents the correct 2-dot form — the two must never drift apart again."""
    src = _MODULE_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"#\s*FIXED:\s*(from\s+\.+cost_optimization\.docker_sandbox\s+import\s+DockerSandbox)"
        r"\s*\n\s*(from\s+\.+cost_optimization\.docker_sandbox\s+import\s+DockerSandbox)",
        src,
    )
    assert match is not None, "FIXED-comment/import pair not found in expected shape"
    assert match.group(1) == match.group(2), "the import must match its own '# FIXED:' comment"
