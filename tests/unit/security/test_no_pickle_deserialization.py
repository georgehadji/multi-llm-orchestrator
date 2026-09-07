"""SEC-002 — nothing in the package may deserialize pickle.

``pickle.loads`` on cache data is remote code execution the moment an attacker
can write to the cache store (a shared Redis, a writable ``.cache/`` directory,
a restored backup). The fix was to delete the only module that did it; this
test keeps it from coming back.

Parsing is done with ``ast`` rather than a regex on purpose: the benchmark
prompt in ``orchestrator/analysis/leaderboard.py`` contains the literal text
``pickle.load(f)`` inside a docstring, and a regex ban would flag it forever.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "orchestrator"

#: Callables that turn bytes back into live objects.
BANNED_ATTRS = frozenset({"load", "loads", "Unpickler"})
BANNED_MODULES = frozenset({"pickle", "cPickle", "dill", "marshal"})


def _python_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _deserialization_calls(tree: ast.AST) -> list[tuple[int, str]]:
    """Find `pickle.loads(...)`-shaped calls and `from pickle import loads`."""
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # pickle.loads(data)
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in BANNED_ATTRS
                and isinstance(func.value, ast.Name)
                and func.value.id in BANNED_MODULES
            ):
                found.append((node.lineno, f"{func.value.id}.{func.attr}()"))

        # from pickle import loads
        elif isinstance(node, ast.ImportFrom):
            if node.module in BANNED_MODULES:
                for alias in node.names:
                    if alias.name in BANNED_ATTRS:
                        found.append((node.lineno, f"from {node.module} import {alias.name}"))

    return found


def test_no_pickle_deserialization_in_package() -> None:
    offenders: list[str] = []

    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            # Not our concern here — other guards cover unparseable modules.
            continue
        for lineno, what in _deserialization_calls(tree):
            offenders.append(f"{path.relative_to(PACKAGE_ROOT.parent)}:{lineno}: {what}")

    assert not offenders, (
        "pickle/dill/marshal deserialization is banned (SEC-002) — "
        "use JSON with an explicit schema:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("module", ["orchestrator.caching", "orchestrator.infrastructure.caching"])
def test_dead_pickle_cache_stays_deleted(module: str) -> None:
    """The multi-layer cache was removed, not repaired.

    It had no importers and raised ``AttributeError`` on every write
    (``datetime.now(datetime.timezone.utc)()``), so there was nothing to
    migrate. Reintroducing it reintroduces the pickle sink.
    """
    import importlib.util

    assert importlib.util.find_spec(module) is None, (
        f"{module} is back — if a multi-layer cache is genuinely needed, "
        "build it on a JSON codec, not pickle."
    )
