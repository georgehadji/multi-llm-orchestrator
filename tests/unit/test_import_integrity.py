"""Reachability gate: every module in the package must actually import.

Why this exists
---------------
The V4 P3-P11 sweep found 12 modules that raise on ``import`` — among them
``engine_core/sagas.py`` (850 LOC) and ``integrations/mcp_server.py`` (653 LOC) —
while all eleven CI gates passed. They passed because every other gate verifies
a property of code that *runs*, and nothing verified that the code can be loaded
at all. A module nothing imports is a module whose broken import nobody notices.

Eleven of the twelve had one cause: a file moved from ``orchestrator/X.py`` into a
sub-package kept its single-dot relative imports, so ``from .log_config import``
silently started meaning ``orchestrator.engine_core.log_config``.

Importing for real is the ground truth here. A static import-graph checker would
have to model ``TYPE_CHECKING`` blocks, conditional imports, and the entry-point
indirection ``engine_core/container.py`` uses for stage discovery — and would
still be a model. This cannot be fooled.

What counts as a failure
------------------------
Only *internal* breakage: an ``orchestrator.*`` module or symbol that does not
exist. A missing third-party package is not this gate's business — CI installs
``.[dev]`` plus fastapi, not the ``dashboard``/``security``/``tracing`` extras, so
optional-dependency misses are expected here and are covered by the packaging
gates instead. ``ImportError.name`` distinguishes the two exactly: it holds the
*orchestrator* module for an internal failure and the *third-party package* for a
dependency miss.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings

import pytest

import orchestrator

PACKAGE_PREFIX = "orchestrator"


def _internal_import_failures() -> dict[str, str]:
    """Import every module under ``orchestrator``; return the internal failures.

    Returns a mapping of module name -> diagnosis, empty when the package is
    fully loadable.
    """
    failures: dict[str, str] = {}

    # Import-time side effects (logger configuration, "frontend not found"
    # warnings) are noise here, not signal — the gate is about loadability.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for module in pkgutil.walk_packages(orchestrator.__path__, f"{PACKAGE_PREFIX}."):
            try:
                importlib.import_module(module.name)
            except ImportError as exc:
                # exc.name is the orchestrator module for internal breakage and
                # the distribution name for a missing optional dependency.
                missing = getattr(exc, "name", None) or ""
                if missing == PACKAGE_PREFIX or missing.startswith(f"{PACKAGE_PREFIX}."):
                    failures[module.name] = f"{type(exc).__name__}: {exc}"
            except Exception:  # noqa: BLE001
                # A module that raises something other than ImportError is
                # misbehaving at import time, but that is a different defect
                # class with different remedies. Out of scope for this gate.
                continue

    return failures


@pytest.mark.unit
def test_every_module_imports() -> None:
    """No module may fail to import because of a broken *internal* reference."""
    failures = _internal_import_failures()

    assert not failures, "modules that cannot be imported:\n" + "\n".join(
        f"  {name}\n      {why}" for name, why in sorted(failures.items())
    )


@pytest.mark.unit
def test_no_unawaited_get_event_bus() -> None:
    """`get_event_bus()` must never be called without `await` (PX-BUS1).

    It is a coroutine function, so an un-awaited call binds a truthy coroutine
    with no `publish`/`subscribe`. That shape was fixed as P1-5, then again as
    P2-S2-2/2b, then found at eight further sites by the P3-P11 sweep — three
    rounds of patching call sites without converging. Synchronous callers must
    use `get_event_bus_sync()` instead; this gate is what makes that stick.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "orchestrator"
    offenders: list[str] = []

    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue

        awaited = {
            id(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Call)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or id(node) in awaited:
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "get_event_bus":
                offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")

    assert not offenders, (
        "get_event_bus() called without await — use get_event_bus_sync() from "
        "synchronous code:\n" + "\n".join(f"  {o}" for o in offenders)
    )
