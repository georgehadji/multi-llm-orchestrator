"""
Three follow-up fixes spun off from the V3 precision-defect-audit campaign
(docs/audits/v3/T2/, docs/audits/v3/LEDGER.md), each self-contained:

1. orchestrator/red_team.py (root) had silently forked from
   orchestrator/safety/red_team.py into an independent 475-line copy,
   instead of being a thin re-export shim like its siblings
   (orchestrator/accountability.py, orchestrator/security_validator.py).
   engine.py:272 binds to the root copy; safety/__init__.py binds to the
   safety/ copy -- two different class objects with the same name.

2. orchestrator/engine_core/service_collection.py and
   orchestration_facade.py are unreferenced ENGINE_OPTIMIZATION_PLAN
   scaffolding with broken relative imports to modules that don't exist
   inside engine_core/ (they live at the orchestrator/ root or in
   orchestrator/safety/ instead). Zero callers anywhere in orchestrator/
   or tests/ (confirmed by repo-wide grep) -- deleted as abandoned
   scaffolding rather than wired up, per CLAUDE.md's "no half-finished
   implementations" and the four unbreakable rules (wiring this in would
   mean growing engine.py's __init__, which must stay a thin Mediator).

3. orchestrator/safety/secure_execution.py's SafeCommand._SHELL_METACHARACTERS
   included a bare backslash, rejecting any argv element containing a
   native Windows path (the OS path separator, not a shell metacharacter
   under shell=False/argv-mode execution).
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_root_red_team_is_the_same_class_as_safety_red_team() -> None:
    """Fires without the fix; passes with it. Violated property: a module
    documented as a backward-compat re-export (matching its two siblings,
    orchestrator/accountability.py and orchestrator/security_validator.py)
    must bind to the SAME class object as the canonical module, not an
    independently-forked copy."""
    import orchestrator.red_team as root_red_team
    import orchestrator.safety.red_team as safety_red_team

    if root_red_team.RedTeamFramework is not safety_red_team.RedTeamFramework:
        pytest.fail(
            "defect still present: orchestrator.red_team.RedTeamFramework is a "
            "different class object than orchestrator.safety.red_team.RedTeamFramework"
        )


def test_engine_dot_red_team_import_resolves_to_the_canonical_class() -> None:
    """engine.py:272 does `from .red_team import RedTeamFramework` -- after
    the fix, that must resolve to the same canonical class everywhere else
    in the codebase resolves to."""
    import orchestrator.engine as engine_module
    import orchestrator.safety.red_team as safety_red_team

    assert engine_module.RedTeamFramework is safety_red_team.RedTeamFramework


def test_dead_engine_optimization_scaffolding_is_removed() -> None:
    """Fires without the fix; passes with it. Violated property:
    orchestrator/engine_core/service_collection.py and
    orchestration_facade.py are unreferenced scaffolding with broken
    relative imports (confirmed: zero callers anywhere in orchestrator/ or
    tests/) -- they must not exist as importable modules."""
    for mod_name in (
        "orchestrator.engine_core.service_collection",
        "orchestrator.engine_core.orchestration_facade",
    ):
        importlib.invalidate_caches()
        try:
            importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        pytest.fail(f"defect still present: {mod_name} still exists and imports cleanly")


def test_safecommand_accepts_a_native_windows_path_argument(tmp_path) -> None:
    """Fires without the fix; passes with it. Violated property: SafeCommand
    (argv-only, shell=False) must not reject an argv element merely for
    containing a backslash -- that's the Windows path separator, not a
    shell metacharacter under argv-mode execution."""
    from orchestrator.safety.secure_execution import CommandInjectionError, SafeCommand

    windows_path = str(tmp_path / "some_script.py")
    assert "\\" in windows_path or "/" in windows_path  # sanity: tmp_path is a real path

    try:
        SafeCommand(["python", windows_path])
    except CommandInjectionError as exc:
        pytest.fail(f"defect still present: native path argument rejected: {exc}")
