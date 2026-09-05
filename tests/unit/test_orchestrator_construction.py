"""
The most basic invariant this project has: `Orchestrator()` must construct.

It did not. On master, `Orchestrator()` raised — first
`AttributeError: type object 'ConstitutionGate' has no attribute '_build_stage'`
from the DI container's stage discovery, and behind that
`TypeError: build_health_tracker() got an unexpected keyword argument 'dashboard'`,
a parameter-name drift between engine.py's call site and the factory in
engine_slimming.py (whose parameter is `dashboard_bridge`).

Because construction failed at import-of-fixture time, 26 tests across
tests/test_phase8_mvos.py and tests/unit/test_engine_run_project.py errored
during setup rather than reporting a cause — the failure looked like fixture
noise instead of "the main entry point is dead".

There was no test asserting plain construction. There is now.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    """OPENROUTER_API_KEY is validated eagerly at client construction."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-dummy")


@pytest.mark.unit
def test_orchestrator_constructs_with_no_arguments():
    from orchestrator.engine import Orchestrator

    assert Orchestrator() is not None


@pytest.mark.unit
def test_orchestrator_constructs_with_a_budget():
    from orchestrator.budget import Budget
    from orchestrator.engine import Orchestrator

    assert Orchestrator(budget=Budget(max_usd=1.0)) is not None


@pytest.mark.unit
def test_health_tracker_call_site_matches_factory_signature():
    """Pins the drift directly, so a rename on either side fails loudly here.

    engine.py calls build_health_tracker(...) by keyword; every keyword it
    passes must exist as a parameter of the factory.
    """
    import re
    from pathlib import Path

    from orchestrator.engine_slimming import build_health_tracker

    accepted = set(inspect.signature(build_health_tracker).parameters)

    engine_src = (Path(__file__).resolve().parents[2] / "orchestrator" / "engine.py").read_text(
        encoding="utf-8"
    )
    call = re.search(r"build_health_tracker\((.*?)\n        \)", engine_src, re.S)
    assert call, "could not locate the build_health_tracker call site in engine.py"
    passed = set(re.findall(r"^\s*([a-z_][a-z_0-9]*)=", call.group(1), re.M))

    assert passed, "parsed no keyword arguments from the call site"
    unknown = passed - accepted
    assert not unknown, (
        f"engine.py passes keyword(s) {sorted(unknown)} that "
        f"build_health_tracker() does not accept (it accepts {sorted(accepted)})"
    )
