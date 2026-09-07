"""
T17 (duplicate-pair convergence sweep) proof-of-defect and no-regression tests.

The wave enumerated every subpackage module sharing a filename with a root
module where *both* sides carry their own definitions — 65 candidates — and
converged the subset where convergence is provably behaviour-preserving.

C1 — `integrations/swiftstack_integration.py` was a byte-identical copy of
     `orchestrator/swiftstack_integration.py` sitting one package deeper, so
     its root-relative imports (`from .api_builder import ...`) resolved
     against `orchestrator.integrations.*` and raised ModuleNotFoundError on
     every import. A byte-identical duplicate that could not actually load.
C3 — four further byte-identical duplicate pairs (component_library 879 lines,
     frontend_security 1058, indesign_plugin_rules 1131, ios_hig_prompts 474)
     kept two independently-editable copies of the same code reachable under
     two import paths, so a fix to one would silently not reach the other.
C4 — `design/design_system.py` was a stale 155-line fork of the 310-line
     canonical root module, missing the `tone`/`font_heading`/`font_body`/
     `accessibility` fields and the `__post_init__` that materialises
     `spacing`/`shadow`/`animation`/`border_radius`. Every real consumer
     imports the root module, but the fork was exposed through
     `design/__init__.py`'s wildcard, so `from orchestrator.design import
     DesignSystem` handed out a class raising AttributeError on exactly the
     attributes `website_generator.py` formats into its output.
Gate — `scripts/check_duplicate_pairs.py` freezes the remaining 59 pairs so
     new duplication cannot appear unnoticed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "check_duplicate_pairs.py"


# --- C1 -----------------------------------------------------------------


def test_c1_swiftstack_subpackage_copy_imports_and_matches_root():
    import orchestrator.integrations.swiftstack_integration as sub
    import orchestrator.swiftstack_integration as root

    assert sub.APIIntegration is root.APIIntegration
    assert sub.APIIntegrationBuilder is root.APIIntegrationBuilder


# --- C3 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("sub_mod", "root_mod", "symbol"),
    [
        (
            "orchestrator.design.component_library",
            "orchestrator.component_library",
            "ComponentLibrary",
        ),
        ("orchestrator.design.frontend_security", "orchestrator.frontend_security", "CSPDirective"),
        (
            "orchestrator.security.indesign_plugin_rules",
            "orchestrator.indesign_plugin_rules",
            "InDesignPluginRules",
        ),
        (
            "orchestrator.security.ios_hig_prompts",
            "orchestrator.ios_hig_prompts",
            "HIG_ACCESSIBILITY_TEMPLATE",
        ),
    ],
)
def test_c3_byte_identical_duplicates_now_share_one_definition(sub_mod, root_mod, symbol):
    import importlib

    sub = importlib.import_module(sub_mod)
    root = importlib.import_module(root_mod)
    assert getattr(sub, symbol) is getattr(root, symbol), (
        f"{sub_mod}.{symbol} is a separate object from {root_mod}.{symbol} — "
        "the two copies can drift apart again"
    )


# --- C4 -----------------------------------------------------------------


def test_c4_design_package_exposes_the_full_design_system():
    """The package-level DesignSystem must carry the fields website_generator uses."""
    from orchestrator.design import DesignSystem
    from orchestrator.design_system import DesignSystem as RootDesignSystem

    assert DesignSystem is RootDesignSystem

    ds = DesignSystem(tone="luxury")
    # Each of these is formatted into generated output by website_generator.py;
    # the stale fork defined none of them.
    assert ds.tone == "luxury"
    assert ds.font_heading
    assert ds.accessibility.min_contrast_ratio
    assert ds.border_radius.md


# --- Gate ---------------------------------------------------------------


def _run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_gate_passes_on_current_tree():
    result = _run_gate()
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_flags_a_new_duplicate_pair_and_accepts_a_shim(tmp_path):
    """Inject a genuine new pair, assert it fails; shim it, assert it passes."""
    root_probe = REPO_ROOT / "orchestrator" / "__t17_gate_probe.py"
    sub_probe = REPO_ROOT / "orchestrator" / "analysis" / "__t17_gate_probe.py"
    try:
        root_probe.write_text("class Probe:\n    pass\n", encoding="utf-8")
        sub_probe.write_text("class Probe:\n    pass\n", encoding="utf-8")
        failed = _run_gate()
        assert failed.returncode == 1
        assert "__t17_gate_probe.py" in failed.stderr

        # Converging the pair into a shim must satisfy the gate.
        sub_probe.write_text(
            "from ..__t17_gate_probe import *  # noqa: F401, F403\n", encoding="utf-8"
        )
        passed = _run_gate()
        assert passed.returncode == 0, passed.stdout + passed.stderr
    finally:
        root_probe.unlink(missing_ok=True)
        sub_probe.unlink(missing_ok=True)


def test_gate_ignores_same_filename_with_disjoint_definitions(tmp_path):
    """R5: `orchestrator/performance.py` (LRU cache) and
    `orchestrator/application/refinement/operators/performance.py` (E-12
    refinement operator) share a filename and both define things, but share
    zero symbol names — they are unrelated modules, not a stale fork. The
    gate must not flag a same-filename collision unless a top-level
    class/function name is actually shared.
    """
    root_probe = REPO_ROOT / "orchestrator" / "__t17_gate_probe.py"
    sub_probe = REPO_ROOT / "orchestrator" / "analysis" / "__t17_gate_probe.py"
    try:
        root_probe.write_text("class RootOnlyThing:\n    pass\n", encoding="utf-8")
        sub_probe.write_text("class SubOnlyThing:\n    pass\n", encoding="utf-8")
        result = _run_gate()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "__t17_gate_probe.py" not in result.stdout + result.stderr
    finally:
        root_probe.unlink(missing_ok=True)
        sub_probe.unlink(missing_ok=True)
