"""Ratchet: a root module with a same-name subpackage twin must be a short shim.

Audit finding #19: about 20 root/subpackage module pairs held two diverging
copies of the same code (~8,700 lines). Collapsing them one PR at a time means
the root copy becomes a re-export shim of the subpackage copy. This gate stops
new pairs appearing and makes each collapse permanent.

ALLOWLIST holds the pairs not yet collapsed. It may only shrink: an entry whose
root file is already a short shim fails the test until it is removed. Some
entries are plain name clashes (for example ``models.py`` and
``nexus_search/models.py``), not copies; review those before removing them.

Approved as a new gate by the owner on 2026-10-29 (change-control: new gates
need human approval).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

MAX_SHIM_LINES = 40
PACKAGE = Path(__file__).resolve().parents[2] / "orchestrator"

ALLOWLIST: frozenset[str] = frozenset(
    {
        "adaptive_router.py",
        "ara_integration.py",
        "architecture_advisor.py",
        "architecture_rules.py",
        "assembler.py",
        "assumption_gate.py",
        "autonomous_debugger.py",
        "capability_logger.py",
        "checkpoints.py",
        "circuit_breaker.py",
        "code_executor.py",
        "component_library.py",
        "concurrency_controller.py",
        "config.py",
        "control_plane.py",
        "dashboard_bridge.py",
        "deployment_feedback.py",
        "design_system.py",
        "design_to_code.py",
        "diff_generator.py",
        "enhancer.py",
        "federated_learning.py",
        "feedback_loop.py",
        "frontend_security.py",
        "gradual_rollout.py",
        "hitl_workflow.py",
        "image_generator.py",
        "indesign_plugin_rules.py",
        "ios_hig_prompts.py",
        "knowledge_base.py",
        "knowledge_graph.py",
        "log_config.py",
        "memory_tier.py",
        "metrics.py",
        "models.py",
        "performance.py",
        "phase_aware_models.py",
        "preflight.py",
        "product_manager.py",
        "progress.py",
        "progress_writer.py",
        "progressive_output.py",
        "prompt_builder.py",
        "quality_control.py",
        "rate_limiter.py",
        "reference_monitor.py",
        "reranker.py",
        "secrets_generator.py",
        "session_lifecycle.py",
        "session_watcher.py",
        "streaming.py",
        "swiftstack_integration.py",
        "tdd_config.py",
        "tenancy.py",
        "transfer_learning.py",
        "triggers.py",
        "visualization.py",
        "website_generator.py",
    }
)


def _twinned_root_modules() -> dict[str, int]:
    subpackages = [d for d in PACKAGE.iterdir() if d.is_dir() and d.name != "__pycache__"]
    found: dict[str, int] = {}
    for module in PACKAGE.glob("*.py"):
        if module.name == "__init__.py":
            continue
        if any((d / module.name).is_file() for d in subpackages):
            text = module.read_text(encoding="utf-8", errors="replace")
            found[module.name] = len(text.splitlines())
    return found


def test_no_new_long_root_twins() -> None:
    long_twins = {n for n, lines in _twinned_root_modules().items() if lines > MAX_SHIM_LINES}
    new = sorted(long_twins - ALLOWLIST)
    assert not new, (
        f"Root modules with a subpackage twin must be <= {MAX_SHIM_LINES} lines "
        f"(a re-export shim). Collapse the pair instead: {new}"
    )


def test_allowlist_only_shrinks() -> None:
    long_twins = {n for n, lines in _twinned_root_modules().items() if lines > MAX_SHIM_LINES}
    stale = sorted(ALLOWLIST - long_twins)
    assert not stale, f"Already collapsed or gone; remove from ALLOWLIST: {stale}"
