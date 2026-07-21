"""
Contract tests for architecture invariants established by the
Architecture Remediation Plan.

These are **executable architecture guards** — they embed the rules that
must never regress. Import-linter catches some of these; these tests
assert higher-level invariants that import-linter's module-to-module
contracts don't directly express.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ORCHESTRATOR = PROJECT_ROOT / "orchestrator"

# ═══════════════════════════════════════════════════════════════════════════════
# Kernel allowlist (synced with scripts/check_new_root_files.py)
# ═══════════════════════════════════════════════════════════════════════════════
KERNEL_ALLOWLIST: set[str] = {
    "__init__.py",
    "__main__.py",
    "models.py",
    "model_registry.py",
    "task_factory.py",
    "prompt_builder.py",
    "budget.py",
    "exceptions.py",
    "constants.py",
    "log_config.py",
    "telemetry.py",
    "tracing.py",
    "api_clients.py",
    "resilience.py",
    "rate_limiter.py",
    "policy_engine.py",
    "telemetry_store.py",
    "model_selector.py",
    "engine.py",
    "cli.py",
    "state.py",
    "cache.py",
    "config.py",
    "autonomy_config.py",
    "concurrency_controller.py",
    "command_registry.py",
    "visualization.py",
    "output_organizer.py",
    "output_writer.py",
    "progress.py",
    "project_file.py",
    "assembler.py",
    "app_builder.py",
    "enhancer.py",
    "meta_integration.py",
    "semantic_cache.py",
    "checkpoints.py",
    "feature_flags.py",
    "automations.py",
    "validators.py",
}

# ─────────────────────────────────────────────────────────────────────────────
# Invariant 1: Root kernel allowlist
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_root_kernel_allowlist_not_growing():
    """Root-level orchestrator/*.py outside the kernel allowlist must not grow.

    The CI freeze guard (scripts/check_new_root_files.py --baseline) enforces
    this in CI. This local test checks that the violation count is monotonically
    decreasing as Workstream A progresses.

    Current baseline: 224 violations. This number MUST go down, never up.
    """
    exceptions = {"__init__.py", "__main__.py"}
    violations: list[str] = []
    for f in sorted(ORCHESTRATOR.glob("*.py")):
        if f.name in exceptions:
            continue
        if f.name not in KERNEL_ALLOWLIST:
            violations.append(f.name)

    # Baseline: 224 as of 2026-06-24 (Week 2 of remediation).
    # Update this number DOWN as files are deduplicated and moved.
    MAX_VIOLATIONS = 224
    assert len(violations) <= MAX_VIOLATIONS, (
        f"Root dump GREW from {MAX_VIOLATIONS} to {len(violations)} violations.\n"
        f"New violations: {set(violations) - set(KERNEL_ALLOWLIST)}\n"
        f"Root-level files must NOT increase until Workstream A is complete."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 2: engine_core/stages must not import from application
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_engine_core_stages_no_application_imports():
    """engine_core/stages pipeline stages must not import from application.

    Established in Workstream C1 (VSSamplerPort). Direct imports from
    application.verbalized_sampling break the architectural boundary.
    Imports via port injection (domain ports) are the correct path.
    """
    stages_dir = ORCHESTRATOR / "engine_core" / "stages"
    violations: list[str] = []

    for f in sorted(stages_dir.rglob("*.py")):
        if f.name == "__init__.py":
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        # Ignore TYPE_CHECKING blocks — those are runtime-noop
        for match in re.finditer(
            r"^(from\s+\.\.\.application|from\s+orchestrator\.application)", text, re.MULTILINE
        ):
            violations.append(f"{f.name}: {match.group(0).strip()}")

    assert (
        len(violations) == 0
    ), f"engine_core/stages imports from application ({len(violations)}):\n  " + "\n  ".join(
        violations
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 3: application/ must not import aiosqlite
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_application_no_aiosqlite():
    """Application-layer modules must not import aiosqlite directly.

    Established in Workstream C2 (SkillStore port). The aiosqlite
    dependency lives in infrastructure/skill_store_adapter.py, not
    in application/skill_store.py.
    """
    app_dir = ORCHESTRATOR / "application"
    violations: list[str] = []

    for f in sorted(app_dir.rglob("*.py")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        # Check for actual import statements, not comments/docstrings
        stripped = re.sub(r'"""', "", text)
        for line in stripped.split("\n"):
            if re.search(r"\bimport aiosqlite\b", line) or re.search(r"\bfrom aiosqlite\b", line):
                violations.append(f"{f.name}: {line.strip()}")

    assert (
        len(violations) == 0
    ), f"Application layer aiosqlite imports ({len(violations)}):\n  " + "\n  ".join(violations)


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 4: No fix-named modules in production paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_no_fix_named_modules():
    """No production module should contain '_fix_' or '_bug' in its name.

    Established in Workstream D4. Fix-named modules indicate temporal
    coupling and should be renamed to describe what they do.
    """
    exceptions = {"test_bug_regression.py", "test_bug_fixes_v2.py", "test_autonomy_config.py"}
    violations: list[str] = []

    for f in sorted(ORCHESTRATOR.rglob("*.py")):
        if f.name in exceptions:
            continue
        if "_fix_" in f.stem or "_bug" in f.stem:
            rel = f.relative_to(PROJECT_ROOT)
            violations.append(str(rel))

    assert len(violations) == 0, f"Fix-named modules found ({len(violations)}):\n  " + "\n  ".join(
        violations
    )
