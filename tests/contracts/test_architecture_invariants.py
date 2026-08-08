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
# Invariant 9: Refinement domain purity + no module-level singleton (Phase 6)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_refinement_domain_is_stdlib_only():
    """Phase 6: ``domain/refinement.py`` imports stdlib only (Contract 1)."""
    import ast as _ast

    path = ORCHESTRATOR / "domain" / "refinement.py"
    tree = _ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, _ast.ImportFrom):
            imports.append(node.module or "")
    stdlib = {
        "__future__",
        "dataclasses",
        "enum",
        "typing",
        "functools",
        "abc",
        "re",
        "collections",
    }
    foreign = [i for i in imports if i.split(".")[0] not in stdlib]
    assert foreign == [], f"domain/refinement.py imports non-stdlib: {foreign}"


@pytest.mark.contract
def test_readiness_domain_is_stdlib_only():
    """Phase 7: ``domain/readiness.py`` imports stdlib only (Contract 1)."""
    import ast as _ast

    path = ORCHESTRATOR / "domain" / "readiness.py"
    tree = _ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, _ast.ImportFrom):
            imports.append(node.module or "")
    stdlib = {"__future__", "dataclasses", "enum", "typing"}
    foreign = [i for i in imports if i.split(".")[0] not in stdlib]
    assert foreign == [], f"domain/readiness.py imports non-stdlib: {foreign}"


@pytest.mark.contract
def test_no_module_level_singleton_registry_in_metrics():
    """Phase 6: metric collectors have no module-level singleton registry.

    The plan (§3.4.2) requires the collector registry to be constructed in
    the composition root like every other collaborator, never as a global.
    """
    text = (ORCHESTRATOR / "infrastructure" / "metrics" / "collector.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    assert "_registry" not in text
    assert re.search(r"^registry\s*=", text, re.MULTILINE) is None


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 10: No fix-named modules in production paths
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


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 5: Exactly one authoritative test-execution implementation (F-6)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_single_authoritative_test_runner():
    """F-6: exactly one non-shim ``class TestRunner`` exists in the tree.

    The four historical runner implementations (runtime/sandbox.py,
    quality/quality_control.py, testing/first_generator.py, quality/run_tests.py)
    were consolidated into ``infrastructure/test_runners/``. Any remaining
    ``class TestRunner`` must be a deprecation shim that delegates.
    """
    shims = {
        "orchestrator/runtime/sandbox.py",
        "orchestrator/quality/quality_control.py",
    }
    non_shim: list[str] = []
    for f in sorted(ORCHESTRATOR.rglob("*.py")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^class TestRunner\b", text, re.MULTILINE):
            rel = f.relative_to(PROJECT_ROOT).as_posix()
            if rel not in shims:
                non_shim.append(rel)
    assert non_shim == [], (
        f"More than one authoritative TestRunner ({non_shim}); "
        f"consolidate into infrastructure/test_runners/ (F-6)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 6: No blocking subprocess.run under testing/ (F-1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_no_blocking_subprocess_in_testing():
    """F-1: no ``subprocess.run`` remains under ``orchestrator/testing/``.

    Blocking subprocess calls inside the async test-execution path stall
    the event loop (D-1). All child-process work must go through
    ``asyncio.create_subprocess_exec``.
    """
    testing_dir = ORCHESTRATOR / "testing"
    violations: list[str] = []
    for f in sorted(testing_dir.rglob("*.py")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        for line in text.split("\n"):
            if re.search(r"subprocess\.run\(", line):
                violations.append(f"{f.name}: {line.strip()}")
    assert violations == [], f"subprocess.run in testing/ (F-1):\n  " + "\n  ".join(violations)


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 7: No exec()/eval() of model-derived strings (F-2)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_no_exec_of_model_derived_strings():
    """F-2: no ``exec(``/``eval(`` applied to artifact strings in verification.

    D-2 removed the in-process ``exec(compile(artifact))`` from
    ``verification_checks.py``; the import check now runs out of process.
    This guard prevents the pattern from returning. ``exec()`` of literal,
    non-model-derived constants (e.g. exec of a fixed template) is allowed.
    """
    violations: list[str] = []
    for f in sorted(ORCHESTRATOR.rglob("*.py")):
        if "__pycache__" in str(f):
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"\bexec\(\s*(compile\(|artifact|code|source)", stripped):
                violations.append(f"{f.relative_to(PROJECT_ROOT)}: {stripped}")
    assert violations == [], f"exec() of model-derived string (F-2):\n  " + "\n  ".join(violations)


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 8: Testing limits live only in config/limits.json (F-8)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_testing_limits_have_single_source_of_truth():
    """F-8: testing limits are declared once in config/limits.json.

    ``max_repair_iterations``, ``suite_timeout_s``, ``mutation_sample_size``
    and ``mutation_timeout_s`` are read via ``testing_config.py``. The
    first_generator must not re-declare their default values as literals.
    """
    import json

    limits_path = PROJECT_ROOT / "orchestrator" / "config" / "limits.json"
    data = json.loads(limits_path.read_text(encoding="utf-8"))
    testing = data.get("testing", {})
    for key in (
        "max_repair_iterations",
        "suite_timeout_s",
        "mutation_sample_size",
        "mutation_timeout_s",
    ):
        assert key in testing, f"limits.json missing testing.{key} (F-8)"

    # The loader may declare defaults; the generator must not duplicate them.
    # Env overrides and explicit per-call overrides (e.g. in tests) are fine.
    # LLM-call timeouts (timeout=120) are a different limit and are allowed.
    generator_text = (PROJECT_ROOT / "orchestrator" / "testing" / "first_generator.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    for literal in ("120", "60"):
        line_hits = [
            ln.strip()
            for ln in generator_text.split("\n")
            if re.search(rf"timeout_s\s*=\s*{literal}\b|\b{literal}\b.*suite_timeout", ln)
            and "ORCH_" not in ln
            and "limits.json" not in ln
            and "testing_config" not in ln
            and "F-8" not in ln
        ]
        assert (
            not line_hits
        ), f"F-8: suite timeout {literal} hardcoded in first_generator.py:\n  " + "\n  ".join(
            line_hits
        )
