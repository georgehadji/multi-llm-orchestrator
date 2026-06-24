#!/usr/bin/env python3
"""
Check for new root-level orchestrator/*.py files.

Fails CI if a new root-level Python file appears that is not on the
documented kernel allowlist. This enforces Workstream A1 of the
Architecture Remediation Plan (freeze the root dump).

Usage:
    python scripts/check_new_root_files.py

In CI (comparing against origin/master):
    git fetch origin master
    python scripts/check_new_root_files.py --baseline origin/master
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Kernel allowlist — files deliberately kept at orchestrator/ root level
# (Documented in ARCHITECTURE_REMEDIATION_PLAN.md Workstream F)
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
    "engine.py",  # Mediator — will be slimmed in Workstream B
    "cli.py",  # Thin dispatch wrapper
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
    "feature_flags.py",  # crosscutting config
    "automations.py",
    "validators.py",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_DIR = PROJECT_ROOT / "orchestrator"


def get_changed_root_files(baseline: str) -> list[Path]:
    """Return newly-*added* root-level .py files vs. the baseline git ref.

    Only ``--diff-filter=A`` (added) is used: the purpose is to freeze the root
    dump by blocking *new* root-level modules, not to flag edits to existing
    ones (a blanket reformat would otherwise trip every root file). Results are
    restricted to direct children of ``orchestrator/`` — the git pathspec glob
    matches nested subpackage files too, which are explicitly allowed.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", baseline, "--", "orchestrator/*.py"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"ERROR: git diff failed: {result.stderr}", file=sys.stderr)
        sys.exit(2)
    paths = [p for p in result.stdout.strip().splitlines() if p]
    # Keep only true root-level files: orchestrator/<name>.py (no subpackage).
    root_only = [p for p in paths if p.count("/") == 1]
    return [PROJECT_ROOT / p for p in root_only]


def get_all_root_files() -> list[Path]:
    """Return all current root-level orchestrator/*.py files."""
    return sorted(ORCHESTRATOR_DIR.glob("*.py"))


def check_files(files: list[Path], label: str) -> list[Path]:
    """Check which files from the list are NOT on the kernel allowlist."""
    violations: list[Path] = []
    for f in files:
        if f.name not in KERNEL_ALLOWLIST:
            violations.append(f)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for new root-level orchestrator/*.py files not on the kernel allowlist."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Git ref to compare against (e.g. origin/master). "
        "If omitted, checks all current root files.",
    )
    parser.add_argument(
        "--allowlist",
        action="store_true",
        help="Print the current kernel allowlist and exit.",
    )
    args = parser.parse_args()

    if args.allowlist:
        print(f"Kernel allowlist ({len(KERNEL_ALLOWLIST)} files):")
        for name in sorted(KERNEL_ALLOWLIST):
            print(f"  {name}")
        print(f"\nCurrent root files: {len(get_all_root_files())}")
        return 0

    if args.baseline:
        # CI mode: check files changed vs. baseline
        changed = get_changed_root_files(args.baseline)
        violations = check_files(changed, "CI")
        context = f"vs. {args.baseline}"
    else:
        # Full audit mode: check ALL current root files
        all_files = get_all_root_files()
        violations = check_files(all_files, "audit")
        context = "current state"

    if violations:
        print(f"❌ Root-level file violations ({context}):")
        for v in sorted(violations):
            print(f"   {v.relative_to(PROJECT_ROOT)}")
        print(f"\n{len(violations)} violation(s) found.")
        print("New root-level files must be added to KERNEL_ALLOWLIST in")
        print("scripts/check_new_root_files.py, or placed in a subpackage.")
        return 1
    else:
        print(f"✅ No root-level violations ({context}).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
