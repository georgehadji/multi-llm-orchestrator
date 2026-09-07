#!/usr/bin/env python3
"""
Duplicate root/subpackage pair freeze check.

This repository keeps its public surface at ``orchestrator/`` depth 1 and its
implementations in subpackages, bridged by re-export shims.  A *duplicate pair*
is a subpackage module that shares its filename with a root module while both
sides carry their own class/function definitions — i.e. two independent forks
reachable under two import paths.

That shape has produced repeated defects (hunt tiers T5, T13, T14, T16, T17):
a fix lands on one copy, the other keeps the bug, and whichever one a caller
happens to import decides whether the bug is live.  This check freezes the
known pairs so new ones cannot appear unnoticed.

A pair counts as *resolved* once either side's module body contains no class or
function definitions — that side is a re-export shim, so both import paths
yield the same objects.

Usage::

    python scripts/check_duplicate_pairs.py          # check (CI)
    python scripts/check_duplicate_pairs.py --list   # show current pairs
    python scripts/check_duplicate_pairs.py --update # update baseline
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCHESTRATOR_DIR = os.path.join(PROJECT_ROOT, "orchestrator")

# Assembled from parts so the markers below are the only literal occurrences.
_MARK = ">>>GENERATED-BASELINE"
_BEGIN = f"# {_MARK}-BEGIN<<<"
_END = f"# {_MARK}-END<<<"

# >>>GENERATED-BASELINE-BEGIN<<<
# BASELINE — auto-generated with --update.  DO NOT edit manually.
BASELINE: set[str] = {
    "agents/metrics.py",
    "agents/product_manager.py",
    "agents/rate_limiter.py",
    "analysis/assumption_gate.py",
    "analysis/metrics.py",
    "analysis/progress.py",
    "analysis/progress_writer.py",
    "analysis/progressive_output.py",
    "analysis/visualization.py",
    "appbuilder/assembler.py",
    "application/dashboard_bridge.py",
    "crosscutting/config.py",
    "design/design_to_code.py",
    "design/prompt_builder.py",
    "engine_core/stages/preflight.py",
    "events/triggers.py",
    "generators/diff_generator.py",
    "generators/image_generator.py",
    "generators/website_generator.py",
    "ide_backend/log_config.py",
    "infrastructure/cache_optimizer.py",
    "infrastructure/nexusscope/config.py",
    "infrastructure/reranker.py",
    "infrastructure/streaming.py",
    "integrations/tenancy.py",
    "learning/knowledge_graph.py",
    "learning/log_config.py",
    "meta/config.py",
    "meta/performance.py",
    "nexus_search/config.py",
    "nexus_search/models.py",
    "nexus_search/optimization/circuit_breaker.py",
    "nexus_search/optimization/reranker.py",
    "operations/feedback_loop.py",
    "operations/gradual_rollout.py",
    "product/product_manager.py",
    "project_mgmt/assembler.py",
    "quality/preflight.py",
    "quality/tdd_config.py",
    "reasoning/ara_integration.py",
    "safety/code_executor.py",
    "safety/red_team.py",
    "security/enhancer.py",
    "state_mgmt/capability_logger.py",
    "state_mgmt/session_lifecycle.py",
    "state_mgmt/session_watcher.py",
    "supervisor/models.py",
}
# >>>GENERATED-BASELINE-END<<<


def _has_definitions(path: str) -> bool:
    """True when the module body declares its own classes or functions.

    A module whose body is only a docstring plus imports is a re-export shim,
    so it cannot diverge from the module it re-exports.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (SyntaxError, OSError):
        # Unparseable files are not this gate's concern; other gates cover them.
        return False
    return any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in tree.body
    )


def current_pairs() -> set[str]:
    """Return subpackage modules duplicating a root module, both sides defining."""
    root_modules = {
        entry[:-3]
        for entry in os.listdir(ORCHESTRATOR_DIR)
        if entry.endswith(".py") and os.path.isfile(os.path.join(ORCHESTRATOR_DIR, entry))
    }
    pairs: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(ORCHESTRATOR_DIR):
        if dirpath == ORCHESTRATOR_DIR or "__pycache__" in dirpath:
            continue
        for name in filenames:
            if not name.endswith(".py") or name == "__init__.py":
                continue
            if name[:-3] not in root_modules:
                continue
            sub_path = os.path.join(dirpath, name)
            root_path = os.path.join(ORCHESTRATOR_DIR, name)
            if _has_definitions(sub_path) and _has_definitions(root_path):
                rel = os.path.relpath(sub_path, ORCHESTRATOR_DIR).replace(os.sep, "/")
                pairs.add(rel)
    return pairs


def update_baseline(pairs: set[str]) -> None:
    """Rewrite this file's baseline block in place."""
    script = os.path.abspath(__file__)
    with open(script, encoding="utf-8") as fh:
        source = fh.read()
    body = "".join(f'    "{p}",\n' for p in sorted(pairs))
    block = (
        f"{_BEGIN}\n"
        "# BASELINE — auto-generated with --update.  DO NOT edit manually.\n"
        f"BASELINE: set[str] = {{\n{body}}}\n"
        f"{_END}"
    )
    # count=1 — the markers appear again in this function's own source.
    new, n = re.subn(
        re.escape(_BEGIN) + ".*?" + re.escape(_END),
        lambda _m: block,
        source,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        print("ERROR: baseline block not found; file may be corrupt.", file=sys.stderr)
        raise SystemExit(2)
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(new)
    print(f"Baseline updated: {len(pairs)} duplicate pairs recorded.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Duplicate root/subpackage pair check.")
    parser.add_argument("--update", action="store_true", help="rewrite the baseline")
    parser.add_argument("--list", action="store_true", help="print current pairs")
    args = parser.parse_args()

    pairs = current_pairs()

    if args.update:
        update_baseline(pairs)
        return 0

    if args.list:
        for pair in sorted(pairs):
            print(pair)
        return 0

    added = sorted(pairs - BASELINE)
    removed = sorted(BASELINE - pairs)

    if added:
        print("FAIL: new duplicate root/subpackage pairs detected:", file=sys.stderr)
        for pair in added:
            print(
                f"  orchestrator/{pair}  duplicates  orchestrator/{os.path.basename(pair)}",
                file=sys.stderr,
            )
        print(
            "\nBoth sides define their own classes/functions, so a fix to one will not\n"
            "reach the other. Make one side a re-export shim, or run --update if this\n"
            "duplication is deliberate and reviewed.",
            file=sys.stderr,
        )
        return 1

    if removed:
        print(f"OK: {len(pairs)} duplicate pairs ({len(removed)} resolved since baseline).")
        print("Run --update to record the improvement.")
        return 0

    print(f"OK: {len(pairs)} duplicate pairs match baseline (no new duplication).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
