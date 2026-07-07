#!/usr/bin/env python3
"""
Config <-> Model-enum drift checker for the Multi-LLM Orchestrator.

WHY THIS EXISTS
---------------
orchestrator/models.py builds ROUTING_TABLE / COST_TABLE / FALLBACK_CHAIN from
orchestrator/config/{routing,costs,fallbacks}.json with guards like:

    {Model(k): v for k, v in data.items() if k in Model._value2member_map_}

Any JSON key (or routing/fallback *value*) that does not EXACTLY equal a
Model enum value is SILENTLY DROPPED -- no error, no log. The symptom is a
model that "mysteriously" never gets routed to, or a fallback that never
fires. This has bitten the repo repeatedly (2026-06-23 fixes, ab17b5f4).

WHAT IT CHECKS
--------------
Direction A (HARD DRIFT -> exit 1):
  - costs.json keys not in Model enum values
  - routing.json keys not in TaskType enum values
  - routing.json list entries not in Model enum values
  - fallbacks.json keys OR values not in Model enum values

Direction B (informational, printed but non-fatal by default):
  - Model enum values with no costs.json entry (model exists but is uncosted)
  - TaskType enum values with no routing.json entry

Pass --strict to also fail on Direction B.

IMPLEMENTATION NOTE
-------------------
Enum values are extracted by AST-parsing orchestrator/models.py (stdlib only,
no import of the orchestrator package). This keeps the check working even if
the package import chain is broken -- which is exactly when you need it.

Usage:
    python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py
    python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py --strict

Exit codes: 0 = no drift, 1 = drift found, 2 = could not run (missing files).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

# Repo root = 5 levels up from this file (.claude/skills/<skill>/scripts/).
REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_PY = REPO_ROOT / "orchestrator" / "models.py"
CONFIG_DIR = REPO_ROOT / "orchestrator" / "config"


def extract_enum_values(source: str, class_name: str) -> set[str]:
    """Return the set of string values of an Enum class via AST (no import)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.Assign) and isinstance(item.value, ast.Constant):
                    if isinstance(item.value.value, str):
                        values.add(item.value.value)
            return values
    return set()


def load_json(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on enum values missing from costs.json / routing.json (Direction B).",
    )
    args = parser.parse_args()

    if not MODELS_PY.exists():
        print(f"ERROR: {MODELS_PY} not found", file=sys.stderr)
        return 2

    src = MODELS_PY.read_text(encoding="utf-8")
    model_values = extract_enum_values(src, "Model")
    tasktype_values = extract_enum_values(src, "TaskType")
    if not model_values or not tasktype_values:
        print("ERROR: could not extract Model/TaskType enum values via AST", file=sys.stderr)
        return 2

    costs = load_json("costs.json")
    routing = load_json("routing.json")
    fallbacks = load_json("fallbacks.json")

    hard_drift: list[str] = []

    # -- Direction A: config entries that models.py will silently drop --------
    for k in costs:
        if k not in model_values:
            hard_drift.append(f"costs.json key not a Model value (SILENTLY DROPPED): {k!r}")

    for k, v in routing.items():
        if k not in tasktype_values:
            hard_drift.append(f"routing.json key not a TaskType value (SILENTLY DROPPED): {k!r}")
        for m in v:
            if m not in model_values:
                hard_drift.append(
                    f"routing.json[{k!r}] entry not a Model value (SILENTLY DROPPED): {m!r}"
                )

    for k, v in fallbacks.items():
        if k not in model_values:
            hard_drift.append(f"fallbacks.json key not a Model value (SILENTLY DROPPED): {k!r}")
        if v not in model_values:
            hard_drift.append(
                f"fallbacks.json[{k!r}] value not a Model value (SILENTLY DROPPED): {v!r}"
            )

    # -- Direction B: enum entries with no config coverage --------------------
    uncosted = sorted(model_values - set(costs))
    unrouted_tasks = sorted(tasktype_values - set(routing))

    print(
        f"Checked {len(costs)} cost keys, {len(routing)} routing keys, "
        f"{len(fallbacks)} fallback pairs against {len(model_values)} Model values "
        f"and {len(tasktype_values)} TaskType values."
    )

    if hard_drift:
        print(f"\nHARD DRIFT — {len(hard_drift)} config entr(ies) silently dropped by models.py:")
        for line in hard_drift:
            print(f"  {line}")
        print("\nFix: make the JSON key/value EXACTLY equal the Model/TaskType enum .value.")
    else:
        print("\nOK: every config key/value maps to an enum member (no silent drops).")

    if uncosted:
        print(f"\nINFO: {len(uncosted)} Model value(s) have no costs.json entry:")
        for m in uncosted:
            print(f"  {m}")
    if unrouted_tasks:
        print(f"\nINFO: {len(unrouted_tasks)} TaskType value(s) have no routing.json entry:")
        for t in unrouted_tasks:
            print(f"  {t}")

    if hard_drift:
        return 1
    if args.strict and (uncosted or unrouted_tasks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
