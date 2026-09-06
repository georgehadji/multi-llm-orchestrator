#!/usr/bin/env python3
"""
Hunt T20 gate — no NEW config field that nothing reads.

Every field declared on a `BaseSettings` class in
`orchestrator/crosscutting/config.py` is a promise: set the matching
`ORCH_*` environment variable and something changes. `extra="ignore"` means
pydantic accepts the variable silently, so a field nothing reads produces a
setting that appears to work and does nothing.

T20 found 20 such fields, including `knowledge_rerank_enabled` — declared,
documented in a plan, backed by a fully implemented and tested two-stage
reranker, and read by no module. This gate freezes that list so the count can
only go down.

A field counts as read if its name appears anywhere under `orchestrator/`
outside `config.py`, as `.name` or as a quoted string. That is deliberately
generous: the gate exists to stop *new* dead settings, not to adjudicate
every existing one.

Run with no arguments; `--update` rewrites the baseline below.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT / "orchestrator" / "crosscutting" / "config.py"
PKG = ROOT / "orchestrator"

_MARK_OPEN = ">>>GENERATED-BASELINE"
_MARK_CLOSE = "<<<GENERATED-BASELINE"

# >>>GENERATED-BASELINE
BASELINE: frozenset[str] = frozenset(
    {
        "audit_log_path",
        "bilevel_level15_enabled",
        "bilevel_tabu_enabled",
        "cache_home",
        "cache_ttl_hours",
        "compression_model",
        "dashboard_host",
        "dashboard_port",
        "default_budget_usd",
        "default_timeout_seconds",
        "mcp_host",
        "mcp_http_mode",
        "mcp_port",
        "semantic_cache_threshold",
        "use_embedding_cache",
        "use_json_schema_responses",
        "use_model_variants",
        "use_native_fallbacks",
        "use_provider_sorting",
        "use_streaming",
    }
)
# <<<GENERATED-BASELINE


def declared_fields() -> dict[str, str]:
    """field name -> declaring class."""
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                out.setdefault(stmt.target.id, node.name)
    return out


def unread_fields() -> dict[str, str]:
    blob = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted(PKG.rglob("*.py"))
        if p != CONFIG
    )
    dead = {}
    for name, cls in declared_fields().items():
        attr = re.search(rf"\.{re.escape(name)}\b", blob)
        quoted = re.search(rf"""["']{re.escape(name)}["']""", blob)
        if not attr and not quoted:
            dead[name] = cls
    return dead


def update_baseline(dead: dict[str, str]) -> None:
    body = "\n".join(f'        "{n}",' for n in sorted(dead))
    block = (
        f"# {_MARK_OPEN}\nBASELINE: frozenset[str] = frozenset(\n"
        f"    {{\n{body}\n    }}\n)\n# {_MARK_CLOSE}"
    )
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    pattern = re.compile(rf"# {_MARK_OPEN}\n.*?\n# {_MARK_CLOSE}", re.DOTALL)
    new, n = pattern.subn(lambda _m: block, src, count=1)
    if n != 1:
        print(f"refusing to write: matched baseline block {n} times", file=sys.stderr)
        raise SystemExit(2)
    pathlib.Path(__file__).write_text(new, encoding="utf-8")
    print(f"baseline updated: {len(dead)} unread field(s)")


def main() -> int:
    dead = unread_fields()

    if "--update" in sys.argv:
        update_baseline(dead)
        return 0

    added = sorted(set(dead) - BASELINE)
    if added:
        print(
            "T20 gate: new config field(s) that nothing reads.\n"
            "A declared setting is a promise that setting ORCH_<NAME> does "
            "something. Wire it to a consumer, or don't declare it.\n",
            file=sys.stderr,
        )
        for name in added:
            print(f"  {dead[name]}.{name}", file=sys.stderr)
        return 1

    removed = sorted(BASELINE - set(dead))
    total = len(declared_fields())
    if removed:
        print(f"{len(removed)} baseline field(s) now wired: {', '.join(removed)}")
        print("Run with --update to shrink the baseline.")
    print(f"T20 gate: OK ({len(dead)}/{total} declared fields unread, baseline {len(BASELINE)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
