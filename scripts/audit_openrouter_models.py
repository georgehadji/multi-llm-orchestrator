#!/usr/bin/env python3
"""
OpenRouter Model Registry Audit
===============================
Cross-checks every ``provider/model`` string literal referenced in the
orchestrator's model registry against the *live* OpenRouter catalogue
(https://openrouter.ai/api/v1/models, keyless GET) and reports any id that is
no longer available.

Why this exists
---------------
The registry drifts from the live catalogue over time: providers rename ids,
drop models, or bump versions. Stale ids surface at runtime as 404/400 crashes
and "Unknown model X - allowing through" warnings. This script is the single
audit that catches that drift, and is wired into CI.

Runtime-resolvable ids
----------------------
A few ids are absent from the public ``/models`` snapshot yet still resolve at
runtime because OpenRouter normalizes them server-side. The clearest case is
Anthropic's API-style hyphenated ids (``anthropic/claude-opus-4-6`` →
``anthropic/claude-opus-4.6``). ``normalize_for_lookup`` encodes those rules so
the audit does not flag valid ids as dead. Verified with real cheap calls.

Hexagonal note
--------------
This network fetch lives here in ``scripts/`` (a driving adapter), never in
``orchestrator/models.py`` (pure data). The orchestrator core stays I/O-free.

Usage
-----
    python scripts/audit_openrouter_models.py            # fetch live + audit
    python scripts/audit_openrouter_models.py --snapshot snap.json   # offline
    python scripts/audit_openrouter_models.py --save-snapshot snap.json
    python scripts/audit_openrouter_models.py --json     # machine-readable

Exit code is non-zero when any referenced id is dead — suitable for CI.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files whose string literals are audited. These are the registry surfaces that
# reference concrete OpenRouter model ids.
DEFAULT_TARGET_FILES = [
    REPO_ROOT / "orchestrator" / "models.py",
    REPO_ROOT / "orchestrator" / "domain" / "model_registry.py",
    REPO_ROOT / "orchestrator" / "phase_aware_models.py",
]

# Matches "provider/model[:variant]" string literals.
_ID_PATTERN = re.compile(r"""["']([a-z0-9-]+/[a-zA-Z0-9._:-]+)["']""")

# Anthropic API-style hyphenated ids are absent from the /models snapshot but
# resolve at runtime via OpenRouter server-side normalization
# (claude-opus-4-6 -> claude-opus-4.6). Verified with real calls 2026-06-20.
_ANTHROPIC_HYPHEN = re.compile(r"^anthropic/claude-(opus|sonnet|haiku)-(\d+)-(\d+)$")

# Video-generation models are served via OpenRouter's generation endpoint and
# billed per-second, so they never appear in the /api/v1/models chat catalogue.
# They DO resolve at /api/v1/models/<id>/endpoints (verified 2026-06-23). The
# audit treats these as acceptable and `verify_runtime_only_ids` probes the
# endpoints route to keep this allowlist honest (see the integration test).
RUNTIME_ONLY_IDS = frozenset(
    {
        "openai/sora-2-pro",
        "google/veo-3.1",
        "google/veo-3.1-fast",
        "google/veo-3.1-lite",
        "kwaivgi/kling-v3.0-pro",
        "kwaivgi/kling-v3.0-std",
        "kwaivgi/kling-video-o1",
        "minimax/hailuo-2.3",
        "bytedance/seedance-2.0",
        "bytedance/seedance-2.0-fast",
        "bytedance/seedance-1-5-pro",
        "alibaba/wan-2.7",
        "alibaba/wan-2.6",
        "x-ai/grok-imagine-video",
    }
)


def normalize_for_lookup(model_id: str) -> str:
    """Return the canonical live id a referenced id resolves to.

    Strips a trailing OpenRouter routing variant (``:nitro`` / ``:floor`` /
    ``:exacto`` / ``:free`` …) and applies Anthropic hyphen→dot normalization.
    """
    base = model_id.split(":", 1)[0]
    m = _ANTHROPIC_HYPHEN.match(base)
    if m:
        return f"anthropic/claude-{m.group(1)}-{m.group(2)}.{m.group(3)}"
    return base


def fetch_live_ids(url: str = OPENROUTER_MODELS_URL) -> set[str]:
    """Fetch the live catalogue and return the set of available model ids."""
    req = urllib.request.Request(url, headers={"User-Agent": "orchestrator-audit"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted URL)
        data = json.loads(resp.read().decode("utf-8"))
    return _ids_from_catalog(data)


def _ids_from_catalog(data: dict) -> set[str]:
    ids: set[str] = set()
    for entry in data.get("data", []):
        if entry.get("id"):
            ids.add(entry["id"])
        if entry.get("canonical_slug"):
            ids.add(entry["canonical_slug"])
    return ids


def load_snapshot_ids(path: Path) -> set[str]:
    """Load model ids from a previously saved /models snapshot file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return _ids_from_catalog(data)


def extract_referenced_ids(files: list[Path]) -> dict[str, set[str]]:
    """Return {model_id: {filenames referencing it}} from the target files."""
    refs: dict[str, set[str]] = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        for match in _ID_PATTERN.finditer(text):
            refs.setdefault(match.group(1), set()).add(f.name)
    return refs


def load_known_deprecated() -> dict[str, str]:
    """Return ModelRegistry.UNAVAILABLE_MODELS (deprecated id -> replacement).

    Parses the source file via AST instead of importing the module, so it works
    even when orchestrator's full import chain fails (e.g. missing optional deps
    in a minimal CI environment).
    """
    import ast

    src_path = REPO_ROOT / "orchestrator" / "domain" / "model_registry.py"
    try:
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name == "ModelRegistry"):
                continue
            for item in node.body:
                if not isinstance(item, ast.Assign):
                    continue
                if not any(
                    isinstance(t, ast.Name) and t.id == "UNAVAILABLE_MODELS" for t in item.targets
                ):
                    continue
                if isinstance(item.value, ast.Dict):
                    result: dict[str, str] = {}
                    for k, v in zip(item.value.keys, item.value.values):
                        if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                            result[str(k.value)] = str(v.value)
                    return result
    except Exception:  # pragma: no cover
        pass
    return {}


def find_dead_ids(
    refs: dict[str, set[str]],
    live_ids: set[str],
    known_deprecated: dict[str, str] | None = None,
    runtime_only: frozenset[str] = RUNTIME_ONLY_IDS,
) -> dict[str, set[str]]:
    """Return referenced ids that are neither live nor documented-deprecated.

    An id is acceptable when it is in the live catalogue, resolves there via a
    runtime normalizer, is a key in ``known_deprecated`` (a deliberately
    recorded dead id with a live replacement), or is a ``runtime_only`` id
    (a generation model absent from /models but reachable via /endpoints).
    """
    known = known_deprecated or {}
    dead: dict[str, set[str]] = {}
    for model_id, sources in refs.items():
        if model_id in live_ids:
            continue
        if normalize_for_lookup(model_id) in live_ids:
            continue
        if model_id in known:
            continue
        if model_id in runtime_only:
            continue
        dead[model_id] = sources
    return dead


def verify_runtime_only_ids(ids: frozenset[str] = RUNTIME_ONLY_IDS) -> dict[str, str]:
    """Probe each runtime-only id at /endpoints; return {id: reason} for failures.

    Keeps the RUNTIME_ONLY_IDS allowlist honest — a typo'd or retired video id
    that no longer resolves is reported instead of being silently trusted.
    """
    failures: dict[str, str] = {}
    for mid in sorted(ids):
        url = f"https://openrouter.ai/api/v1/models/{mid}/endpoints"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "orchestrator-audit"})
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                if resp.status != 200:
                    failures[mid] = f"HTTP {resp.status}"
        except Exception as exc:  # noqa: BLE001
            failures[mid] = repr(exc)
    return failures


def find_stale_replacements(known_deprecated: dict[str, str], live_ids: set[str]) -> dict[str, str]:
    """Return deprecated→replacement entries whose replacement is itself dead."""
    return {
        dead_id: repl
        for dead_id, repl in known_deprecated.items()
        if repl not in live_ids and normalize_for_lookup(repl) not in live_ids
    }


def run_audit(files: list[Path], live_ids: set[str]) -> dict[str, set[str]]:
    refs = extract_referenced_ids(files)
    return find_dead_ids(refs, live_ids, load_known_deprecated())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Audit against a saved /models snapshot instead of fetching live.",
    )
    parser.add_argument(
        "--save-snapshot",
        type=Path,
        help="Fetch live catalogue and write it to this path (then audit).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.snapshot:
        live_ids = load_snapshot_ids(args.snapshot)
        source = f"snapshot {args.snapshot}"
    else:
        try:
            req = urllib.request.Request(
                OPENROUTER_MODELS_URL, headers={"User-Agent": "orchestrator-audit"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8")
        except Exception as exc:  # network failure should not crash CI ambiguously
            print(f"ERROR: could not fetch OpenRouter catalogue: {exc}", file=sys.stderr)
            return 2
        if args.save_snapshot:
            args.save_snapshot.write_text(raw, encoding="utf-8")
        live_ids = _ids_from_catalog(json.loads(raw))
        source = "live OpenRouter catalogue"

    known_deprecated = load_known_deprecated()
    refs = extract_referenced_ids(DEFAULT_TARGET_FILES)
    dead = find_dead_ids(refs, live_ids, known_deprecated)
    stale_replacements = find_stale_replacements(known_deprecated, live_ids)

    if args.json:
        print(
            json.dumps(
                {
                    "source": source,
                    "live_model_count": len(live_ids),
                    "referenced_id_count": len(refs),
                    "known_deprecated_count": len(known_deprecated),
                    "dead": {k: sorted(v) for k, v in sorted(dead.items())},
                    "stale_replacements": dict(sorted(stale_replacements.items())),
                },
                indent=2,
            )
        )
    else:
        print(
            f"Audited {len(refs)} referenced ids against {source} "
            f"({len(live_ids)} live ids; {len(known_deprecated)} documented "
            f"deprecated)."
        )
        if not dead and not stale_replacements:
            print("OK: every referenced model id is live or a documented redirect.")
        if dead:
            print(f"\nFOUND {len(dead)} DEAD model id(s):\n")
            for model_id in sorted(dead):
                print(f"  {model_id}  <- {', '.join(sorted(dead[model_id]))}")
            print(
                "\nRemap each dead id to a live replacement (add it to "
                "ModelRegistry.UNAVAILABLE_MODELS) and re-run."
            )
        if stale_replacements:
            print(
                f"\nFOUND {len(stale_replacements)} UNAVAILABLE_MODELS entry(ies) "
                "whose replacement is itself dead:\n"
            )
            for dead_id, repl in sorted(stale_replacements.items()):
                print(f"  {dead_id} -> {repl}  (replacement not live)")

    return 1 if (dead or stale_replacements) else 0


if __name__ == "__main__":
    raise SystemExit(main())
