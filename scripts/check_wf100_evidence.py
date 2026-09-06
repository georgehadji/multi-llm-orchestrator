#!/usr/bin/env python3
"""
Hunt T22 gate — a WF-100 check must declare the evidence it reads.

`auditor.py` refuses to run a check whose evidence is absent:

    if check.requires - evidence.available:  ->  OUTSTANDING

That guarantee is only as good as each check's declaration. A check that
reads `ev.scripts` while declaring only MARKUP slips the gate, runs against
an empty string, and reports a definite verdict it never established.

T22 found exactly that in G7 (form abuse protection): `ev.markup + ev.scripts`
under `requires = {MARKUP}`. With the site's JavaScript uncollected the same
page flipped from PASS ("protection in place: captcha") to FAIL ("public forms
with no captcha, honeypot or rate limiting — they will be found by bots").

Reading undeclared evidence is not automatically wrong: A9 and D9 use
`ev.http` when it is there and fall back to files, and E8 falls back to
structured data and returns OUTSTANDING when it still cannot decide. Those
guarded reads are allowlisted with their reason. What must never happen
silently is an *unguarded* read of undeclared evidence.

Run with no arguments; exits 1 on any unlisted mismatch.
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHECKS = ROOT / "orchestrator" / "generators" / "wf100" / "checks.py"

# SiteEvidence attribute -> the Evidence kind reading it implies.
ATTR_TO_KIND = {
    "pages": "MARKUP",
    "markup": "MARKUP",
    "styles": "STYLES",
    "scripts": "SCRIPTS",
    "assets": "ASSETS",
    "files": "ASSETS",
    "http": "HTTP",
    "record": "RECORD",
}
# checks.py helpers that read pages on the caller's behalf.
PAGE_HELPERS = {"_content_pages", "_images"}

# check id -> why reading undeclared evidence is correct there.
ALLOWLIST: dict[str, str] = {
    "A9": "reads ev.http under `if ev.http is not None`, else falls back to 404.html/_redirects",
    "D9": "reads ev.http under `if ev.http is not None`, else decides from page markup alone",
    "E8": "reads ev.record under `if record else`, falls back to structured data, and returns "
    "OUTSTANDING when no locality can be established either way",
    "G7": "reads ev.scripts, and since T22 returns OUTSTANDING rather than FAIL when the markup "
    "loads scripts the auditor did not collect. Declaring SCRIPTS instead would be wrong: a "
    "site with no JavaScript has no SCRIPTS evidence and G7 decides those from markup fine",
}


def _reads(node: ast.AST) -> set[str]:
    kinds: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in ATTR_TO_KIND:
            kinds.add(ATTR_TO_KIND[child.attr])
        if isinstance(child, ast.Call) and getattr(child.func, "id", "") in PAGE_HELPERS:
            kinds.add("MARKUP")
    return kinds


def mismatches() -> list[tuple[str, list[str], list[str]]]:
    sys.path.insert(0, str(ROOT))
    from orchestrator.generators.wf100.standard import get

    tree = ast.parse(CHECKS.read_text(encoding="utf-8"))
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "implements"):
                continue
            cid = dec.args[0].value
            declared = {e.name for e in get(cid).requires}
            missing = _reads(node) - declared
            if missing:
                out.append((cid, sorted(missing), sorted(declared)))
    return out


def main() -> int:
    found = mismatches()
    unlisted = [m for m in found if m[0] not in ALLOWLIST]

    if unlisted:
        print(
            "T22 gate: check(s) reading evidence they do not declare.\n"
            "auditor.py gates on `requires`, so an undeclared read runs against "
            "empty data and reports a verdict it did not establish. Declare the "
            "evidence, guard the read and say so, or add it to ALLOWLIST with a "
            "reason.\n",
            file=sys.stderr,
        )
        for cid, missing, declared in unlisted:
            print(
                f"  {cid}: reads {', '.join(missing)} but declares "
                f"{', '.join(declared) or '(nothing)'}",
                file=sys.stderr,
            )
        return 1

    print(f"T22 gate: OK ({len(found)} guarded read(s) allowlisted, 0 undeclared).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
