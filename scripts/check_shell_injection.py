#!/usr/bin/env python3
"""
Hunt T19 gate — no shell-interpreted command built from a non-literal.

Fails when shipped code under `orchestrator/` hands a *computed* command
string to a shell:

  * `asyncio.create_subprocess_shell(<not a literal>)`
  * `subprocess.*(..., shell=True)` with a non-literal command
  * `os.system(<not a literal>)`

A literal (or a join/concat of literals) is fine: nothing outside the file
can steer it. An f-string, a `%`/`+` expression, a name or a call is not —
that is how `nexus_search/server_manager.py` came to run an attacker-chosen
command from a caller-supplied compose path.

Sites reviewed and deliberately allowed are listed in ALLOWLIST with the
reason each is safe. Run with no arguments; exits 1 on any violation.
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN_DIR = ROOT / "orchestrator"

# path -> why a shell is legitimate there.
ALLOWLIST: dict[str, str] = {
    "orchestrator/tools/shell_tool.py": "the shell tool: running a shell is its entire purpose",
    "orchestrator/safety/sandbox_executor.py": (
        "runs the caller's own test command inside the sandbox; that is the API"
    ),
    "orchestrator/dev_server.py": (
        "commands come from a hardcoded ProjectType table; the only interpolated "
        "value is a port already validated as an int in 1..65535"
    ),
}


def _is_literal(node: ast.AST | None) -> bool:
    """True if the node can only ever be text written in this file."""
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_literal(node.left) and _is_literal(node.right)
    if isinstance(node, ast.IfExp):  # os.system("cls" if nt else "clear")
        return _is_literal(node.body) and _is_literal(node.orelse)
    if isinstance(node, ast.JoinedStr):  # f-string: literal only if it has no {}
        return all(isinstance(v, ast.Constant) for v in node.values)
    return False


def _callee(node: ast.Call) -> str:
    parts: list[str] = []
    cur: ast.AST = node.func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _first_arg(node: ast.Call) -> ast.AST | None:
    return node.args[0] if node.args else None


def violations(path: pathlib.Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node)
        shell_kw = next(
            (k for k in node.keywords if k.arg == "shell"),
            None,
        )
        uses_shell = shell_kw is not None and (
            isinstance(shell_kw.value, ast.Constant) and shell_kw.value.value is True
        )

        if name.endswith("create_subprocess_shell"):
            api = "create_subprocess_shell"
        elif name.endswith("system") and "os" in name:
            api = "os.system"
        elif uses_shell:
            api = "shell=True"
        else:
            continue

        if not _is_literal(_first_arg(node)):
            found.append((node.lineno, f"{api}: command is not a literal"))
    return found


def main() -> int:
    failures: list[str] = []
    for path in sorted(SCAN_DIR.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWLIST:
            continue
        for lineno, why in violations(path):
            failures.append(f"{rel}:{lineno}: {why}")

    if failures:
        print(
            "T19 gate: shell-interpreted command built from a non-literal.\n"
            "Pass argv to create_subprocess_exec/subprocess.run instead, or add "
            "the site to ALLOWLIST in scripts/check_shell_injection.py with a "
            "reason.\n",
            file=sys.stderr,
        )
        for line in failures:
            print(line, file=sys.stderr)
        return 1

    print(f"T19 gate: clean ({len(ALLOWLIST)} reviewed exception(s) allowlisted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
