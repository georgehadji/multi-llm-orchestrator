"""
check_subprocess_cleanup.py — CI guard script.

Verifies that every ``asyncio.create_subprocess_exec`` call in the
verification module has an associated ``proc.kill()`` (or ``proc.terminate()``)
in its ``except asyncio.TimeoutError`` handler.

This prevents recurrence of D1 (subprocess leak on timeout).

Usage:
    python scripts/check_subprocess_cleanup.py
    # Exits 0 if clean, 1 if violations found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _find_subprocess_calls(tree: ast.AST) -> list[ast.AST]:
    """Return every ``await asyncio.create_subprocess_exec(...)`` node."""
    calls: list[ast.AST] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Await(self, node: ast.Await) -> None:
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                attr = node.value.func
                if (
                    attr.attr == "create_subprocess_exec"
                    and isinstance(attr.value, ast.Name)
                    and attr.value.id == "asyncio"
                ):
                    calls.append(node)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return calls


def _find_timeout_handlers(tree: ast.AST) -> list[ast.ExceptHandler]:
    """Return every ``except asyncio.TimeoutError`` handler."""
    handlers: list[ast.ExceptHandler] = []

    class _Visitor(ast.NodeVisitor):
        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if isinstance(node.type, ast.Attribute):
                attr = node.type
                if (
                    attr.attr == "TimeoutError"
                    and isinstance(attr.value, ast.Name)
                    and attr.value.id == "asyncio"
                ):
                    handlers.append(node)
            elif isinstance(node.type, ast.Name) and node.type.id in ("TimeoutError",):
                handlers.append(node)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return handlers


def _handler_contains_kill(handler: ast.ExceptHandler, var_name: str = "proc") -> bool:
    """Return True if *handler* body calls ``<var_name>.kill()`` (with or without await)."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("kill", "terminate") and isinstance(node.func.value, ast.Name):
                if node.func.value.id == var_name:
                    return True
    return False


def _handler_contains_wait(handler: ast.ExceptHandler, var_name: str = "proc") -> bool:
    """Return True if *handler* body calls ``<var_name>.wait()`` (directly or wrapped)."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "wait" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == var_name:
                    return True
    return False


def check_file(filepath: str) -> list[str]:
    """Run all checks on *filepath*; return list of violation messages."""
    path = Path(filepath)
    if not path.exists():
        return [f"File not found: {filepath}"]

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f" SyntaxError: {exc}"]

    # Assume the subprocess variable is named ``proc`` (the convention)
    var_name = "proc"

    errors: list[str] = []
    subprocess_calls = _find_subprocess_calls(tree)
    timeout_handlers = _find_timeout_handlers(tree)

    if not subprocess_calls:
        return []  # Nothing to check — clean

    for idx, call_node in enumerate(subprocess_calls):
        # Check the function containing this call against ALL timeout handlers
        func_def = _enclosing_function(tree, call_node)
        if func_def is None:
            continue

        relevant_handlers = [h for h in timeout_handlers if _node_in_function(h, func_def)]

        if not relevant_handlers:
            # No timeout handler in this function — nothing to check.
            # The subprocess call may use a default timeout or no timeout.
            continue

        for handler in relevant_handlers:
            # Skip handlers that live inside another timeout handler
            # (e.g., the wait_for wrapper handler).
            if _is_nested_handler(handler, timeout_handlers):
                continue
            has_kill = _handler_contains_kill(handler, var_name)
            has_wait = _handler_contains_wait(handler, var_name)

            if not has_kill:
                errors.append(
                    f"{filepath} L{handler.lineno}: except asyncio.TimeoutError "
                    f"missing {var_name}.kill() or {var_name}.terminate() "
                    f"(see D1: subprocess leak on timeout)"
                )
            if not has_wait:
                errors.append(
                    f"{filepath} L{handler.lineno}: except asyncio.TimeoutError "
                    f"missing await {var_name}.wait() to reap the process"
                )

    return errors


def _enclosing_function(tree: ast.AST, node: ast.AST) -> ast.FunctionDef | None:
    """Find the FunctionDef or AsyncFunctionDef enclosing *node*."""
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _node_in_function(node, parent):
                return parent
    return None


def _node_in_function(node: ast.AST, func: ast.FunctionDef) -> bool:
    """Return True if *node* is a descendant of *func*."""
    return any(child is node for child in ast.walk(func))


def _is_nested_handler(handler: ast.ExceptHandler, all_handlers: list[ast.ExceptHandler]) -> bool:
    """Return True if *handler* lives inside another handler's body."""
    for other in all_handlers:
        if other is handler:
            continue
        if _node_in_function(handler, other):
            return True
    return False


def main() -> int:
    base = Path(__file__).resolve().parent.parent
    targets = [
        base / "orchestrator" / "infrastructure" / "verification_checks.py",
    ]

    all_errors: list[str] = []
    for target in targets:
        if target.exists():
            all_errors.extend(check_file(str(target)))

    if all_errors:
        print("SUBPROCESS CLEANUP VIOLATIONS:")
        for err in all_errors:
            print(f"  {err}")
        print()
        print("Every create_subprocess_exec must have a paired kill() + wait()")
        print("in its except asyncio.TimeoutError handler.")
        print("See D1: https://github.com/georgehadji/multi-llm-orchestrator/commit/10403400")
        return 1

    print("OK — all subprocess calls have cleanup paired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
