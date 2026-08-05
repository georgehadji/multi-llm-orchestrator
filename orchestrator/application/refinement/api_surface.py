"""Public API surface extractor (Phase 6, E-11 rule 6).

AST-based: module-level and class-level public symbols with their
signatures. Refinement candidates that add, remove, or re-sign a public
symbol are rejected by the API-surface freeze — refinement restructures
internals, it does not redesign interfaces (plan §3.4.3 invariant 5).
"""

from __future__ import annotations

import ast


def _func_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Compact signature: name(arg1, arg2=..., *args, **kwargs)."""
    args = node.args
    parts: list[str] = []
    pos = args.posonlyargs + args.args
    defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(pos, defaults):
        text = arg.arg
        if default is not None:
            try:
                text += f"={ast.unparse(default)}"
            except Exception:  # pragma: no cover
                text += "=..."
        parts.append(text)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if default is not None:
            try:
                text += f"={ast.unparse(default)}"
            except Exception:  # pragma: no cover
                text += "=..."
        parts.append(text)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}{node.name}({', '.join(parts)})"


def extract_api_surface(source: str) -> dict[str, str]:
    """Return the public API surface: symbol name -> signature/kind.

    Includes module-level functions, classes (with their public methods)
    and module-level public assignments (constants).

    Args:
        source: Python module source.

    Returns:
        Mapping of ``"name"`` -> signature for functions, ``"Class"`` ->
        ``"class"``, ``"Class.method"`` -> method signature, and constants
        -> ``"const"``. Empty on syntax error.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    surface: dict[str, str] = {}

    def _is_public(name: str) -> bool:
        return not name.startswith("_")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            surface[node.name] = _func_signature(node)
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            surface[node.name] = "class"
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(
                    item.name
                ):
                    surface[f"{node.name}.{item.name}"] = _func_signature(item)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if targets and isinstance(targets[0], ast.Name) and _is_public(targets[0].id):
                surface[targets[0].id] = "const"

    return surface


def api_surface_equal(before: dict[str, str], after: dict[str, str]) -> bool:
    """True when two API surfaces are identical (freeze check)."""
    return before == after
