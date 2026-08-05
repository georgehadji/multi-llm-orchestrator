"""Stub implementation synthesizer for the E-2 RED-gate.

Given a test module's source, produce a stub implementation module where
every symbol the tests reference from the module under test raises
``NotImplementedError``. Any test that **passes** against this stub is
vacuous — it asserts nothing about a real implementation, and may not be
trusted as an oracle.

Pure AST, stdlib only (Contract 1). Python-focused; other frameworks are
out of scope for synthesis (the RED-gate logs and proceeds for them).
"""

from __future__ import annotations

import ast

_STDLIB_MODULES = {
    "os",
    "sys",
    "typing",
    "json",
    "re",
    "collections",
    "pathlib",
    "datetime",
    "itertools",
    "functools",
    "math",
    "random",
    "pytest",
    "unittest",
    "dataclasses",
    "enum",
    "time",
}


def _referenced_symbols(test_tree: ast.AST, module: str) -> set[str]:
    """Return the names the test source references from *module*.

    Handles ``from main import add``, ``import main``, and ``main.add(...)``
    attribute accesses. Stdlib/framework imports are never treated as
    implementation symbols.
    """
    names: set[str] = set()
    for node in ast.walk(test_tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == module:
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name)
            elif node.module and node.module.split(".")[0] not in _STDLIB_MODULES:
                # Import from a sibling module (multi-file task): synthesize
                # the imported names so the stub module stays importable.
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == module:
                names.add(node.attr)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    names.add(alias.asname or module)
    return names


def synthesize_stub(test_code: str, module: str = "main") -> str:
    """Return stub module source for every symbol referenced from *module*.

    Args:
        test_code: The test module source to analyze.
        module: Name of the implementation module under test (default
            ``main`` — the workspace convention).

    Returns:
        Python source defining each referenced symbol as a callable that
        raises ``NotImplementedError``. Constants referenced without a call
        (e.g. ``assert main.VERSION == "1.0"``) compare unequal to any real
        value, so such tests fail against the stub — exactly the point.
    """
    tree = ast.parse(test_code)
    references = _referenced_symbols(tree, module)

    blocks: list[str] = []
    for name in sorted(references):
        blocks.append(
            f"def {name}(*args, **kwargs):\n" f"    raise NotImplementedError({name!r})\n"
        )
    if not blocks:
        # No references: synthesize a module that at least imports cleanly
        # so the RED execution yields a collection verdict, not a crash.
        blocks.append(
            "# No implementation symbols referenced by the test suite.\n"
            "def _no_impl_symbols():\n"
            "    raise NotImplementedError('_no_impl_symbols')\n"
        )
    return "\n".join(blocks)


def is_vacuous_test(test_code: str, passed_node_ids: set[str]) -> set[str]:
    """Return the subset of *passed_node_ids* that are genuine test nodes.

    The RED-gate runs the suite against the stub; node_ids that executed
    (start with the test file prefix and match a ``test_*`` function) and
    passed are vacuous.
    """
    return {nid for nid in passed_node_ids if nid.split("::")[-1].startswith("test_")}
