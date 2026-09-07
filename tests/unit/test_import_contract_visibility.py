"""Non-vacuity gate for the .importlinter contracts.

Why this exists
---------------
Until 2026-09-07 ``orchestrator/infrastructure/`` had no ``__init__.py``. Python
imported it fine (PEP 420 implicit namespace package), but grimp — the static
import-graph builder underneath import-linter — never registered the package or
anything beneath it. Four of the five contracts name ``orchestrator.infrastructure``
in ``forbidden_modules``, so none of them could find an edge into it, violating or
not. They reported KEPT for their entire existence, and CI reported the
architecture gate green.

Proven by planting ``orchestrator/domain/__zz_probe.py`` containing
``from orchestrator.infrastructure.llm_client import LLMClient`` — a textbook
breach of Domain-purity's stated rule — and watching the contract report KEPT.

The failure is silent in both directions: nothing about a passing ``lint-imports``
run distinguishes "no violations exist" from "the checker cannot see the modules
in question". Adding an ``__init__.py`` to any package a contract names is enough
to switch a contract off, permanently and invisibly.

What this gate checks
---------------------
Every module named in any contract's ``source_modules`` or ``forbidden_modules``
must be present in the graph import-linter actually builds. That is the exact
precondition for a contract to be capable of failing, and it is the property that
was missing. It generalises past the one package that caused the bug: any future
contract naming a namespace package fails here instead of passing vacuously.

It deliberately does not assert that contracts pass — ``lint-imports`` in CI does
that, and it is the thing this gate keeps honest.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CONFIG = Path(__file__).resolve().parents[2] / ".importlinter"


@pytest.fixture(scope="module")
def graph():
    import grimp

    return grimp.build_graph("orchestrator")


def _declared_modules(field: str) -> list[tuple[str, str]]:
    """(contract name, module) pairs for every module listed under `field`."""
    parser = configparser.ConfigParser()
    # Explicit encoding: configparser otherwise decodes with the platform
    # default, and on a cp1252 Windows shell that raises UnicodeDecodeError
    # here at collection time, aborting the entire unit suite.
    parser.read(_CONFIG, encoding="utf-8")

    pairs = []
    for section in parser.sections():
        if not section.startswith("importlinter:contract:"):
            continue
        for line in parser[section].get(field, "").splitlines():
            module = line.strip()
            # Wildcard expressions match sets of modules, not one named module,
            # so "is this exact name in the graph" does not apply to them.
            if module and not module.startswith("#") and "*" not in module:
                pairs.append((section.rpartition(":")[2], module))
    return pairs


@pytest.mark.parametrize(
    ("contract", "module"),
    _declared_modules("forbidden_modules"),
    ids=lambda v: v,
)
def test_forbidden_module_is_visible_to_the_import_graph(contract, module, graph):
    assert module in graph.modules, (
        f"Contract {contract!r} forbids imports of {module!r}, but that module is "
        f"absent from the graph import-linter builds — so the contract cannot "
        f"fail, whatever the code does. Usual cause: the package has no "
        f"__init__.py and is an implicit namespace package, invisible to grimp."
    )


@pytest.mark.parametrize(
    ("contract", "module"),
    _declared_modules("source_modules"),
    ids=lambda v: v,
)
def test_source_module_is_visible_to_the_import_graph(contract, module, graph):
    assert module in graph.modules, (
        f"Contract {contract!r} constrains {module!r}, but that module is absent "
        f"from the graph import-linter builds, so nothing under it is actually "
        f"checked. Usual cause: no __init__.py (implicit namespace package)."
    )
