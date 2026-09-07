"""No module may be shadowed by a same-named package (P3-SHADOW1).

Where both ``orchestrator/X.py`` and ``orchestrator/X/__init__.py`` exist, Python
resolves ``orchestrator.X`` to the **package**. The sibling module is then dead by
construction: no import statement can reach it, no matter what it contains.

The P3 sweep found seven of these. Five were deprecation re-export shims whose
warnings could therefore never fire — they were deleted, which is behaviour-
preserving by construction since nothing could ever have executed them. Two carry
real, *distinct* implementations (not duplicates of their packages) and are
awaiting a disposition decision; they are listed in PENDING_DISPOSITION below so
this gate can be enforced now rather than after that decision, with the remaining
debt explicit instead of silent.

A new collision must never be added: this test fails on anything not in that list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ORCHESTRATOR = Path(__file__).resolve().parents[2] / "orchestrator"

# All seven collisions found by P3-SHADOW1 are resolved, so this list is empty and
# the gate is now absolute. It exists only so a future collision can be recorded
# deliberately rather than silently: an entry belongs here only when the shadowed
# module holds an implementation that differs from its package, and it may only
# ever shrink.
#
# How the original seven were resolved:
#   five re-export shims  — deleted; their DeprecationWarnings could never fire
#   gateway.py            — deleted; byte-identical to integrations/gateway.py
#   agents.py             — moved to agents/pool.py; its TaskChannel/AgentPool
#                           were distinct, and engine.py had been silently
#                           binding TaskChannel to None because of the shadowing
PENDING_DISPOSITION: set[str] = set()


def _collisions() -> set[str]:
    return {
        path.stem
        for path in ORCHESTRATOR.glob("*.py")
        if (ORCHESTRATOR / path.stem / "__init__.py").exists()
    }


@pytest.mark.unit
def test_no_new_module_package_shadowing() -> None:
    unexpected = _collisions() - PENDING_DISPOSITION
    assert (
        not unexpected
    ), "module(s) shadowed by a same-named package — unimportable by construction:\n" + "\n".join(
        f"  orchestrator/{name}.py vs orchestrator/{name}/" for name in sorted(unexpected)
    )


@pytest.mark.unit
def test_pending_dispositions_are_still_real() -> None:
    """Keeps the allowlist honest: an entry that is resolved must be removed."""
    stale = PENDING_DISPOSITION - _collisions()
    assert not stale, (
        "PENDING_DISPOSITION lists collisions that no longer exist — delete these "
        f"entries so the gate tightens: {sorted(stale)}"
    )
