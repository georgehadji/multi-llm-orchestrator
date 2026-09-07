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

# Known collisions awaiting a wire-or-delete decision. Each holds an
# implementation that differs from its shadowing package, so deleting it would
# lose code rather than remove a duplicate:
#   gateway.py — APIGateway/APIRequest/APIResponse/RateLimitInfo, against the
#                package's OrchestratorGateway/GatewaySession
#   agents.py  — TaskChannel/AgentPool, against the package's
#                AgentBase/AgentOrchestrator framework
# Removing an entry here (by resolving it) must never be accompanied by adding one.
PENDING_DISPOSITION = {"gateway", "agents"}


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
