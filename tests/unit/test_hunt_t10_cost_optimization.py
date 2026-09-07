"""
T10 (cost_optimization/ remainder) proof-of-defect and no-regression tests.

Four VERIFIED DEFECTs from docs/hunts/t10-cost-optimization/inventory.md:

C1 — orchestrator/token_budget.py was a byte-for-byte independent duplicate
     (not a shim) of orchestrator/infrastructure/token_budget.py — the same
     unshimmed-duplicate-pair shape every prior tier (T1/T2/T3/T5/T7/T9) has
     found silently diverging once one copy gets a fix the other doesn't.
C2 — orchestrator/provisioned_throughput.py was likewise an unshimmed,
     byte-for-byte independent duplicate of
     orchestrator/operations/provisioned_throughput.py (the module
     orchestrator/infrastructure/provisioned_throughput.py already shims to).
C3 — cost_optimization/cost_optimization_integration.py's
     Tier1OptimizationMixin used a same-package-depth relative import
     (`from .log_config import get_logger`) one dot short for its actual
     depth (orchestrator/cost_optimization/), raising ModuleNotFoundError
     unconditionally at import time. The module was deleted outright in
     docs/plans/2026-09-07-patterns-convergence-and-wire-or-delete.md S6
     (TEST-ONLY, and the batch client it wrapped could never provide real
     batching over the OpenRouter adapter) — its C3 regression test below
     went with it.
C4 — cost_optimization/docker_sandbox.py::DockerSandbox.execute() wrote
     caller-supplied `code_files` filenames straight into the sandbox
     workspace with no path-containment check — an absolute path or a
     "../" traversal filename could write outside the temporary sandbox
     directory onto the host filesystem.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


def test_c1_root_token_budget_is_canonical():
    from orchestrator.infrastructure.token_budget import TokenBudgetManager as canonical
    from orchestrator.token_budget import TokenBudgetManager as via_root

    assert via_root is canonical


# --- C2 -----------------------------------------------------------------


def test_c2_root_provisioned_throughput_is_canonical():
    from orchestrator.operations.provisioned_throughput import (
        ProvisionedThroughputManager as canonical,
    )
    from orchestrator.provisioned_throughput import (
        ProvisionedThroughputManager as via_root,
    )

    assert via_root is canonical


# --- C4 -----------------------------------------------------------------


async def _fake_check_docker(self) -> bool:
    return True


class _FakeDockerModule:
    def from_env(self):
        # Never actually reached before the containment check fires.
        return object()


@pytest.mark.asyncio
async def test_c4_docker_sandbox_rejects_path_traversal_filename(monkeypatch):
    import sys

    from orchestrator.cost_optimization.docker_sandbox import DockerSandbox

    monkeypatch.setitem(sys.modules, "docker", _FakeDockerModule())
    monkeypatch.setattr(DockerSandbox, "_check_docker", _fake_check_docker)

    sandbox = DockerSandbox()
    result = await sandbox.execute(
        code_files={"../../../etc/evil.txt": "malicious content"},
        command="true",
    )

    assert result.return_code == -1
    assert (
        "escapes the sandbox" in result.error
    ), f"expected a containment error, got: {result.error!r}"


@pytest.mark.asyncio
async def test_c4_docker_sandbox_accepts_normal_filename(monkeypatch):
    """Confirms the fix doesn't break the legitimate case."""
    import sys

    from orchestrator.cost_optimization.docker_sandbox import DockerSandbox

    monkeypatch.setitem(sys.modules, "docker", _FakeDockerModule())
    monkeypatch.setattr(DockerSandbox, "_check_docker", _fake_check_docker)

    sandbox = DockerSandbox()
    result = await sandbox.execute(
        code_files={"main.py": "print('hello')"},
        command="true",
    )

    # Still fails (the fake docker client's from_env() object has no real
    # .containers.run()), but NOT on the containment check — confirms a
    # normal filename passes through it untouched.
    assert "escapes the sandbox" not in (result.error or "")
