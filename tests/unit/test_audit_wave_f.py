"""Wave F of the 2026-10-29 audit remediation plan: lean (dead code out)."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


# ── E3 follow-up: the unused retry stacks are gone ──────────────────────────


@pytest.mark.parametrize(
    "module",
    ["orchestrator.retry_utils", "orchestrator.operations.retry_utils"],
)
def test_retry_utils_is_deleted(module) -> None:
    # The SDK retries (client.with_options(max_retries=...)); no second stack.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_run_with_resilience_is_deleted() -> None:
    import orchestrator.operations.resilience as canonical
    import orchestrator.resilience as shim

    assert not hasattr(canonical, "run_with_resilience")
    assert not hasattr(shim, "run_with_resilience")
