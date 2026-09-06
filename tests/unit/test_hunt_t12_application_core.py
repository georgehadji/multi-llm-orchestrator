"""
Hunt T12 — application/ orchestration core, agents/, supervisor/, nash/, meta/
================================================================================
Regression tests for the defects found and fixed in wave T12 of the backend-
remainder defect hunt (docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md).

C1: application/decomposer_service.py's Instructor fast path made a real,
    billable API call but never charged the run budget.
C2: engine_core/method_selector.py, container.py, and engine_deps.py had
    wrong-depth relative imports that made the entire ARA reasoning-pipeline
    subsystem (~5,000 lines) unreachable, silently swallowed by bare
    ``except ImportError`` blocks.
C3: routing/__init__.py imported a routing/selector.py that never existed,
    breaking the whole routing/ package and, transitively, model_routing.py.
C4: meta/integration.py had fallen out of sync with an AttributeError fix
    already applied to its canonical counterpart, orchestrator/meta_integration.py.
C5: application/cache_warmup.py imported a nonexistent operations/cache_warmup
    module instead of the real orchestrator.cost_optimization.prompt_cache.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.application.cache_warmup import warm_cache_for_level
from orchestrator.application.decomposer_service import decompose_project
from orchestrator.models import Model

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_c1_instructor_fast_path_charges_budget():
    """The Instructor fast path must charge_fn a non-zero amount, not skip it."""
    fake_task = SimpleNamespace(id="task_1")
    fake_result = MagicMock()
    fake_result.to_tasks.return_value = [fake_task]

    fake_decomposer_inst = MagicMock()
    fake_decomposer_inst.decompose = AsyncMock(return_value=fake_result)
    fake_decomposer_inst._calculate_decomposition_tokens.return_value = 500

    fallback_decomposer = MagicMock()
    fallback_decomposer.decompose = AsyncMock(
        side_effect=AssertionError("fallback path should not run when the fast path succeeds")
    )

    charge_fn = AsyncMock()

    with patch("orchestrator.structured_outputs.TaskDecomposer", return_value=fake_decomposer_inst):
        tasks = await decompose_project(
            project="Build a small CLI tool that reverses a string.",
            criteria="Works on stdin",
            model=Model.GPT_4O_MINI,
            client=None,
            decomposer=fallback_decomposer,
            api_health={},
            record_failure_fn=AsyncMock(),
            charge_fn=charge_fn,
        )

    assert tasks == {"task_1": fake_task}
    charge_fn.assert_awaited_once()
    (charged_amount,), _ = charge_fn.await_args
    assert charged_amount > 0.0


def test_c2_ara_reasoning_pipeline_imports_succeed():
    """The full ARA import chain (method_selector, ara_integration,
    ara_execution_strategy) must resolve — previously ModuleNotFoundError
    on every one of these three wrong-depth relative imports."""
    from orchestrator.ara_execution_strategy import ARAExecutionStrategy
    from orchestrator.ara_integration import create_ara_integration
    from orchestrator.engine_core.method_selector import MethodSelector, ReasoningMethod

    assert ReasoningMethod.MULTI_PERSPECTIVE == "multi_perspective"
    assert callable(create_ara_integration)
    # .enabled is a property requiring both config.enabled and a real
    # ara_integration — a mock stands in here since only importability and
    # constructibility are under test, not ARA's own dispatch logic.
    strategy = ARAExecutionStrategy(ara_integration=MagicMock())
    assert strategy.enabled is True
    assert MethodSelector is not None


def test_c3_routing_package_and_model_routing_import_succeed():
    """routing/__init__.py's reference to a nonexistent selector.py broke
    orchestrator.routing and, transitively, orchestrator.model_routing
    (parent package __init__.py always runs first)."""
    import orchestrator.model_routing
    import orchestrator.routing

    assert hasattr(orchestrator.model_routing, "TIER_ROUTING")


def test_c4_meta_integration_shim_matches_canonical():
    """meta/integration.py must resolve to the same, already-fixed class as
    the canonical root orchestrator.meta_integration — not a stale copy that
    still raises AttributeError when state.status is a plain string."""
    from orchestrator.meta.integration import MetaOptimizationV2Wrapper as ViaShim
    from orchestrator.meta_integration import MetaOptimizationV2Wrapper as Canonical

    assert ViaShim is Canonical


@pytest.mark.asyncio
async def test_c5_cache_warmup_imports_real_prompt_cache_module(caplog):
    """warm_cache_for_level must reach cost_optimization.prompt_cache.warm_prompt_cache.
    Pre-fix, it imported a nonexistent operations.cache_warmup module, which
    raised ModuleNotFoundError, silently caught by the function's own
    ``except Exception`` and logged as a warning instead."""
    context_service = MagicMock()
    context_service.build_system_prompt.return_value = "You are a helpful assistant."
    context_service.build_project_context.return_value = "Project: reverse a string"

    with caplog.at_level(logging.WARNING):
        with patch(
            "orchestrator.cost_optimization.prompt_cache.warm_prompt_cache",
            new_callable=AsyncMock,
        ) as mock_warm:
            await warm_cache_for_level(
                context_service=context_service,
                results={},
                client=MagicMock(),
            )

    mock_warm.assert_awaited_once()
    assert not any("Cache warming failed" in rec.message for rec in caplog.records)
