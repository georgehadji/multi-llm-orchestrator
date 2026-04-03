# tests/test_adaptive_router.py
import asyncio
import time
import pytest
from orchestrator.adaptive_router import AdaptiveRouter, ModelState
from orchestrator.models import Model, TaskType


@pytest.mark.asyncio
async def test_initial_state_all_healthy():
    router = AdaptiveRouter()
    for m in Model:
        assert await router.get_state(m) == ModelState.HEALTHY


@pytest.mark.asyncio
async def test_consecutive_timeouts_degrade_model():
    router = AdaptiveRouter(timeout_threshold=3)
    for _ in range(3):
        await router.record_timeout(Model.DEEPSEEK_CHAT)
    assert await router.get_state(Model.DEEPSEEK_CHAT) == ModelState.DEGRADED


@pytest.mark.asyncio
async def test_degraded_model_recovers_after_cooldown():
    router = AdaptiveRouter(timeout_threshold=3, cooldown_seconds=0.1)
    for _ in range(3):
        await router.record_timeout(Model.DEEPSEEK_CHAT)
    assert await router.get_state(Model.DEEPSEEK_CHAT) == ModelState.DEGRADED
    time.sleep(0.15)
    assert await router.get_state(Model.DEEPSEEK_CHAT) == ModelState.HEALTHY


@pytest.mark.asyncio
async def test_auth_failure_permanently_disables():
    router = AdaptiveRouter()
    await router.record_auth_failure(Model.DEEPSEEK_CHAT)
    assert await router.get_state(Model.DEEPSEEK_CHAT) == ModelState.DISABLED


@pytest.mark.asyncio
async def test_success_resets_timeout_counter():
    router = AdaptiveRouter(timeout_threshold=3)
    await router.record_timeout(Model.DEEPSEEK_CHAT)
    await router.record_timeout(Model.DEEPSEEK_CHAT)
    await router.record_success(Model.DEEPSEEK_CHAT)
    # After success, counter resets — needs 3 more to degrade
    await router.record_timeout(Model.DEEPSEEK_CHAT)
    await router.record_timeout(Model.DEEPSEEK_CHAT)
    assert await router.get_state(Model.DEEPSEEK_CHAT) == ModelState.HEALTHY


@pytest.mark.asyncio
async def test_prefer_fastest_healthy_model():
    router = AdaptiveRouter()
    await router.record_latency(Model.DEEPSEEK_CHAT, 500.0)
    await router.record_latency(Model.GPT_4O_MINI, 2000.0)
    candidates = [Model.DEEPSEEK_CHAT, Model.GPT_4O_MINI]
    best = await router.preferred_model(candidates, task_type=TaskType.CODE_GEN)
    assert best == Model.DEEPSEEK_CHAT


@pytest.mark.asyncio
async def test_disabled_model_excluded_from_preferred():
    router = AdaptiveRouter()
    await router.record_auth_failure(Model.DEEPSEEK_CHAT)
    candidates = [Model.DEEPSEEK_CHAT, Model.GPT_4O_MINI]
    best = await router.preferred_model(candidates, task_type=TaskType.CODE_GEN)
    assert best == Model.GPT_4O_MINI


@pytest.mark.asyncio
async def test_all_disabled_returns_none():
    router = AdaptiveRouter()
    await router.record_auth_failure(Model.DEEPSEEK_CHAT)
    await router.record_auth_failure(Model.GPT_4O_MINI)
    best = await router.preferred_model(
        [Model.DEEPSEEK_CHAT, Model.GPT_4O_MINI], task_type=TaskType.CODE_GEN
    )
    assert best is None
