# tests/test_adaptive_router.py
import time
from orchestrator.adaptive_router import AdaptiveRouter, ModelState
from orchestrator.models import Model, TaskType


def test_initial_state_all_healthy():
    router = AdaptiveRouter()
    for m in Model:
        assert router.get_state(m) == ModelState.HEALTHY


def test_consecutive_timeouts_degrade_model():
    router = AdaptiveRouter(timeout_threshold=3)
    for _ in range(3):
        router.record_timeout(Model.DEEPSEEK_CHAT)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.DEGRADED


def test_degraded_model_recovers_after_cooldown():
    router = AdaptiveRouter(timeout_threshold=3, cooldown_seconds=0.1)
    for _ in range(3):
        router.record_timeout(Model.DEEPSEEK_CHAT)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.DEGRADED
    time.sleep(0.15)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.HEALTHY


def test_auth_failure_permanently_disables():
    router = AdaptiveRouter()
    router.record_auth_failure(Model.DEEPSEEK_CHAT)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.DISABLED


def test_success_resets_timeout_counter():
    router = AdaptiveRouter(timeout_threshold=3)
    router.record_timeout(Model.DEEPSEEK_CHAT)
    router.record_timeout(Model.DEEPSEEK_CHAT)
    router.record_success(Model.DEEPSEEK_CHAT)
    # After success, counter resets — needs 3 more to degrade
    router.record_timeout(Model.DEEPSEEK_CHAT)
    router.record_timeout(Model.DEEPSEEK_CHAT)
    assert router.get_state(Model.DEEPSEEK_CHAT) == ModelState.HEALTHY


def test_prefer_fastest_healthy_model():
    router = AdaptiveRouter()
    router.record_latency(Model.DEEPSEEK_CHAT, 500.0)
    router.record_latency(Model.KIMI_K2_5, 2000.0)
    candidates = [Model.DEEPSEEK_CHAT, Model.KIMI_K2_5]
    best = router.preferred_model(candidates, task_type=TaskType.CODE_GEN)
    assert best == Model.DEEPSEEK_CHAT


def test_disabled_model_excluded_from_preferred():
    router = AdaptiveRouter()
    router.record_auth_failure(Model.DEEPSEEK_CHAT)
    candidates = [Model.DEEPSEEK_CHAT, Model.KIMI_K2_5]
    best = router.preferred_model(candidates, task_type=TaskType.CODE_GEN)
    assert best == Model.KIMI_K2_5


def test_all_disabled_returns_none():
    router = AdaptiveRouter()
    router.record_auth_failure(Model.DEEPSEEK_CHAT)
    router.record_auth_failure(Model.KIMI_K2_5)
    best = router.preferred_model(
        [Model.DEEPSEEK_CHAT, Model.KIMI_K2_5], task_type=TaskType.CODE_GEN
    )
    assert best is None
