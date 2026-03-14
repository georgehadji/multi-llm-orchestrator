"""Tests for Content Preflight Gate integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from orchestrator.hooks import EventType
from orchestrator.models import TaskResult, Model, Task, TaskType
from orchestrator.preflight import PreflightResult, PreflightAction, PreflightMode


def test_preflight_check_event_exists():
    assert hasattr(EventType, "PREFLIGHT_CHECK")
    assert EventType.PREFLIGHT_CHECK.value == "preflight_check"


def test_task_result_has_preflight_fields():
    result = TaskResult(
        task_id="t_001",
        output="hello world",
        score=0.9,
        model_used=Model.DEEPSEEK_CHAT,
    )
    assert result.preflight_passed is True
    assert result.preflight_result is None


def test_task_result_accepts_preflight_result():
    pf = PreflightResult(action=PreflightAction.WARN, passed=False, warnings=["test warning"])
    result = TaskResult(
        task_id="t_001",
        output="hello world",
        score=0.9,
        model_used=Model.DEEPSEEK_CHAT,
        preflight_result=pf,
        preflight_passed=False,
    )
    assert result.preflight_passed is False
    assert result.preflight_result.action == PreflightAction.WARN


# ─────────────────────────────────────────
# _run_preflight_check unit tests
# ─────────────────────────────────────────

def _make_orchestrator():
    """Minimal Orchestrator stub for unit testing preflight."""
    from orchestrator.engine import Orchestrator
    from orchestrator.preflight import PreflightValidator
    from orchestrator.hooks import HookRegistry
    orch = Orchestrator.__new__(Orchestrator)
    orch._preflight_validator = PreflightValidator()
    orch._hook_registry = HookRegistry()
    return orch


def _make_task(task_type=TaskType.CODE_GEN):
    return Task(
        id="t_test",
        prompt="Write a hello world function",
        type=task_type,
        max_output_tokens=1000,
        acceptance_threshold=0.8,
        max_iterations=3,
    )


@pytest.mark.asyncio
async def test_preflight_pass_returns_unchanged():
    orch = _make_orchestrator()
    task = _make_task()
    output = "def hello(): return 'world'"
    score = 0.9

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    assert final_output == output
    assert final_score == score
    assert pf_result.action == PreflightAction.PASS


@pytest.mark.asyncio
async def test_preflight_warn_applies_score_penalty():
    orch = _make_orchestrator()
    # Inject a validator that always returns WARN so we test the WARN branch of
    # _run_preflight_check without depending on AUTO-mode routing heuristics.
    from orchestrator.preflight import PreflightResult, PreflightAction
    warn_result = PreflightResult(
        action=PreflightAction.WARN,
        passed=False,
        warnings=["TODO placeholder found"],
    )
    orch._preflight_validator.validate = MagicMock(return_value=warn_result)

    task = _make_task()
    output = "def hello(): pass  # [TODO] implement this properly"
    score = 0.9

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    assert final_output == output
    assert final_score == pytest.approx(0.9 * 0.85, abs=0.001)
    assert pf_result.action == PreflightAction.WARN


@pytest.mark.asyncio
async def test_preflight_block_retries_and_recovers():
    orch = _make_orchestrator()
    task = _make_task()
    # Bearer token triggers PRIVACY BLOCK (severity=9, >=8 -> BLOCK in AUTO mode)
    bad_output = 'headers = {"Authorization": "Bearer eyJhbGc.eyJzdWI.sig"}'
    good_output = 'headers = {"Authorization": f"Bearer {token}"}'  # clean revision
    score = 0.8

    mock_response = MagicMock()
    mock_response.text = good_output
    orch.client = MagicMock()
    orch.client.call = AsyncMock(return_value=mock_response)

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=bad_output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    orch.client.call.assert_called_once()
    assert final_output == good_output   # revised output delivered
    assert final_score == score          # score preserved on recovery


@pytest.mark.asyncio
async def test_preflight_block_still_blocked_after_retry():
    orch = _make_orchestrator()
    task = _make_task()
    bad_output = 'headers = {"Authorization": "Bearer eyJhbGc.eyJzdWI.sig"}'
    score = 0.8
    # Revision also leaks a token (still blocked)
    still_bad = 'auth = "Bearer eyJuZXc.eyJzdWI.newsig"'

    mock_response = MagicMock()
    mock_response.text = still_bad
    orch.client = MagicMock()
    orch.client.call = AsyncMock(return_value=mock_response)

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=bad_output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    assert final_score == 0.0           # degraded
    assert final_output == bad_output   # original preserved (not revision)


@pytest.mark.asyncio
async def test_preflight_validator_exception_is_fail_open():
    orch = _make_orchestrator()
    orch._preflight_validator.validate = MagicMock(side_effect=RuntimeError("validator down"))
    task = _make_task()
    output = "def hello(): return 'world'"
    score = 0.9

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    # Fail-open: exception -> PASS, pipeline unaffected
    assert final_output == output
    assert final_score == score
    assert pf_result.action == PreflightAction.PASS
