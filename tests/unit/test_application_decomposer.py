"""
Unit tests for the decomposition layer (orchestrator.application.decomposer).

Covers two collaborators:
  - DecomposerService — the never-raising callback wrapper + metrics
  - Decomposer        — the pure JSON parsing / partial-recovery / model-selection logic

LLM calls are never exercised; only the deterministic logic around them is tested.
"""

import asyncio

import pytest

pytestmark = pytest.mark.unit

from orchestrator.application.decomposer import (
    Decomposer,
    DecomposerMetrics,
    DecomposerResult,
    DecomposerService,
)
from orchestrator.exceptions import OrchestratorError
from orchestrator.models import Model, Task, TaskType

# ── DecomposerResult ────────────────────────────────────────────────────────


class TestDecomposerResult:
    def test_succeeded_true_when_tasks_and_no_error(self):
        task = Task(id="t1", type=TaskType.CODE_GEN, prompt="do it")
        result = DecomposerResult(tasks={"t1": task}, wall_time_ms=10.0)
        assert result.succeeded
        assert result.task_count == 1

    def test_succeeded_false_when_error(self):
        result = DecomposerResult(tasks={}, wall_time_ms=1.0, error=ValueError("boom"))
        assert not result.succeeded

    def test_succeeded_false_when_empty(self):
        result = DecomposerResult(tasks={}, wall_time_ms=1.0)
        assert not result.succeeded
        assert result.task_count == 0


# ── DecomposerMetrics ───────────────────────────────────────────────────────


class TestDecomposerMetrics:
    def test_record_success_updates_counters(self):
        metrics = DecomposerMetrics()
        task = Task(id="t1", type=TaskType.CODE_GEN, prompt="x")
        metrics.record(DecomposerResult(tasks={"t1": task}, wall_time_ms=20.0))

        assert metrics.total_calls == 1
        assert metrics.total_succeeded == 1
        assert metrics.total_failed == 0
        assert metrics.total_tasks_generated == 1

    def test_record_failure_updates_counters(self):
        metrics = DecomposerMetrics()
        metrics.record(DecomposerResult(tasks={}, wall_time_ms=5.0, error=RuntimeError("nope")))

        assert metrics.total_calls == 1
        assert metrics.total_succeeded == 0
        assert metrics.total_failed == 1
        assert metrics.total_tasks_generated == 0

    def test_to_dict_computes_average(self):
        metrics = DecomposerMetrics()
        task = Task(id="t1", type=TaskType.CODE_GEN, prompt="x")
        metrics.record(DecomposerResult(tasks={"t1": task}, wall_time_ms=100.0))
        metrics.record(DecomposerResult(tasks={"t1": task}, wall_time_ms=200.0))

        snapshot = metrics.to_dict()
        assert snapshot["total_calls"] == 2
        assert snapshot["avg_wall_ms"] == 150.0

    def test_to_dict_average_zero_when_no_calls(self):
        assert DecomposerMetrics().to_dict()["avg_wall_ms"] == 0.0


# ── DecomposerService (callback wrapper) ────────────────────────────────────


class TestDecomposerService:
    def test_decompose_success_records_metrics(self):
        task = Task(id="t1", type=TaskType.CODE_GEN, prompt="build")

        async def fake_fn(project, criteria, **kwargs):
            return {"t1": task}

        service = DecomposerService(decompose_fn=fake_fn)
        result = asyncio.run(service.decompose("proj", "crit"))

        assert result.succeeded
        assert result.tasks == {"t1": task}
        assert service.metrics.total_succeeded == 1

    def test_decompose_swallows_exception_and_wraps(self):
        async def boom_fn(project, criteria, **kwargs):
            raise RuntimeError("kaboom")

        service = DecomposerService(decompose_fn=boom_fn)
        result = asyncio.run(service.decompose("proj", "crit"))

        # Never raises — the error is captured on the result.
        assert not result.succeeded
        assert isinstance(result.error, OrchestratorError)
        assert service.metrics.total_failed == 1

    def test_decompose_timeout_produces_orchestrator_error(self):
        async def slow_fn(project, criteria, **kwargs):
            await asyncio.sleep(1.0)
            return {}

        service = DecomposerService(decompose_fn=slow_fn, decompose_timeout=0.01)
        result = asyncio.run(service.decompose("proj", "crit"))

        assert not result.succeeded
        assert isinstance(result.error, OrchestratorError)
        assert "timed out" in str(result.error)

    def test_decompose_fn_setter_late_binds(self):
        async def first_fn(project, criteria, **kwargs):
            return {}

        async def second_fn(project, criteria, **kwargs):
            return {"t1": Task(id="t1", type=TaskType.CODE_GEN, prompt="p")}

        service = DecomposerService(decompose_fn=first_fn)
        service.decompose_fn = second_fn
        result = asyncio.run(service.decompose("proj", "crit"))

        assert result.task_count == 1

    def test_decompose_forwards_project_context_kwarg(self):
        seen = {}

        async def capture_fn(project, criteria, **kwargs):
            seen.update(kwargs)
            return {}

        service = DecomposerService(decompose_fn=capture_fn)
        asyncio.run(service.decompose("proj", "crit", project_context="CTX"))

        assert seen.get("project_context") == "CTX"
        assert "policy" in seen  # policy is always forwarded


# ── Decomposer — pure parsing logic ─────────────────────────────────────────


@pytest.fixture
def decomposer():
    """Decomposer with no client/selector — only parsing methods are exercised."""
    return Decomposer(client=None, selector=None)


class TestDecomposerParsing:
    def test_parse_valid_json_list(self, decomposer):
        text = """[
            {"id": "t1", "type": "code_generation", "prompt": "Write a function"},
            {"id": "t2", "type": "code_review", "prompt": "Review it", "dependencies": ["t1"]}
        ]"""
        tasks = decomposer._parse_decomposition(text)

        assert set(tasks) == {"t1", "t2"}
        assert tasks["t1"].type == TaskType.CODE_GEN
        assert tasks["t2"].dependencies == ["t1"]

    def test_parse_strips_markdown_fences(self, decomposer):
        text = '```json\n[{"id": "t1", "type": "code_generation", "prompt": "x"}]\n```'
        tasks = decomposer._parse_decomposition(text)
        assert "t1" in tasks

    def test_parse_unwraps_tasks_key(self, decomposer):
        text = '{"tasks": [{"id": "t1", "type": "code_generation", "prompt": "x"}]}'
        tasks = decomposer._parse_decomposition(text)
        assert "t1" in tasks

    def test_parse_maps_alias_task_type(self, decomposer):
        # "code_gen" is not a TaskType value but maps to CODE_GEN.
        text = '[{"id": "t1", "type": "code_gen", "prompt": "x"}]'
        tasks = decomposer._parse_decomposition(text)
        assert tasks["t1"].type == TaskType.CODE_GEN

    def test_parse_skips_items_missing_fields(self, decomposer):
        text = '[{"id": "t1", "type": "code_generation"}, {"prompt": "no id"}]'
        assert decomposer._parse_decomposition(text) == {}

    def test_parse_unknown_type_skipped(self, decomposer):
        text = '[{"id": "t1", "type": "totally_unknown", "prompt": "x"}]'
        assert decomposer._parse_decomposition(text) == {}

    def test_parse_empty_returns_empty(self, decomposer):
        assert decomposer._parse_decomposition("") == {}

    def test_parse_non_list_returns_empty(self, decomposer):
        assert decomposer._parse_decomposition('{"id": "t1"}') == {}

    def test_parse_string_dependency_coerced_to_list(self, decomposer):
        text = '[{"id": "t1", "type": "code_generation", "prompt": "x", "dependencies": "t0"}]'
        tasks = decomposer._parse_decomposition(text)
        assert tasks["t1"].dependencies == ["t0"]


class TestDecomposerPartialRecovery:
    def test_closes_truncated_array(self, decomposer):
        text = '[{"id": "t1", "type": "code_generation", "prompt": "x"}'
        recovered = decomposer._try_parse_partial_json_array(text)
        assert isinstance(recovered, list)
        assert recovered[0]["id"] == "t1"

    def test_extracts_objects_via_regex(self, decomposer):
        # Malformed wrapper, but two complete objects embedded.
        text = 'garbage {"id": "t1", "prompt": "a"} , {"id": "t2", "prompt": "b"} trailing'
        recovered = decomposer._try_parse_partial_json_array(text)
        assert recovered is not None
        assert len(recovered) == 2

    def test_returns_none_on_empty(self, decomposer):
        assert decomposer._try_parse_partial_json_array("   ") is None

    def test_repair_partial_tasks_builds_tasks(self, decomposer):
        objects = [
            {"id": "t1", "type": "code_generation", "prompt": "do x"},
            {"id": "t2", "type": "bad_type", "prompt": "skip me"},
        ]
        tasks = decomposer._repair_partial_tasks(objects)
        assert set(tasks) == {"t1"}  # invalid type dropped

    def test_repair_partial_tasks_none_when_empty(self, decomposer):
        assert decomposer._repair_partial_tasks([{"prompt": "no id or type"}]) is None


# ── Decomposer — model selection ────────────────────────────────────────────


class _FakeSelector:
    def __init__(self, model):
        self._model = model

    def decomposition_model(self, project_description):
        return self._model


class TestDecomposerModelSelection:
    def test_primary_first_and_flash_fallbacks_appended(self):
        decomposer = Decomposer(client=None, selector=_FakeSelector(Model.GPT_5_4))
        models = decomposer._get_decomposition_models("build an api")

        assert models[0] == Model.GPT_5_4
        assert Model.QWEN_3_6_FLASH in models
        assert Model.XIAOMI_MIMO_V2_FLASH in models
        # Both flash fallbacks appended exactly once (append guard works).
        assert models.count(Model.QWEN_3_6_FLASH) == 1
        assert models.count(Model.XIAOMI_MIMO_V2_FLASH) == 1
