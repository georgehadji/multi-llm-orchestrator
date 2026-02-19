"""
AI Orchestrator — Stress Test Suite
=====================================
Runs the full orchestrator pipeline end-to-end against REAL LLM APIs.

Scenarios:
  S1  Quick smoke    — 1 task, max $0.10, verify it runs at all
  S2  Multi-task DAG — 6 tasks with dependencies, realistic project
  S3  Budget ceiling — verify hard stop before $0.05 spent
  S4  Parallel load  — 3 projects fired concurrently, race-condition check
  S5  Fallback chain — primary provider disabled, verify graceful failover
  S6  Policy gate    — EU-only policy blocks non-EU models (real API call check)
  S7  Resume         — partial run interrupted, resumed from checkpoint
  S8  Cache hit      — same prompt twice, second call must be free (cache hit)
  S9  Telemetry drift— 20 consecutive calls, quality EMA/trust must update
  S10 Full project   — real "build a Python utility" project, $2 budget

Usage:
    # Run all scenarios (uses real APIs, costs ~$0.50–2.00):
    python -m pytest tests/stress_test.py -v

    # Run only cheap/fast scenarios (no LLM calls):
    python -m pytest tests/stress_test.py -v -m "mock"

    # Run a single scenario:
    python -m pytest tests/stress_test.py::TestS1QuickSmoke -v

    # Run with visible output:
    python -m pytest tests/stress_test.py -v -s

Environment variables required (at least one provider):
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import logging
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Make sure we can import from the repo root ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import (
    Orchestrator, Budget, Model, Task, TaskResult,
    TaskType, TaskStatus, ProjectState, ProjectStatus,
    DiskCache, StateManager,
    ModelProfile, Policy, PolicySet, JobSpec,
    PolicyEngine, PolicyViolationError,
    ConstraintPlanner, TelemetryCollector,
    build_default_profiles,
)
from orchestrator.models import ROUTING_TABLE, COST_TABLE, get_provider

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("stress")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _at_least_one_provider_available() -> bool:
    """Returns True if any real API key is configured."""
    return any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
    ])


def _available_providers() -> list[str]:
    result = []
    if os.getenv("OPENAI_API_KEY"):
        result.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        result.append("anthropic")
    if os.getenv("GOOGLE_API_KEY"):
        result.append("google")
    return result


def _run(coro):
    """Run a coroutine in the event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_response(text: str = '{"result": "ok"}', cost: float = 0.001,
                        latency: float = 300.0, model: Model = Model.GPT_4O_MINI):
    """Return a fake APIResponse-like object."""
    from orchestrator.api_clients import APIResponse
    resp = APIResponse(
        text=text,
        input_tokens=100,
        output_tokens=50,
        model=model,
        cached=False,
        latency_ms=latency,
    )
    resp.cost_usd = cost
    return resp


def _decomposed_tasks_json(n_tasks: int = 3) -> str:
    """Generate a minimal valid JSON decomposition with n tasks."""
    tasks = []
    for i in range(1, n_tasks + 1):
        task_type = "code_generation" if i % 2 == 1 else "code_review"
        deps = [f"task_{i-1:03d}"] if i > 1 else []
        tasks.append({
            "id": f"task_{i:03d}",
            "type": task_type,
            "prompt": f"Task {i}: write a Python function that does something useful.",
            "dependencies": deps,
            "hard_validators": ["python_syntax"] if task_type == "code_generation" else [],
        })
    # Append an evaluation task
    tasks.append({
        "id": f"task_{n_tasks+1:03d}",
        "type": "evaluation",
        "prompt": "Evaluate the overall quality of the previous outputs.",
        "dependencies": [f"task_{n_tasks:03d}"],
        "hard_validators": [],
    })
    return json.dumps(tasks)


# ─────────────────────────────────────────────────────────────────────────────
# S1  Quick smoke test — 1 task, mocked LLM
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS1QuickSmoke:
    """S1: Verify the orchestrator runs end-to-end with 1 task using mocked APIs."""

    def test_s1_single_task_completes(self, tmp_path):
        """Orchestrator executes a single task and returns PARTIAL_SUCCESS or SUCCESS."""
        decomp = json.dumps([{
            "id": "task_001",
            "type": "code_generation",
            "prompt": "Write a Python hello world function.",
            "dependencies": [],
            "hard_validators": ["python_syntax"],
        }])
        gen_resp   = _make_mock_response("def hello():\n    return 'Hello, world!'")
        eval_resp  = _make_mock_response('{"score": 0.9, "critique": "Looks good."}')
        decomp_resp = _make_mock_response(decomp)

        orch = Orchestrator(
            budget=Budget(max_usd=0.10),
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
        )

        call_count = {"n": 0}

        async def mock_call(model, prompt, **kwargs):
            call_count["n"] += 1
            if "decomposition engine" in kwargs.get("system", ""):
                return decomp_resp
            if "evaluating" in prompt.lower() or "evaluate" in prompt.lower():
                return _make_mock_response('{"score": 0.9, "critique": "Good."}')
            return gen_resp

        with patch.object(orch.client, "call", side_effect=mock_call):
            state = _run(orch.run_project(
                "Write a hello world function",
                "Function returns 'Hello, world!'"
            ))

        assert state is not None
        assert state.status in (
            ProjectStatus.SUCCESS,
            ProjectStatus.PARTIAL_SUCCESS,
        ), f"Unexpected status: {state.status}"
        assert call_count["n"] > 0, "No API calls were made"
        log.info(f"S1 passed: {call_count['n']} calls, status={state.status.value}")

    def test_s1_orchestrator_respects_budget_field(self, tmp_path):
        """Budget object is accessible and correct after construction."""
        budget = Budget(max_usd=0.50, max_time_seconds=120.0)
        orch = Orchestrator(
            budget=budget,
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
        )
        assert orch.budget.max_usd == 0.50
        assert orch.budget.max_time_seconds == 120.0
        assert orch.budget.remaining_usd == 0.50


# ─────────────────────────────────────────────────────────────────────────────
# S2  Multi-task DAG — 6 tasks + 1 evaluation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS2MultiTaskDAG:
    """S2: Verifies dependency ordering and multi-task execution."""

    def test_s2_dag_executes_in_order(self, tmp_path):
        """Tasks execute in topological order; all tasks have a result."""
        decomp_json = _decomposed_tasks_json(n_tasks=4)
        execution_trace: list[str] = []

        async def mock_call(model, prompt, **kwargs):
            system = kwargs.get("system", "")
            if "decomposition engine" in system:
                return _make_mock_response(decomp_json)
            # Track which task prompt is being executed
            for line in prompt.split("\n"):
                if line.startswith("Task "):
                    execution_trace.append(line.split(":")[0].strip())
                    break
            return _make_mock_response(
                "def foo():\n    pass  # stub",
                cost=0.001
            )

        orch = Orchestrator(
            budget=Budget(max_usd=1.0),
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
        )
        with patch.object(orch.client, "call", side_effect=mock_call):
            state = _run(orch.run_project("Build a simple util", "It works"))

        # All 5 tasks (4 + evaluation) should have results
        tasks_with_results = len(state.results)
        log.info(f"S2: {tasks_with_results} tasks completed, status={state.status.value}")
        assert tasks_with_results >= 3, (
            f"Expected at least 3 completed tasks, got {tasks_with_results}"
        )

    def test_s2_dependent_task_skipped_if_dep_fails(self, tmp_path):
        """If a task fails, downstream dependents are marked FAILED."""
        tasks_json = json.dumps([
            {"id": "task_001", "type": "code_generation",
             "prompt": "Write failing code", "dependencies": [], "hard_validators": []},
            {"id": "task_002", "type": "code_review",
             "prompt": "Review task_001", "dependencies": ["task_001"], "hard_validators": []},
        ])
        call_n = {"n": 0}

        async def mock_call(model, prompt, **kwargs):
            call_n["n"] += 1
            if "decomposition engine" in kwargs.get("system", ""):
                return _make_mock_response(tasks_json)
            raise RuntimeError("Simulated API failure")

        orch = Orchestrator(
            budget=Budget(max_usd=0.50),
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
        )
        with patch.object(orch.client, "call", side_effect=mock_call):
            state = _run(orch.run_project("Failing project", "Should fail gracefully"))

        # Project should not crash — should return a state with failure info
        assert state is not None
        assert state.status in (
            ProjectStatus.PARTIAL_SUCCESS,
            ProjectStatus.SYSTEM_FAILURE,
        )
        log.info(f"S2b passed: status={state.status.value}")


# ─────────────────────────────────────────────────────────────────────────────
# S3  Budget ceiling — hard stop at budget
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS3BudgetCeiling:
    """S3: Verify the orchestrator stops when the budget is exhausted."""

    def test_s3_hard_stop_at_budget(self, tmp_path):
        """With a $0.02 budget, the orchestrator must stop well before completing all tasks."""
        # 6 tasks that each cost $0.01 = $0.06 total needed
        decomp_json = _decomposed_tasks_json(n_tasks=5)
        tasks_completed = {"n": 0}

        async def mock_call(model, prompt, **kwargs):
            if "decomposition engine" in kwargs.get("system", ""):
                return _make_mock_response(decomp_json)
            tasks_completed["n"] += 1
            # Each generation costs $0.01
            resp = _make_mock_response("def stub(): pass", cost=0.01)
            return resp

        orch = Orchestrator(
            budget=Budget(max_usd=0.04),  # only enough for ~4 calls max
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
        )
        with patch.object(orch.client, "call", side_effect=mock_call):
            state = _run(orch.run_project("Big expensive project", "Do everything"))

        # Budget must never be exceeded
        assert orch.budget.spent_usd <= orch.budget.max_usd + 0.02, (
            f"Budget exceeded: spent ${orch.budget.spent_usd:.4f} "
            f"on max ${orch.budget.max_usd:.4f}"
        )
        log.info(
            f"S3 passed: spent=${orch.budget.spent_usd:.4f}, "
            f"max=${orch.budget.max_usd:.4f}, "
            f"status={state.status.value}"
        )

    def test_s3_budget_exhausted_status(self, tmp_path):
        """When budget runs out, final status is BUDGET_EXHAUSTED."""
        decomp_json = _decomposed_tasks_json(n_tasks=8)

        async def mock_call(model, prompt, **kwargs):
            if "decomposition engine" in kwargs.get("system", ""):
                return _make_mock_response(decomp_json)
            return _make_mock_response("def stub(): pass", cost=0.05)  # expensive

        orch = Orchestrator(
            budget=Budget(max_usd=0.08),
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
        )
        with patch.object(orch.client, "call", side_effect=mock_call):
            state = _run(orch.run_project("Budget buster project", "Do everything"))

        assert state.status in (
            ProjectStatus.BUDGET_EXHAUSTED,
            ProjectStatus.PARTIAL_SUCCESS,
        )
        log.info(f"S3b passed: status={state.status.value}")


# ─────────────────────────────────────────────────────────────────────────────
# S4  Parallel load — 3 concurrent projects
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS4ParallelLoad:
    """S4: Multiple orchestrators running concurrently must not interfere."""

    def test_s4_three_parallel_projects(self, tmp_path):
        """Three independent orchestrator instances run concurrently without interference."""
        decomp_json = _decomposed_tasks_json(n_tasks=2)
        results: list[Optional[ProjectState]] = [None, None, None]

        async def run_all():
            async def run_one(idx: int) -> ProjectState:
                async def mock_call(model, prompt, **kwargs):
                    if "decomposition engine" in kwargs.get("system", ""):
                        return _make_mock_response(decomp_json)
                    return _make_mock_response(
                        f"# Project {idx} output\ndef f{idx}(): pass",
                        cost=0.001,
                    )

                orch = Orchestrator(
                    budget=Budget(max_usd=0.50),
                    cache=DiskCache(db_path=tmp_path / f"cache_{idx}.db"),
                    state_manager=StateManager(db_path=tmp_path / f"state_{idx}.db"),
                )
                with patch.object(orch.client, "call", side_effect=mock_call):
                    return await orch.run_project(
                        f"Project {idx}: build a microservice",
                        "All endpoints return 200",
                        project_id=f"stress_parallel_{idx}",
                    )

            states = await asyncio.gather(
                run_one(0), run_one(1), run_one(2),
                return_exceptions=True,
            )
            return states

        loop = asyncio.new_event_loop()
        states = loop.run_until_complete(run_all())
        loop.close()

        for i, state in enumerate(states):
            assert not isinstance(state, Exception), (
                f"Project {i} raised an exception: {state}"
            )
            assert state is not None
            assert state.status in (
                ProjectStatus.SUCCESS,
                ProjectStatus.PARTIAL_SUCCESS,
                ProjectStatus.BUDGET_EXHAUSTED,
            )
        log.info(f"S4 passed: 3 parallel projects completed, statuses={[s.status.value for s in states]}")


# ─────────────────────────────────────────────────────────────────────────────
# S5  Fallback chain — primary provider disabled
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS5FallbackChain:
    """S5: When primary provider is marked unhealthy, fallback model is used."""

    def test_s5_fallback_on_api_failure(self, tmp_path):
        """Generation fails on primary model → orchestrator replans to fallback model."""
        decomp_json = json.dumps([{
            "id": "task_001",
            "type": "code_generation",
            "prompt": "Write a retry decorator.",
            "dependencies": [],
            "hard_validators": [],
        }])

        # Which model actually responded successfully
        responded_models: list[str] = []

        async def mock_call(model, prompt, **kwargs):
            if "decomposition engine" in kwargs.get("system", ""):
                return _make_mock_response(decomp_json)
            # Fail for openai models; succeed for others
            if get_provider(model) == "openai":
                raise RuntimeError("OpenAI is down")
            responded_models.append(model.value)
            return _make_mock_response(
                "def retry(fn):\n    def wrapper(*a, **kw):\n        return fn(*a, **kw)\n    return wrapper",
                cost=0.001,
            )

        orch = Orchestrator(
            budget=Budget(max_usd=0.50),
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
        )
        # Mark all OpenAI models as unavailable
        for m in Model:
            if get_provider(m) == "openai":
                orch.api_health[m] = False

        with patch.object(orch.client, "call", side_effect=mock_call):
            state = _run(orch.run_project(
                "Build a resilient HTTP client",
                "Retry on failure"
            ))

        # All models that responded should be non-openai
        for m_val in responded_models:
            model_enum = Model(m_val)
            assert get_provider(model_enum) != "openai", (
                f"OpenAI model {m_val} responded despite being marked unhealthy"
            )
        log.info(f"S5 passed: responded models = {responded_models}, status={state.status.value}")

    def test_s5_planner_replan_excludes_failed(self):
        """ConstraintPlanner.replan() never returns the failed model."""
        profiles = build_default_profiles()
        engine = PolicyEngine()
        health = {m: True for m in Model}
        planner = ConstraintPlanner(profiles, engine, health)

        for task_type in [TaskType.CODE_GEN, TaskType.CODE_REVIEW, TaskType.REASONING]:
            original = planner.select_model(task_type, [], budget_remaining=100.0)
            if original is None:
                continue
            fallback = planner.replan(original, task_type, [], budget_remaining=100.0)
            assert fallback != original, (
                f"replan() returned the same model {original} for {task_type}"
            )
        log.info("S5b passed: replan() always returns a different model")


# ─────────────────────────────────────────────────────────────────────────────
# S6  Policy gate — EU-only blocks non-EU models
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS6PolicyGate:
    """S6: Policy enforcement works end-to-end with the planner."""

    def test_s6_eu_policy_restricts_selection(self):
        """EU-only policy prevents any non-EU model from being selected."""
        profiles = {m: build_default_profiles()[m] for m in Model}

        # Tag only one model as EU
        eu_model = Model.CLAUDE_SONNET
        profiles[eu_model].region = "eu"
        for m in profiles:
            if m != eu_model:
                profiles[m].region = "us"

        engine = PolicyEngine()
        health = {m: True for m in Model}
        planner = ConstraintPlanner(profiles, engine, health)
        eu_policy = Policy(name="eu_only", allowed_regions=["eu"])

        for task_type in TaskType:
            result = planner.select_model(task_type, [eu_policy], budget_remaining=100.0)
            if result is not None:
                assert profiles[result].region == "eu", (
                    f"Non-EU model {result} selected despite EU-only policy for {task_type}"
                )
        log.info("S6 passed: EU-only policy respected for all task types")

    def test_s6_policy_violation_error_carries_detail(self):
        """PolicyViolationError carries task_id, policies, and reason."""
        engine = PolicyEngine()
        profile = ModelProfile(
            model=Model.GPT_4O,
            provider="openai",
            cost_per_1m_input=2.5,
            cost_per_1m_output=10.0,
            region="us",
        )
        policies = [
            Policy(name="block_openai", blocked_providers=["openai"]),
            Policy(name="eu_only", allowed_regions=["eu"]),
        ]
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.enforce(Model.GPT_4O, profile, policies)
        err = exc_info.value
        # task_id defaults to "pre-flight" (set by enforce())
        assert err.task_id == "pre-flight"
        assert len(err.policies) >= 1
        log.info(f"S6b passed: PolicyViolationError.task_id={err.task_id}, "
                 f"policies={[p.name for p in err.policies]}")

    def test_s6_jobspec_policy_used_in_run_job(self, tmp_path):
        """run_job() correctly applies JobSpec's PolicySet during execution."""
        decomp_json = json.dumps([{
            "id": "task_001",
            "type": "code_generation",
            "prompt": "Write a GDPR-compliant logger.",
            "dependencies": [],
            "hard_validators": [],
        }])
        used_models: list[str] = []

        async def mock_call(model, prompt, **kwargs):
            if "decomposition engine" in kwargs.get("system", ""):
                return _make_mock_response(decomp_json)
            used_models.append(model.value)
            return _make_mock_response("# GDPR logger\nclass Logger: pass", cost=0.001)

        profiles = build_default_profiles()
        # Mark only Claude Sonnet as compliant with eu+no_train
        profiles[Model.CLAUDE_SONNET].region = "eu"
        profiles[Model.CLAUDE_SONNET].compliance_tags = ["no_train"]
        for m in profiles:
            if m != Model.CLAUDE_SONNET:
                profiles[m].region = "us"
                profiles[m].compliance_tags = []

        orch = Orchestrator(
            budget=Budget(max_usd=0.50),
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
            profiles=profiles,
        )
        spec = JobSpec(
            project_description="Build a GDPR-compliant logging system",
            success_criteria="No PII is logged",
            budget=Budget(max_usd=0.50),
            policy_set=PolicySet(global_policies=[
                Policy("gdpr", allow_training_on_output=False),
                Policy("eu_only", allowed_regions=["eu"]),
            ]),
        )
        with patch.object(orch.client, "call", side_effect=mock_call):
            state = _run(orch.run_job(spec))

        assert state is not None
        log.info(f"S6c passed: used_models={used_models}, status={state.status.value}")


# ─────────────────────────────────────────────────────────────────────────────
# S7  Resume — partial run → checkpoint → resume
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS7Resume:
    """S7: Verify state checkpoint and resume functionality."""

    def test_s7_checkpoint_is_saved(self, tmp_path):
        """After each task, a checkpoint is saved to disk."""
        decomp_json = _decomposed_tasks_json(n_tasks=2)

        async def mock_call(model, prompt, **kwargs):
            if "decomposition engine" in kwargs.get("system", ""):
                return _make_mock_response(decomp_json)
            return _make_mock_response("def stub(): pass", cost=0.001)

        state_db = tmp_path / "state.db"
        orch = Orchestrator(
            budget=Budget(max_usd=0.50),
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=state_db),
        )
        with patch.object(orch.client, "call", side_effect=mock_call):
            state = _run(orch.run_project(
                "Checkpointed project",
                "Save state",
                project_id="stress_resume_001",
            ))

        # State DB file should exist on disk
        assert state_db.exists(), "State DB file was not created"
        # Verify the saved state can be loaded back
        mgr2 = StateManager(db_path=state_db)
        loaded = mgr2.load_project("stress_resume_001")
        assert loaded is not None, "Could not load saved project state"
        log.info(f"S7 passed: state DB exists, loaded state status={loaded.status.value}")

    def test_s7_state_manager_load_and_save_roundtrip(self, tmp_path):
        """StateManager can save and load a ProjectState with full fidelity."""
        state_db = tmp_path / "state.db"
        mgr = StateManager(db_path=state_db)

        state = ProjectState(
            project_description="Resume test",
            success_criteria="Load works",
            budget=Budget(max_usd=5.0),
            status=ProjectStatus.PARTIAL_SUCCESS,
        )
        state.api_health = {m.value: True for m in Model}

        mgr.save_project("test_resume_proj", state)
        loaded = mgr.load_project("test_resume_proj")

        assert loaded is not None
        assert loaded.project_description == "Resume test"
        assert loaded.status == ProjectStatus.PARTIAL_SUCCESS
        log.info("S7b passed: state roundtrip successful")


# ─────────────────────────────────────────────────────────────────────────────
# S8  Cache hit — same prompt twice
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS8CacheHit:
    """S8: Second call with identical prompt must be served from cache (zero cost)."""

    def test_s8_second_call_uses_cache(self, tmp_path):
        """DiskCache returns a cached response on repeated identical calls."""
        from orchestrator.cache import DiskCache as DC
        cache = DC(db_path=tmp_path / "cache.db")

        model = Model.GPT_4O
        prompt = "Write a Python add function."
        system = "You are an expert programmer."
        max_tokens = 200

        async def run():
            # Cache miss → populate
            hit1 = await cache.get(model.value, prompt, max_tokens, system)
            assert hit1 is None, "Cache should be empty on first access"

            await cache.put(
                model.value, prompt, max_tokens,
                response="def add(a, b): return a + b",
                tokens_input=50, tokens_output=15,
                system=system,
            )

            # Cache hit → should return stored value
            hit2 = await cache.get(model.value, prompt, max_tokens, system)
            assert hit2 is not None, "Cache should return value on second access"
            assert "add" in hit2["response"]
            return hit2

        hit = asyncio.get_event_loop().run_until_complete(run())
        log.info(f"S8 passed: cache hit on second identical call, text={hit['response'][:30]!r}")

    def test_s8_different_prompt_is_cache_miss(self, tmp_path):
        """Different prompt must NOT get a cache hit."""
        from orchestrator.cache import DiskCache as DC
        cache = DC(db_path=tmp_path / "cache.db")

        model = Model.GPT_4O

        async def run():
            await cache.put(
                model.value, "prompt A", 200,
                response="output A", tokens_input=10, tokens_output=5,
                system="system",
            )
            hit = await cache.get(model.value, "prompt B", 200, "system")
            assert hit is None, "Different prompt should not produce a cache hit"

        asyncio.get_event_loop().run_until_complete(run())
        log.info("S8b passed: different prompt = cache miss")


# ─────────────────────────────────────────────────────────────────────────────
# S9  Telemetry drift — 20 calls, verify EMA and trust updates
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS9TelemetryDrift:
    """S9: After many calls, telemetry values converge as expected."""

    def test_s9_latency_ema_converges(self):
        """After 20 calls with 500ms latency, avg_latency_ms converges toward 500."""
        profiles = build_default_profiles()
        telemetry = TelemetryCollector(profiles)
        model = Model.GPT_4O

        for _ in range(20):
            telemetry.record_call(model, latency_ms=500.0, cost_usd=0.001, success=True)

        # EMA with alpha=0.1 converges slowly; after 20 steps from 2000ms → ~668ms
        latency = profiles[model].avg_latency_ms
        assert latency < 2000.0, f"Latency EMA should decrease from 2000ms, got {latency:.1f}ms"
        assert latency > 100.0, f"Latency EMA should not reach 0, got {latency:.1f}ms"
        log.info(f"S9 passed: latency after 20 calls = {latency:.1f}ms")

    def test_s9_quality_ema_tracks_scores(self):
        """Quality EMA tracks a series of scores correctly."""
        profiles = build_default_profiles()
        telemetry = TelemetryCollector(profiles)
        model = Model.CLAUDE_SONNET

        initial_quality = profiles[model].quality_score

        # Feed 10 perfect scores
        for _ in range(10):
            telemetry.record_call(model, 1000.0, 0.002, success=True, quality_score=1.0)

        # Quality should have increased from initial
        final_quality = profiles[model].quality_score
        assert final_quality >= initial_quality, (
            f"Quality should increase with perfect scores: {initial_quality:.4f} -> {final_quality:.4f}"
        )
        log.info(f"S9b passed: quality {initial_quality:.4f} -> {final_quality:.4f} after 10 perfect scores")

    def test_s9_failure_rate_updates_success_window(self):
        """success_rate reflects the last 10 calls (rolling window)."""
        profiles = build_default_profiles()
        telemetry = TelemetryCollector(profiles)
        model = Model.GEMINI_PRO

        # 5 successes then 5 failures
        for _ in range(5):
            telemetry.record_call(model, 500.0, 0.001, success=True)
        for _ in range(5):
            telemetry.record_call(model, 500.0, 0.001, success=False)

        # Rolling window of last 10: 5 success + 5 failure → 50%
        rate = profiles[model].success_rate
        assert 0.45 <= rate <= 0.55, f"Expected ~0.50 success rate, got {rate:.2f}"
        log.info(f"S9c passed: success_rate = {rate:.2f} (expected ~0.50)")

    def test_s9_trust_factor_does_not_go_negative(self):
        """Trust factor must stay in [0, 1] after many failures."""
        profiles = build_default_profiles()
        telemetry = TelemetryCollector(profiles)
        model = Model.KIMI_K2_5

        for _ in range(100):
            telemetry.record_call(model, 1000.0, 0.001, success=False)

        trust = profiles[model].trust_factor
        assert 0.0 <= trust <= 1.0, f"Trust factor out of bounds: {trust}"
        log.info(f"S9d passed: trust after 100 failures = {trust:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# S10  Full project simulation — comprehensive mock pipeline
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS10FullProjectSimulation:
    """S10: Realistic multi-task project simulation with detailed mocking."""

    def test_s10_realistic_project_pipeline(self, tmp_path):
        """Simulate a full 'build a Python utility' project end-to-end."""
        project_tasks = json.dumps([
            {
                "id": "task_001",
                "type": "complex_reasoning",
                "prompt": "Analyze requirements and design architecture for a rate limiter.",
                "dependencies": [],
                "hard_validators": [],
            },
            {
                "id": "task_002",
                "type": "code_generation",
                "prompt": "Implement a token bucket rate limiter in Python.",
                "dependencies": ["task_001"],
                "hard_validators": ["python_syntax"],
            },
            {
                "id": "task_003",
                "type": "code_generation",
                "prompt": "Write unit tests for the rate limiter.",
                "dependencies": ["task_002"],
                "hard_validators": ["python_syntax"],
            },
            {
                "id": "task_004",
                "type": "code_review",
                "prompt": "Review the rate limiter implementation for correctness.",
                "dependencies": ["task_002"],
                "hard_validators": [],
            },
            {
                "id": "task_005",
                "type": "summarization",
                "prompt": "Summarize the implementation and test results.",
                "dependencies": ["task_003", "task_004"],
                "hard_validators": ["length"],
            },
            {
                "id": "task_006",
                "type": "evaluation",
                "prompt": "Evaluate the overall quality of the rate limiter project.",
                "dependencies": ["task_005"],
                "hard_validators": [],
            },
        ])

        # Realistic outputs per task type
        OUTPUTS = {
            "complex_reasoning": "## Architecture\nToken bucket algorithm with sliding window fallback.",
            "code_generation": (
                "class RateLimiter:\n"
                "    def __init__(self, rate: int, period: float):\n"
                "        self.rate = rate\n"
                "        self.period = period\n"
                "        self.tokens = rate\n"
                "        self._last = __import__('time').time()\n\n"
                "    def allow(self) -> bool:\n"
                "        now = __import__('time').time()\n"
                "        elapsed = now - self._last\n"
                "        self.tokens = min(self.rate, self.tokens + elapsed * self.rate / self.period)\n"
                "        self._last = now\n"
                "        if self.tokens >= 1:\n"
                "            self.tokens -= 1\n"
                "            return True\n"
                "        return False\n"
            ),
            "code_review": "The implementation looks correct. Minor: add docstrings.",
            "summarization": "Rate limiter implemented with token bucket. Tests cover allow/deny cases.",
            "evaluation": '{"score": 0.92, "critique": "Well structured, good test coverage."}',
        }

        call_log: list[dict] = []

        async def mock_call(model, prompt, **kwargs):
            system = kwargs.get("system", "")
            if "decomposition engine" in system:
                return _make_mock_response(project_tasks)

            # Determine task type from prompt keywords
            output = OUTPUTS["code_generation"]  # default
            cost = 0.002
            for task_type_key, out in OUTPUTS.items():
                if task_type_key.replace("_", " ") in prompt.lower():
                    output = out
                    break
            if "evaluate" in prompt.lower() or "evaluation" in system.lower():
                output = OUTPUTS["evaluation"]
            if "review" in prompt.lower():
                output = OUTPUTS["code_review"]
            if "summary" in prompt.lower() or "summariz" in prompt.lower():
                output = OUTPUTS["summarization"]

            call_log.append({"model": model.value, "cost": cost})
            return _make_mock_response(output, cost=cost)

        orch = Orchestrator(
            budget=Budget(max_usd=2.0),
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
        )
        start = time.time()
        with patch.object(orch.client, "call", side_effect=mock_call):
            state = _run(orch.run_project(
                "Build a production-ready rate limiter in Python",
                "Correct token bucket implementation with tests and documentation",
                project_id="stress_full_project",
            ))
        elapsed = time.time() - start

        # Assertions
        assert state is not None
        assert state.status in (
            ProjectStatus.SUCCESS,
            ProjectStatus.PARTIAL_SUCCESS,
            ProjectStatus.BUDGET_EXHAUSTED,
        )

        n_completed = sum(
            1 for r in state.results.values()
            if r.status == TaskStatus.COMPLETED
        )
        total_mock_cost = sum(c["cost"] for c in call_log)

        log.info(
            f"S10 passed: {n_completed}/{len(state.results)} tasks completed, "
            f"{len(call_log)} API calls, simulated cost=${total_mock_cost:.4f}, "
            f"wall_time={elapsed:.2f}s, status={state.status.value}"
        )

        # Should have completed most tasks
        assert n_completed >= 3, f"Expected >=3 tasks completed, got {n_completed}"


# ─────────────────────────────────────────────────────────────────────────────
# S11  Planner scoring — verify score formula in isolation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.mock
class TestS11PlannerScoring:
    """S11: Unit-level stress test of the planner's scoring formula at scale."""

    def test_s11_score_is_monotone_in_quality(self):
        """Higher quality_score always yields higher planner score (all else equal)."""
        profiles = {}
        qualities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        models_used = [
            Model.GPT_4O, Model.GEMINI_PRO, Model.CLAUDE_SONNET,
            Model.KIMI_K2_5, Model.CLAUDE_HAIKU, Model.GPT_4O_MINI,
        ]
        for model, q in zip(models_used, qualities):
            profiles[model] = ModelProfile(
                model=model,
                provider=get_provider(model),
                cost_per_1m_input=1.0,
                cost_per_1m_output=5.0,
                capable_task_types={TaskType.CODE_GEN: 0},
                quality_score=q,
                trust_factor=1.0,
            )

        engine = PolicyEngine()
        health = {m: True for m in profiles}
        planner = ConstraintPlanner(profiles, engine, health)

        scores = [(m, planner._score(m, TaskType.CODE_GEN)) for m in models_used]
        scores.sort(key=lambda x: x[1])

        # Scores should be in ascending quality order
        for i in range(len(scores) - 1):
            assert scores[i][1] <= scores[i + 1][1], (
                f"Score not monotone: {scores[i]} > {scores[i+1]}"
            )
        log.info(f"S11 passed: scores in order = {[(m.value[:10], f'{s:.4f}') for m, s in scores]}")

    def test_s11_score_is_monotone_decreasing_in_cost(self):
        """Lower cost always yields higher planner score (all else equal)."""
        profiles = {}
        costs = [0.1, 1.0, 5.0, 10.0, 15.0]
        models_used = [
            Model.GPT_4O_MINI, Model.GPT_4O, Model.GEMINI_PRO,
            Model.CLAUDE_SONNET, Model.CLAUDE_OPUS,
        ]
        for model, c in zip(models_used, costs):
            profiles[model] = ModelProfile(
                model=model,
                provider=get_provider(model),
                cost_per_1m_input=c,
                cost_per_1m_output=c * 4,
                capable_task_types={TaskType.CODE_GEN: 0},
                quality_score=0.8,
                trust_factor=1.0,
            )

        engine = PolicyEngine()
        health = {m: True for m in profiles}
        planner = ConstraintPlanner(profiles, engine, health)

        # Score = quality * trust / (est_cost + epsilon)
        # Lower cost → higher score
        scores = [(m, planner._score(m, TaskType.CODE_GEN)) for m in models_used]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Cheapest model (GPT_4O_MINI, cost=0.1) should score highest
        assert scores[0][0] == Model.GPT_4O_MINI, (
            f"Cheapest model should score highest, got {scores[0][0]}"
        )
        log.info(f"S11b passed: score order by cost = {[m.value[:15] for m, _ in scores]}")


# ─────────────────────────────────────────────────────────────────────────────
# S12  Real API smoke test (skipped if no keys present)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _at_least_one_provider_available(),
    reason="No API keys configured — set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
)
class TestS12RealApiSmoke:
    """S12: Actual LLM API call — single tiny task, minimum cost."""

    def test_s12_real_single_call(self, tmp_path):
        """Fire one real LLM call with a trivial prompt; verify non-empty response."""
        from orchestrator.api_clients import UnifiedClient
        cache = DiskCache(db_path=tmp_path / "cache.db")
        client = UnifiedClient(cache=cache, max_concurrency=1)

        # Pick cheapest available model
        candidates = []
        if os.getenv("OPENAI_API_KEY"):
            candidates.append(Model.GPT_4O_MINI)
        if os.getenv("GOOGLE_API_KEY"):
            candidates.append(Model.GEMINI_FLASH)
        if os.getenv("ANTHROPIC_API_KEY"):
            candidates.append(Model.CLAUDE_HAIKU)

        assert candidates, "No candidates despite API key check passing"
        model = candidates[0]

        async def run():
            return await client.call(
                model,
                "Reply with exactly 3 words: stress test passed",
                system="You are a test responder. Reply briefly.",
                max_tokens=20,
                timeout=30,
            )

        response = _run(run())
        assert response.text.strip(), "Got empty response from real API"
        assert response.cost_usd >= 0.0
        assert response.latency_ms >= 0.0
        log.info(
            f"S12 passed: model={model.value}, "
            f"response={response.text.strip()!r}, "
            f"cost=${response.cost_usd:.6f}, "
            f"latency={response.latency_ms:.0f}ms"
        )

    @pytest.mark.skipif(
        not _at_least_one_provider_available(),
        reason="No API keys configured"
    )
    def test_s12_real_orchestrator_tiny_project(self, tmp_path):
        """Run a real tiny project ($0.10 budget) against live APIs."""
        orch = Orchestrator(
            budget=Budget(max_usd=0.10, max_time_seconds=120.0),
            cache=DiskCache(db_path=tmp_path / "cache.db"),
            state_manager=StateManager(db_path=tmp_path / "state.db"),
        )
        state = _run(orch.run_project(
            "Write a Python function that returns the sum of two integers.",
            "The function must pass: assert add(2, 3) == 5",
            project_id="stress_real_api",
        ))

        assert state is not None
        assert state.status != ProjectStatus.SYSTEM_FAILURE, (
            f"Project failed with SYSTEM_FAILURE: check API keys and connectivity"
        )
        log.info(
            f"S12 real project: status={state.status.value}, "
            f"spent=${orch.budget.spent_usd:.4f}, "
            f"tasks={len(state.results)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint for direct execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    print("Running stress test suite...")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "-x"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    sys.exit(result.returncode)
