"""
Orchestrator Engine — Core Control Loop
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Implements the full generate → critique → revise → evaluate pipeline
with cross-model review, deterministic validation, budget enforcement,
plateau detection, and fallback routing.

FIX #5:  Budget checked within iteration loop (mid-task), not just pre-task.
FIX #6:  Topological sort uses collections.deque instead of list.sort()+pop(0).
FIX #7:  Resume restores persisted budget state instead of creating fresh Budget.
FIX #10: All StateManager calls are now awaited (async migration).
FEAT:    TelemetryCollector + ConstraintPlanner wired at init.
FEAT:    TaskResult.tokens_used populated from APIResponse.
FEAT:    run_job(spec) entry point for policy-driven orchestration.
FEAT:    Budget phase partition enforcement (warn + soft-halt at 2× soft cap).
FEAT:    Dependency context truncation warning.
FEAT:    Decomposition retried once with different model on JSON parse failure.
FEAT:    Circuit breaker — model marked unhealthy after 3 consecutive failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict, deque
from typing import Optional

from .models import (
    Budget, BUDGET_PARTITIONS, FALLBACK_CHAIN, Model, ProjectState,
    ProjectStatus, ROUTING_TABLE, Task, TaskResult, TaskStatus, TaskType,
    get_max_iterations, get_provider, estimate_cost, build_default_profiles,
)
from .api_clients import UnifiedClient, APIResponse
from .validators import run_validators, async_run_validators, all_validators_pass, ValidationResult
from .cache import DiskCache
from .state import StateManager
from .policy import ModelProfile, Policy, PolicySet, JobSpec
from .policy_engine import PolicyEngine
from .planner import ConstraintPlanner
from .telemetry import TelemetryCollector
from .audit import AuditLog
from .optimization import OptimizationBackend
from .hooks import HookRegistry, EventType
from .metrics import MetricsExporter
from .agents import TaskChannel
from .cost import BudgetHierarchy, CostPredictor
from .tracing import traced_task, get_tracer, TracingConfig, configure_tracing

logger = logging.getLogger("orchestrator")


class Orchestrator:
    """
    Main orchestration engine.

    Invariants maintained:
    1. Cross-review always uses different provider than generator
    2. Deterministic validators override LLM scores
    3. Budget ceiling is never exceeded (checked mid-task per iteration)
    4. State is checkpointed after each task
    5. Plateau detection prevents runaway iteration
    """

    # Circuit breaker: model is marked unhealthy after this many consecutive errors
    _CIRCUIT_BREAKER_THRESHOLD: int = 3

    def __init__(self, budget: Optional[Budget] = None,
                 cache: Optional[DiskCache] = None,
                 state_manager: Optional[StateManager] = None,
                 max_concurrency: int = 3,
                 max_parallel_tasks: int = 3,
                 budget_hierarchy: Optional["BudgetHierarchy"] = None,
                 cost_predictor: Optional["CostPredictor"] = None,
                 tracing_cfg: Optional["TracingConfig"] = None):
        self.budget = budget or Budget()
        self.cache = cache or DiskCache()
        self.state_mgr = state_manager or StateManager()
        self.client = UnifiedClient(cache=self.cache, max_concurrency=max_concurrency)
        self.api_health: dict[Model, bool] = {m: True for m in Model}
        self.results: dict[str, TaskResult] = {}
        self._project_id: str = ""
        # Max tasks executed concurrently within one dependency level.
        # JobSpec.max_parallel_tasks overrides this via run_job().
        self._max_parallel_tasks: int = max(1, max_parallel_tasks)

        # Circuit breaker counters — consecutive failures per model
        self._consecutive_failures: dict[Model, int] = {m: 0 for m in Model}

        for model in Model:
            if not self.client.is_available(model):
                self.api_health[model] = False
                logger.warning(f"{model.value}: provider SDK/key not available")

        # Policy-driven components (initialised with default profiles from static tables)
        self._profiles: dict[Model, ModelProfile] = build_default_profiles()
        self._audit_log = AuditLog()
        self._policy_engine = PolicyEngine(audit_log=self._audit_log)
        self._planner = ConstraintPlanner(
            profiles=self._profiles,
            policy_engine=self._policy_engine,
            api_health=self.api_health,
        )
        self._telemetry = TelemetryCollector(self._profiles)
        # Active policy set — replaced by run_job(); empty = no restrictions
        self._active_policies: PolicySet = PolicySet()
        # Context truncation limit per dependency (chars) — configurable.
        # Raised from 20000: code_generation outputs routinely reach 25000+ chars
        # and truncation causes code_review tasks to miss the tail of the source,
        # leading the LLM to claim "source code was not provided".
        self.context_truncation_limit: int = 40000
        # Improvement 3: event hooks + metrics exporter
        self._hook_registry: HookRegistry = HookRegistry()
        self._metrics_exporter: Optional[MetricsExporter] = None
        # Improvement 5: named TaskChannels for inter-task messaging
        self._channels: dict[str, TaskChannel] = {}
        # Improvement 6: cross-run budget hierarchy + adaptive cost predictor
        self._budget_hierarchy: Optional[BudgetHierarchy] = budget_hierarchy
        self._cost_predictor: Optional[CostPredictor] = cost_predictor
        # Task 2: streaming event bus (None unless run_project_streaming() is active)
        self._event_bus: Optional["ProjectEventBus"] = None
        # Task 6: adaptive router v2 — circuit breaker with degraded/disabled states
        from .adaptive_router import AdaptiveRouter
        self._adaptive_router = AdaptiveRouter()
        # Task 7: configure OpenTelemetry tracing if a config was provided
        if tracing_cfg is not None:
            configure_tracing(tracing_cfg)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def set_optimization_backend(self, backend: "OptimizationBackend") -> None:
        """Swap the ConstraintPlanner's optimization strategy at runtime."""
        self._planner.set_backend(backend)

    @property
    def audit_log(self) -> "AuditLog":
        """Read-only access to the policy audit log."""
        return self._audit_log

    @property
    def cost_predictor(self) -> Optional["CostPredictor"]:
        """Read-only access to the CostPredictor, if one was configured."""
        return self._cost_predictor

    def add_hook(self, event: str, callback) -> None:
        """Register an event hook callback. See orchestrator.hooks.EventType for event names."""
        self._hook_registry.add(event, callback)

    def set_metrics_exporter(self, exporter: "MetricsExporter") -> None:
        """Set the MetricsExporter to use when export_metrics() is called."""
        self._metrics_exporter = exporter

    def export_metrics(self) -> None:
        """Export live per-model telemetry stats via the configured MetricsExporter."""
        if self._metrics_exporter is None:
            return
        self._metrics_exporter.export(self._build_metrics_dict())

    def get_channel(self, name: str) -> "TaskChannel":
        """Return the named TaskChannel, creating it lazily on first access."""
        if name not in self._channels:
            self._channels[name] = TaskChannel()
        return self._channels[name]

    def _build_metrics_dict(self) -> dict:
        """Build a per-model metrics dict from live ModelProfile data."""
        result: dict = {}
        for model, profile in self._profiles.items():
            result[model.value] = {
                "call_count":           profile.call_count,
                "failure_count":        profile.failure_count,
                "success_rate":         profile.success_rate,
                "avg_latency_ms":       profile.avg_latency_ms,
                "latency_p95_ms":       profile.latency_p95_ms,
                "quality_score":        profile.quality_score,
                "trust_factor":         profile.trust_factor,
                "avg_cost_usd":         profile.avg_cost_usd,
                "validator_fail_count": profile.validator_fail_count,
                "error_rate":           self._telemetry.error_rate(model),
            }
        return result

    async def run_project(self, project_description: str,
                          success_criteria: str,
                          project_id: str = "",
                          app_profile: Optional["AppProfile"] = None) -> ProjectState:
        """Main entry point. Decomposes project → executes tasks → returns state."""
        tracer = get_tracer()
        with tracer.start_as_current_span("run_project") as span:
            span.set_attribute("project.description", project_description[:200])
            if not project_id:
                project_id = hashlib.md5(
                    f"{project_description[:100]}{time.time()}".encode()
                ).hexdigest()[:12]
            self._project_id = project_id

            logger.info(f"Starting project {project_id}")
            logger.info(f"Budget: ${self.budget.max_usd}, {self.budget.max_time_seconds}s")

            try:
                # Check if resumable
                existing = await self.state_mgr.load_project(project_id)
                if existing and existing.status == ProjectStatus.PARTIAL_SUCCESS:
                    logger.info(f"Resuming project {project_id} from checkpoint")
                    state = await self._resume_project(existing)
                    await self.state_mgr.save_project(project_id, state)
                    self._log_summary(state)
                    return state

                # Phase 1: Decompose
                tasks = await self._decompose(project_description, success_criteria,
                                              app_profile=app_profile)
                if not tasks:
                    return self._make_state(
                        project_description, success_criteria, {},
                        ProjectStatus.SYSTEM_FAILURE
                    )

                # Topological sort
                execution_order = self._topological_sort(tasks)
                logger.info(f"Execution order: {execution_order}")

                # Emit ProjectStarted streaming event
                if self._event_bus:
                    from .streaming import ProjectStarted
                    await self._event_bus.publish(ProjectStarted(
                        project_id=self._project_id,
                        total_tasks=len(tasks),
                        budget_usd=self.budget.max_usd,
                    ))

                # Phase 2-5: Execute
                state = await self._execute_all(
                    tasks, execution_order, project_description, success_criteria
                )

                # Final status determination
                state.execution_order = execution_order
                state.status = self._determine_final_status(state)
                await self.state_mgr.save_project(project_id, state)

                self._log_summary(state)

                # Emit ProjectCompleted streaming event
                if self._event_bus:
                    from .streaming import ProjectCompleted
                    completed_count = sum(1 for r in self.results.values()
                                          if r.status != TaskStatus.FAILED)
                    failed_count = sum(1 for r in self.results.values()
                                       if r.status == TaskStatus.FAILED)
                    await self._event_bus.publish(ProjectCompleted(
                        project_id=self._project_id,
                        status=state.status.value,
                        total_cost_usd=self.budget.spent_usd,
                        elapsed_seconds=self.budget.elapsed_seconds,
                        tasks_completed=completed_count,
                        tasks_failed=failed_count,
                    ))

                return state

            finally:
                # Always close both DB connections so aiosqlite background threads
                # finish their callbacks before asyncio.run() closes the loop.
                await self.state_mgr.close()
                await self.cache.close()

    async def run_job(self, spec: JobSpec) -> ProjectState:
        """
        Policy-driven entry point. Accepts a JobSpec that bundles project
        description, success criteria, budget, quality targets, and policies.

        The active PolicySet is threaded through model selection so that
        ConstraintPlanner enforces compliance on every API call.
        """
        self.budget = spec.budget
        self._active_policies = spec.policy_set
        # JobSpec may override the per-task parallelism limit
        if spec.max_parallel_tasks > 0:
            self._max_parallel_tasks = spec.max_parallel_tasks
        # BudgetHierarchy pre-flight check (Improvement 6)
        if self._budget_hierarchy is not None:
            job_id = getattr(spec, "job_id", "") or ""
            team   = getattr(spec, "team",   "") or ""
            if not self._budget_hierarchy.can_afford_job(job_id, team, spec.budget.max_usd):
                raise ValueError(
                    f"BudgetHierarchy rejects job '{job_id}': "
                    "org/team/job limits would be exceeded"
                )
        state = await self.run_project(
            project_description=spec.project_description,
            success_criteria=spec.success_criteria,
        )
        # Charge actual spend to BudgetHierarchy so cross-run caps are enforced.
        if self._budget_hierarchy is not None:
            actual_spend = self.budget.max_usd - self.budget.remaining_usd
            job_id = getattr(spec, "job_id", "") or ""
            team   = getattr(spec, "team",   "") or ""
            self._budget_hierarchy.charge_job(job_id, team, actual_spend)
        return state

    async def run_project_streaming(
        self,
        project_description: str,
        success_criteria: str,
        project_id: str = "",
    ):
        """
        Streaming variant of run_project().
        Yields StreamEvent objects as execution progresses.
        The final event is always ProjectCompleted.
        """
        from .streaming import ProjectEventBus

        self._event_bus = ProjectEventBus()
        subscription = self._event_bus.subscribe()

        async def _run() -> None:
            bus = self._event_bus
            try:
                await self.run_project(project_description, success_criteria, project_id)
            finally:
                await bus.close()
                if self._event_bus is bus:
                    self._event_bus = None

        task = asyncio.create_task(_run())

        async for event in subscription:
            yield event

        await task  # propagate any unhandled exceptions

    # ─────────────────────────────────────────
    # Phase 1: Decomposition
    # ─────────────────────────────────────────

    async def _decompose(self, project: str, criteria: str,
                          app_profile: Optional["AppProfile"] = None) -> dict[str, Task]:
        """Use cheapest capable model to break project into atomic tasks."""
        valid_types = [t.value for t in TaskType]

        # Build optional app-context block injected into the prompt
        app_context_block = ""
        if app_profile is not None:
            from orchestrator.scaffold import _TEMPLATE_MAP
            from orchestrator.scaffold.templates import generic
            template_files = _TEMPLATE_MAP.get(app_profile.app_type, generic.FILES)
            scaffold_list = "\n".join(f"  - {p}" for p in sorted(template_files))
            tech_stack_str = ", ".join(app_profile.tech_stack) if app_profile.tech_stack else "unknown"
            app_context_block = f"""
APP_TYPE: {app_profile.app_type}
TECH_STACK: {tech_stack_str}
SCAFFOLD_FILES (already exist — fill or extend these):
{scaffold_list}

Each task JSON element MUST also include:
- "target_path": the relative file path this task writes (e.g. "app/page.tsx").
  Use the exact scaffold paths listed above where applicable.
  Tasks producing non-file outputs (code_review, evaluation) use target_path: "".
- "tech_context": brief note on the tech stack relevant to this specific file.
"""

        prompt = f"""You are a project decomposition engine. Break this project into
atomic, executable tasks.

PROJECT: {project}

SUCCESS CRITERIA: {criteria}
{app_context_block}
Return ONLY a JSON array. Each element must have:
- "id": string (e.g., "task_001")
- "type": one of {valid_types}
- "prompt": detailed instruction for the task executor
- "dependencies": list of task id strings this depends on (empty if none)
- "hard_validators": list of validator names — ONLY use these for code tasks:
  - "python_syntax": only for code_generation tasks that produce Python code
  - "json_schema": only for tasks that must return valid JSON
  - "pytest": only for code_generation tasks with runnable tests
  - "ruff": only for code_generation tasks requiring lint checks
  - "latex": only for tasks producing LaTeX documents
  - "length": for tasks requiring minimum/maximum output length
  - Use [] (empty list) for non-code tasks (reasoning, writing, analysis, evaluation)

RULES:
- Tasks must be atomic (one clear deliverable each)
- Dependencies must form a DAG (no cycles)
- Include code_review tasks after code_generation tasks
- Include at least one evaluation task at the end
- 5-15 tasks total for a medium project
- Do NOT add hard_validators to reasoning, writing, analysis, or evaluation tasks

Return ONLY the JSON array, no markdown fences, no explanation."""

        decomp_system = "You are a precise project decomposition engine. Output only valid JSON."
        model = self._get_cheapest_available()

        async def _try_decompose(m: Model) -> dict[str, Task]:
            resp = await self.client.call(
                m, prompt, system=decomp_system, max_tokens=8192, timeout=180,
                bypass_cache=True,  # never reuse a cached decomposition response
            )
            self.budget.charge(resp.cost_usd, "decomposition")
            self._record_success(m, resp)
            result = self._parse_decomposition(resp.text)
            if not result:
                raise ValueError(f"Decomposition returned empty task list from {m.value}")
            return result

        # Try primary model, then fallback, with one retry on empty/malformed output
        for attempt, m in enumerate([model, self._get_fallback(model)]):
            if m is None:
                break
            try:
                return await _try_decompose(m)
            except (Exception, asyncio.CancelledError) as e:
                logger.error(f"Decomposition attempt {attempt + 1} with {m.value} failed: {e}")
                self._record_failure(m, error=e)
        logger.error("All decomposition attempts failed — returning empty task list")
        return {}

    def _parse_decomposition(self, text: str) -> dict[str, Task]:
        """Parse LLM output into Task objects with defensive handling."""
        text = text.strip()

        # Strip markdown fences (``` or ```json)
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        def _try_parse(s: str):
            """Attempt json.loads; on failure try progressively more aggressive fixes."""
            # 1. Direct parse
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
            # 2. Strip trailing commas before ] or } (common LLM mistake)
            cleaned = re.sub(r',\s*([}\]])', r'\1', s)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            # 3. Remove control characters (except \n \r \t)
            cleaned2 = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
            try:
                return json.loads(cleaned2)
            except json.JSONDecodeError:
                pass
            return None

        # Try full text first
        items = _try_parse(text)

        # If the top-level is a dict, look for a key that holds the array
        if isinstance(items, dict):
            for v in items.values():
                if isinstance(v, list):
                    items = v
                    break

        # Extract outermost [...] block and retry (greedy — captures the full tasks array)
        if not isinstance(items, list):
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                items = _try_parse(match.group())

        if not isinstance(items, list):
            logger.error(
                "Could not parse decomposition output as JSON. "
                f"Raw response (first 500 chars): {text[:500]!r}"
            )
            return {}

        tasks = {}
        for item in items:
            try:
                task_type = TaskType(item["type"])
                task = Task(
                    id=item["id"],
                    type=task_type,
                    prompt=item["prompt"],
                    dependencies=item.get("dependencies", []),
                    hard_validators=item.get("hard_validators", []),
                    target_path=item.get("target_path", ""),
                    tech_context=item.get("tech_context", ""),
                )
                tasks[task.id] = task
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed task: {e}")
                continue

        logger.info(f"Decomposed into {len(tasks)} tasks")
        return tasks

    # ─────────────────────────────────────────
    # Phase 2-5: Task Execution Loop
    # ─────────────────────────────────────────

    def _check_phase_budget(self, phase: str) -> None:
        """
        Warn when a phase exceeds its soft cap, and log an error when it
        reaches 2× the soft cap (runaway spend in one phase).
        The caps are soft — execution is not halted, but the warnings are
        visible in logs and can be acted upon by the operator.
        """
        spent = self.budget.phase_spent.get(phase, 0.0)
        cap = self.budget.phase_budget(phase)
        if cap <= 0:
            return
        ratio = spent / cap
        if ratio >= 2.0:
            logger.error(
                f"Phase '{phase}' spent ${spent:.4f} — "
                f"{ratio:.1f}× its soft cap of ${cap:.4f}. "
                f"Consider raising --budget or reducing task count."
            )
        elif ratio >= 1.0:
            logger.warning(
                f"Phase '{phase}' exceeded soft cap: "
                f"${spent:.4f} / ${cap:.4f} ({ratio:.0%})"
            )
            self._hook_registry.fire(
                EventType.BUDGET_WARNING,
                phase=phase, spent=spent, cap=cap, ratio=ratio,
            )

    async def _execute_all(self, tasks: dict[str, Task],
                            execution_order: list[str],
                            project_desc: str,
                            success_criteria: str) -> ProjectState:
        """
        Execute all tasks respecting dependencies, with intra-level parallelism.

        Tasks are grouped into dependency levels using _topological_levels().
        All tasks in the same level have no inter-dependencies and are executed
        concurrently up to self._max_parallel_tasks simultaneous coroutines.

        A semaphore limits concurrency so that API rate limits and memory usage
        remain manageable even when many tasks are eligible at once.
        """
        levels = self._topological_levels(tasks)
        semaphore = asyncio.Semaphore(self._max_parallel_tasks)

        async def _run_one(task_id: str) -> None:
            """Execute a single task under the concurrency semaphore."""
            async with semaphore:
                if not self.budget.can_afford(0.01):
                    logger.warning(f"Budget exhausted, skipping {task_id}")
                    self.results[task_id] = TaskResult(
                        task_id=task_id, output="", score=0.0,
                        model_used=Model.GPT_4O_MINI,
                        status=TaskStatus.FAILED,
                    )
                    return
                if not self.budget.time_remaining():
                    logger.warning(f"Time limit reached, skipping {task_id}")
                    self.results[task_id] = TaskResult(
                        task_id=task_id, output="", score=0.0,
                        model_used=Model.GPT_4O_MINI,
                        status=TaskStatus.FAILED,
                    )
                    return

                task = tasks[task_id]
                self._hook_registry.fire(EventType.TASK_STARTED, task_id=task_id, task=task)
                result = await self._execute_task(task)
                self.results[task_id] = result
                task.status = result.status
                self._hook_registry.fire(EventType.TASK_COMPLETED, task_id=task_id, result=result)

                for phase in ("generation", "cross_review", "evaluation"):
                    self._check_phase_budget(phase)

        for level_idx, level in enumerate(levels):
            if not self.budget.can_afford(0.01):
                logger.warning("Budget exhausted, halting before level %d", level_idx)
                break
            if not self.budget.time_remaining():
                logger.warning("Time limit reached, halting before level %d", level_idx)
                break

            # Filter tasks with unmet or failed dependencies.
            # A task is runnable only if ALL its dependencies completed or degraded.
            # If any dependency FAILED, downstream tasks are skipped — executing
            # them with missing/invalid context would propagate garbage output.
            runnable = []
            for task_id in level:
                dep_results = [
                    self.results.get(dep, TaskResult(dep, "", 0.0, Model.GPT_4O_MINI))
                    for dep in tasks[task_id].dependencies
                ]
                any_failed = any(r.status == TaskStatus.FAILED for r in dep_results)
                all_finished = all(
                    r.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED, TaskStatus.FAILED)
                    for r in dep_results
                )
                if any_failed:
                    failed_deps = [
                        r.task_id for r in dep_results
                        if r.status == TaskStatus.FAILED
                    ]
                    logger.warning(
                        f"Skipping {task_id}: dependencies failed: {failed_deps}"
                    )
                    self.results[task_id] = TaskResult(
                        task_id=task_id, output="", score=0.0,
                        model_used=Model.GPT_4O_MINI,
                        status=TaskStatus.FAILED,
                    )
                elif all_finished:
                    runnable.append(task_id)
                else:
                    logger.warning(f"Skipping {task_id}: unmet dependencies")
                    self.results[task_id] = TaskResult(
                        task_id=task_id, output="", score=0.0,
                        model_used=Model.GPT_4O_MINI,
                        status=TaskStatus.FAILED,
                    )

            if not runnable:
                continue

            parallel_count = len(runnable)
            if parallel_count > 1:
                logger.info(
                    "Executing level %d: %d tasks in parallel (max=%d): %s",
                    level_idx, parallel_count, self._max_parallel_tasks, runnable,
                )
            else:
                logger.info("Executing level %d: %s", level_idx, runnable)

            await asyncio.gather(*(_run_one(tid) for tid in runnable))

            # Checkpoint after each level completes
            state = self._make_state(project_desc, success_criteria, tasks,
                                     execution_order=execution_order)
            if runnable:
                await self.state_mgr.save_checkpoint(
                    self._project_id, runnable[-1], state
                )

        return self._make_state(project_desc, success_criteria, tasks,
                                execution_order=execution_order)

    async def _execute_task(self, task: Task) -> TaskResult:
        """
        Core loop: generate → critique → revise → evaluate
        With plateau detection, deterministic validation, and mid-task budget checks.
        """
        with traced_task(task.id, task.type.value) as span:
            span.set_attribute("task.description", task.prompt[:200])
            models = self._get_available_models(task.type)
            if not models:
                return TaskResult(
                    task_id=task.id, output="", score=0.0,
                    model_used=Model.GPT_4O_MINI,
                    status=TaskStatus.FAILED,
                )

            primary = models[0]
            reviewer = self._select_reviewer(primary, task.type)

            # Emit TaskStarted streaming event
            if self._event_bus:
                from .streaming import TaskStarted
                await self._event_bus.publish(TaskStarted(
                    task_id=task.id,
                    task_type=task.type.value,
                    model=primary.value,
                ))

            # Fire MODEL_SELECTED event so hooks can observe routing decisions
            self._hook_registry.fire(
                EventType.MODEL_SELECTED,
                task_id=task.id,
                model=primary.value,
                backend=get_provider(primary),
            )

            context = self._gather_dependency_context(task.dependencies)
            full_prompt = task.prompt
            if context:
                # For code_review tasks, the dependency context IS the source code
                # being reviewed. Make this explicit so the LLM doesn't claim
                # "source code was not provided".
                if task.type == TaskType.CODE_REVIEW:
                    full_prompt += (
                        f"\n\n--- SOURCE CODE TO REVIEW (from prior tasks) ---\n"
                        f"The following is the actual generated source code you must "
                        f"review. Do NOT claim the code was not provided.\n\n"
                        f"{context}"
                    )
                else:
                    full_prompt += f"\n\n--- CONTEXT FROM PRIOR TASKS ---\n{context}"

            best_output = ""
            best_score = 0.0
            best_critique = ""
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            degraded_count = 0
            scores_history: list[float] = []
            det_passed = True  # default: no validators = passed

            logger.info(f"Executing {task.id} ({task.type.value}): "
                         f"primary={primary.value}, reviewer={reviewer.value if reviewer else 'none'}")

            for iteration in range(task.max_iterations):
                # FIX #5: Mid-task budget check — estimate minimum cost for one
                # generate+critique+revise+evaluate cycle (~0.02 USD min)
                if not self.budget.can_afford(0.02):
                    logger.warning(
                        f"Budget insufficient mid-task for {task.id} "
                        f"at iteration {iteration} "
                        f"(remaining: ${self.budget.remaining_usd:.4f})"
                    )
                    break

                if not self.budget.time_remaining():
                    logger.warning(f"Time limit reached mid-task for {task.id}")
                    break

                # ── GENERATE ──
                # Kimi K2.5 and DeepSeek-R1 are chain-of-thought reasoning models whose
                # internal reasoning tokens count against max_tokens but don't appear in
                # content output. For code tasks on these models, double the token budget
                # (cap at 16384) to ensure complete output. Both also need longer timeouts.
                _provider = get_provider(primary)
                _is_reasoning_model = (
                    _provider == "kimi" or
                    (_provider == "deepseek" and primary.value == "deepseek-reasoner")
                )
                if _is_reasoning_model:
                    gen_timeout = 240
                    if task.type in (TaskType.CODE_GEN, TaskType.CODE_REVIEW):
                        # Double token budget: reasoning tokens eat into output budget
                        effective_max_tokens = min(task.max_output_tokens * 2, 16384)
                    else:
                        effective_max_tokens = task.max_output_tokens
                elif task.type in (TaskType.CODE_GEN, TaskType.CODE_REVIEW):
                    gen_timeout = 120
                    effective_max_tokens = task.max_output_tokens
                else:
                    gen_timeout = 60
                    effective_max_tokens = task.max_output_tokens
                try:
                    gen_response = await self.client.call(
                        primary, full_prompt,
                        system=f"You are an expert executing a {task.type.value} task. "
                               f"Produce high-quality, complete output.",
                        max_tokens=effective_max_tokens,
                        timeout=gen_timeout,
                    )
                    output = gen_response.text
                    gen_cost = gen_response.cost_usd
                    self.budget.charge(gen_cost, "generation")
                    if self._cost_predictor is not None:
                        self._cost_predictor.record(primary, task.type, gen_cost)
                    total_cost += gen_cost
                    total_input_tokens += gen_response.input_tokens
                    total_output_tokens += gen_response.output_tokens
                    self._record_success(primary, gen_response)
                except (Exception, asyncio.CancelledError) as e:
                    logger.error(f"Generation failed for {task.id}: {e}")
                    self._record_failure(primary, error=e)
                    degraded_count += 1
                    fb = self._get_fallback(primary)
                    if fb:
                        try:
                            gen_response = await self.client.call(
                                fb, full_prompt,
                                system=f"You are an expert executing a {task.type.value} task.",
                                max_tokens=effective_max_tokens,
                                timeout=gen_timeout,
                            )
                            output = gen_response.text
                            self.budget.charge(gen_response.cost_usd, "generation")
                            total_cost += gen_response.cost_usd
                            total_input_tokens += gen_response.input_tokens
                            total_output_tokens += gen_response.output_tokens
                            self._record_success(fb, gen_response)
                            primary = fb
                        except (Exception, asyncio.CancelledError) as e2:
                            logger.error(f"Fallback generation also failed: {e2}")
                            self._record_failure(fb, error=e2)
                            break
                    else:
                        break

                # FIX #5: Re-check budget after generation before critique
                if not self.budget.can_afford(0.005):
                    logger.warning(f"Budget depleted after generation for {task.id}")
                    scores_history.append(0.0)
                    if not best_output:
                        best_output = output
                    break

                # ── CRITIQUE (cross-model) ──
                critique = ""
                if reviewer and reviewer != primary:
                    # Reviewer token budget: reasoning models (Kimi K2.5, DeepSeek-R1)
                    # consume their token budget on internal chain-of-thought, so they
                    # need the same doubled budget as when generating. Standard models
                    # only need 800 tokens to produce a focused critique.
                    _rev_provider = get_provider(reviewer)
                    _reviewer_is_reasoning = (
                        _rev_provider == "kimi" or
                        (_rev_provider == "deepseek" and reviewer.value == "deepseek-reasoner")
                    )
                    if _reviewer_is_reasoning:
                        critique_max_tokens = min(task.max_output_tokens * 2, 8192)
                        critique_timeout = 240
                    else:
                        critique_max_tokens = 1200  # raised: 800 was too low for detailed reviews
                        critique_timeout = 60
                    try:
                        critique_response = await self.client.call(
                            reviewer,
                            f"Review this output for correctness, completeness, and quality. "
                            f"Be specific about flaws and suggest concrete improvements.\n\n"
                            f"ORIGINAL TASK: {task.prompt}\n\n"
                            f"OUTPUT TO REVIEW:\n{output}",
                            system="You are a critical reviewer. Find flaws, be specific.",
                            max_tokens=critique_max_tokens,
                            timeout=critique_timeout,
                        )
                        critique = critique_response.text
                        self.budget.charge(critique_response.cost_usd, "cross_review")
                        if self._cost_predictor is not None:
                            self._cost_predictor.record(primary, task.type, critique_response.cost_usd)
                        total_cost += critique_response.cost_usd
                    except (Exception, asyncio.CancelledError) as e:
                        logger.warning(f"Critique failed for {task.id}: {e}")
                        self.api_health[reviewer] = False
                        degraded_count += 1

                # ── REVISE (if critique exists) ──
                # Skip revision for reasoning models (Kimi K2.5, DeepSeek-R1): their
                # chain-of-thought makes revision calls as slow/expensive as generation.
                # Instead, embed the critique into the next iteration's prompt so the
                # model can self-correct on re-generation.
                if critique and not _is_reasoning_model:
                    try:
                        revise_response = await self.client.call(
                            primary,
                            f"Revise your output based on this critique. "
                            f"Address every specific issue raised.\n\n"
                            f"ORIGINAL TASK: {task.prompt}\n\n"
                            f"YOUR OUTPUT:\n{output}\n\n"
                            f"CRITIQUE:\n{critique}\n\n"
                            f"Produce the complete improved version.",
                            system=f"You are revising a {task.type.value} task based on peer review.",
                            max_tokens=effective_max_tokens,
                            timeout=gen_timeout,
                        )
                        output = revise_response.text
                        self.budget.charge(revise_response.cost_usd, "generation")
                        total_cost += revise_response.cost_usd
                    except (Exception, asyncio.CancelledError) as e:
                        logger.warning(f"Revision failed for {task.id}: {e}")
                elif critique and _is_reasoning_model:
                    # Embed critique into the next iteration's prompt for self-correction.
                    full_prompt = (
                        f"{task.prompt}\n\n"
                        f"--- PEER REVIEW FEEDBACK (incorporate in your response) ---\n"
                        f"{critique}\n"
                        f"--- END FEEDBACK ---"
                    )
                    if context:
                        full_prompt += f"\n\n--- CONTEXT FROM PRIOR TASKS ---\n{context}"
                    logger.debug(
                        f"{primary.value}: critique embedded into next iteration "
                        f"prompt for {task.id}"
                    )

                # ── DETERMINISTIC VALIDATION ──
                det_passed = True
                if task.hard_validators:
                    val_results = await async_run_validators(output, task.hard_validators)
                    det_passed = all_validators_pass(val_results)
                    if not det_passed:
                        failed = [v for v in val_results if not v.passed]
                        logger.warning(
                            f"Deterministic check failed for {task.id}: "
                            f"{[f'{v.validator_name}: {v.details}' for v in failed]}"
                        )
                        # Record validator failure in telemetry + fire hook (Improvement 3)
                        self._telemetry.record_validator_failure(primary)
                        self._hook_registry.fire(
                            EventType.VALIDATION_FAILED,
                            task_id=task.id,
                            model=primary.value,
                            validators=task.hard_validators,
                        )

                # ── EVALUATE ──
                if det_passed:
                    score = await self._evaluate(task, output)
                else:
                    score = 0.0

                self.budget.charge(0.0, "evaluation")
                scores_history.append(score)

                if score > best_score:
                    best_output = output
                    best_score = score
                    best_critique = critique

                # Emit TaskProgressUpdate streaming event
                if self._event_bus:
                    from .streaming import TaskProgressUpdate
                    await self._event_bus.publish(TaskProgressUpdate(
                        task_id=task.id,
                        iteration=iteration + 1,
                        score=score,
                        best_score=best_score,
                    ))

                logger.info(
                    f"  {task.id} iter {iteration + 1}: score={score:.3f} "
                    f"(best={best_score:.3f}, threshold={task.acceptance_threshold})"
                )

                # ── CONVERGENCE CHECKS ──
                if best_score >= task.acceptance_threshold:
                    logger.info(f"  {task.id}: threshold met at iteration {iteration + 1}")
                    break

                if len(scores_history) >= 2:
                    delta = abs(scores_history[-1] - scores_history[-2])
                    if delta < 0.02:
                        # Only stop on plateau if we have a usable score.
                        # If best_score is still below half the acceptance threshold,
                        # keep trying — the critique/revision cycle may still help.
                        if best_score >= task.acceptance_threshold * 0.5:
                            logger.info(f"  {task.id}: plateau detected (Δ={delta:.4f})")
                            break
                        elif len(scores_history) >= 3:
                            # After 3+ iterations with no improvement AND bad score,
                            # give up to avoid wasting budget.
                            logger.info(
                                f"  {task.id}: plateau at low score after "
                                f"{len(scores_history)} iters (Δ={delta:.4f}, "
                                f"best={best_score:.3f})"
                            )
                            break

            status = TaskStatus.COMPLETED if best_score >= task.acceptance_threshold else TaskStatus.DEGRADED
            if best_score == 0.0 and not det_passed:
                status = TaskStatus.FAILED

            # Feed final eval score back to telemetry so ConstraintPlanner re-ranks
            if best_score > 0.0:
                self._telemetry.record_call(
                    primary, latency_ms=0.0, cost_usd=0.0,
                    success=(status != TaskStatus.FAILED),
                    quality_score=best_score,
                )

            # Emit TaskCompleted or TaskFailed streaming event
            if self._event_bus:
                from .streaming import TaskCompleted, TaskFailed
                if status == TaskStatus.FAILED:
                    await self._event_bus.publish(TaskFailed(
                        task_id=task.id,
                        reason="all attempts failed",
                        model=primary.value,
                    ))
                else:
                    await self._event_bus.publish(TaskCompleted(
                        task_id=task.id,
                        score=best_score,
                        status=status,
                        model=primary.value,
                        cost_usd=total_cost,
                        iterations=len(scores_history),
                    ))

            span.set_attribute("task.status", status.value)
            span.set_attribute("task.score", best_score or 0.0)
            return TaskResult(
                task_id=task.id,
                output=best_output,
                score=best_score,
                model_used=primary,
                reviewer_model=reviewer,
                tokens_used={"input": total_input_tokens, "output": total_output_tokens},
                iterations=len(scores_history),
                cost_usd=total_cost,
                status=status,
                critique=best_critique,
                deterministic_check_passed=det_passed,
                degraded_fallback_count=degraded_count,
            )

    async def _evaluate(self, task: Task, output: str) -> float:
        """LLM-based scoring with self-consistency (2 runs, Δ ≤ 0.05)."""
        eval_models = self._get_available_models(TaskType.EVALUATE)
        if not eval_models:
            return 0.5

        eval_model = eval_models[0]
        eval_prompt = (
            f"Score this output on a scale of 0.0 to 1.0.\n"
            f"Evaluate: correctness, completeness, quality, adherence to task.\n\n"
            f"TASK: {task.prompt}\n"
            f"ACCEPTANCE THRESHOLD: {task.acceptance_threshold}\n\n"
            f"OUTPUT:\n{output}\n\n"
            f'Return ONLY JSON: {{"score": <float>, "reasoning": "<brief>"}}'
        )

        scores = []
        for run in range(2):
            try:
                response = await self.client.call(
                    eval_model, eval_prompt,
                    system="You are a precise evaluator. Score exactly, return only JSON.",
                    max_tokens=300,
                    temperature=0.1,
                    timeout=60,
                )
                self.budget.charge(response.cost_usd, "evaluation")
                score = self._parse_score(response.text)
                scores.append(score)
            except (Exception, asyncio.CancelledError) as e:
                logger.warning(f"Evaluation run {run + 1} failed: {e}")
                scores.append(0.5)

        if len(scores) == 2:
            delta = abs(scores[0] - scores[1])
            if delta > 0.05:
                logger.warning(
                    f"Evaluation inconsistency: {scores[0]:.3f} vs {scores[1]:.3f} "
                    f"(Δ={delta:.3f} > 0.05). Using lower score."
                )
                return min(scores)
            return sum(scores) / len(scores)

        return scores[0] if scores else 0.5

    def _parse_score(self, text: str) -> float:
        text = text.strip()
        try:
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            data = json.loads(text.strip())
            score = float(data.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        match = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        logger.warning(f"Could not parse score from: {text[:100]}")
        return 0.5

    # ─────────────────────────────────────────
    # Telemetry + circuit breaker helpers
    # ─────────────────────────────────────────

    def _record_success(self, model: Model, response: "APIResponse") -> None:
        """Record a successful API call; reset circuit breaker counter."""
        self._consecutive_failures[model] = 0
        self._adaptive_router.record_success(model)
        self._adaptive_router.record_latency(model, response.latency_ms)
        self._telemetry.record_call(
            model,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            success=True,
        )
        # Feed rate-limit tracker so _apply_filters can enforce sliding-window caps
        self._planner.rate_limit_tracker.record(
            provider=get_provider(model),
            cost_usd=response.cost_usd,
            tokens=response.input_tokens + response.output_tokens,
        )

    def _record_failure(self, model: Model, error: Optional[Exception] = None) -> None:
        """
        Record a failed API call. Increment circuit breaker counter.
        If consecutive failures reach the threshold, mark the model unhealthy.
        401 Unauthorized errors immediately mark the model unhealthy (permanent auth failure).
        """
        # 401/404/400 = permanent failure — mark unhealthy immediately, no retries needed
        # 401 = bad API key, 404 = wrong model name, 400 = bad request (e.g. invalid param)
        error_str = str(error) if error else ""
        is_permanent_error = (
            "401" in error_str or "invalid_authentication" in error_str.lower()
            or "404" in error_str or "not found" in error_str.lower()
            or ("400" in error_str and "invalid_request_error" in error_str.lower())
        )
        if is_permanent_error:
            if self.api_health.get(model, True):
                self.api_health[model] = False
                if "401" in error_str:
                    reason = "auth error (401) — check your API key"
                elif "404" in error_str:
                    reason = "model not found (404) — check model name"
                else:
                    reason = f"invalid request (400) — {error_str[error_str.find('message'):error_str.find('message')+80]}"
                logger.warning(
                    f"Model {model.value} marked unhealthy immediately: {reason}."
                )
            # Task 6: auth/permanent errors permanently disable in adaptive router
            if "401" in error_str or "invalid_authentication" in error_str.lower():
                self._adaptive_router.record_auth_failure(model)
            self._telemetry.record_call(model, latency_ms=0.0, cost_usd=0.0, success=False)
            return

        self._consecutive_failures[model] = self._consecutive_failures.get(model, 0) + 1
        # Task 6: record timeout in adaptive router for degradation tracking
        _is_timeout = (
            "timeout" in error_str.lower() or "timed out" in error_str.lower()
            or "asyncio.timeouterror" in error_str.lower()
            or "TimeoutError" in (type(error).__name__ if error else "")
        )
        if _is_timeout:
            self._adaptive_router.record_timeout(model)
        self._telemetry.record_call(
            model, latency_ms=0.0, cost_usd=0.0, success=False
        )
        if self._consecutive_failures[model] >= self._CIRCUIT_BREAKER_THRESHOLD:
            if self.api_health.get(model, True):
                self.api_health[model] = False
                logger.warning(
                    f"Circuit breaker tripped for {model.value} "
                    f"after {self._consecutive_failures[model]} consecutive failures"
                )

    def _get_active_policies(self, task_id: str = "") -> list[Policy]:
        """Return merged global + node-level policies for the given task."""
        return self._active_policies.policies_for(task_id)

    # ─────────────────────────────────────────
    # Model selection & fallback
    # ─────────────────────────────────────────

    def _get_available_models(self, task_type: TaskType) -> list[Model]:
        candidates = ROUTING_TABLE.get(task_type, [])
        available = [m for m in candidates if self.api_health.get(m, False)]
        if not available:
            available = [m for m in Model if self.api_health.get(m, False)]
        # Task 6: also filter out models the adaptive router has degraded/disabled
        available = [m for m in available if self._adaptive_router.is_available(m)]
        return available

    def _get_cheapest_available(self) -> Model:
        from .models import COST_TABLE
        healthy = [m for m in Model if self.api_health.get(m, False)]
        if not healthy:
            raise RuntimeError("No healthy models available")
        return min(healthy, key=lambda m: COST_TABLE[m]["output"])

    def _select_reviewer(self, generator: Model, task_type: TaskType) -> Optional[Model]:
        gen_provider = get_provider(generator)
        candidates = self._get_available_models(task_type)

        for c in candidates:
            if get_provider(c) != gen_provider:
                return c

        for c in candidates:
            if c != generator:
                return c

        return None

    def _get_fallback(self, failed_model: Model) -> Optional[Model]:
        fb = FALLBACK_CHAIN.get(failed_model)
        if fb and self.api_health.get(fb, False):
            return fb
        for m in Model:
            if m != failed_model and self.api_health.get(m, False):
                return m
        return None

    # ─────────────────────────────────────────
    # DAG & dependency management
    # ─────────────────────────────────────────

    def _topological_sort(self, tasks: dict[str, Task]) -> list[str]:
        """
        Kahn's algorithm with cycle detection.
        FIX #6: Uses deque for O(1) popleft instead of list.sort()+pop(0) O(n²).
        """
        in_degree = {tid: 0 for tid in tasks}
        graph = defaultdict(list)

        for tid, task in tasks.items():
            for dep in task.dependencies:
                if dep in tasks:
                    graph[dep].append(tid)
                    in_degree[tid] += 1

        # Sort initial zero-degree nodes for determinism, then use deque
        queue = deque(sorted(tid for tid, deg in in_degree.items() if deg == 0))
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            # Sort neighbors for deterministic ordering before extending
            newly_ready = []
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    newly_ready.append(neighbor)
            newly_ready.sort()
            queue.extend(newly_ready)

        if len(result) != len(tasks):
            cycle_tasks = set(tasks.keys()) - set(result)
            logger.error(f"Dependency cycle detected involving: {cycle_tasks}")
            return result

        return result

    def _topological_levels(self, tasks: dict[str, Task]) -> list[list[str]]:
        """
        Group tasks into execution levels using Kahn's algorithm.

        Tasks at the same level have no dependencies on each other and can be
        executed in parallel. Level 0 = tasks with no dependencies, Level 1 =
        tasks whose only dependencies are in Level 0, and so on.

        Returns a list of levels, each level being a sorted list of task IDs.
        The union of all levels equals the full topological order.
        """
        in_degree = {tid: 0 for tid in tasks}
        graph: dict[str, list[str]] = defaultdict(list)

        for tid, task in tasks.items():
            for dep in task.dependencies:
                if dep in tasks:
                    graph[dep].append(tid)
                    in_degree[tid] += 1

        levels: list[list[str]] = []
        ready = sorted(tid for tid, deg in in_degree.items() if deg == 0)

        while ready:
            levels.append(ready)
            next_ready: list[str] = []
            for node in ready:
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_ready.append(neighbor)
            ready = sorted(next_ready)

        return levels

    def _gather_dependency_context(self, dep_ids: list[str]) -> str:
        """
        Gather output from completed/degraded dependencies.
        Truncates each dependency's output to self.context_truncation_limit chars
        and logs a warning when truncation occurs so information loss is visible.
        """
        parts = []
        limit = self.context_truncation_limit
        for dep_id in dep_ids:
            result = self.results.get(dep_id)
            if result and result.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED):
                text = result.output
                if len(text) > limit:
                    logger.warning(
                        f"Context truncated for {dep_id}: "
                        f"{len(text)} → {limit} chars. "
                        f"Increase orchestrator.context_truncation_limit to avoid information loss."
                    )
                    # Keep head + tail instead of hard-cutting the end.
                    # This preserves import block at the top AND the tail
                    # (often __main__ guards, class definitions, or conclusions).
                    # Guard: if limit is very small the marker alone may exceed it;
                    # in that case fall back to a simple head-only truncation.
                    head_size = int(limit * 0.6)
                    tail_size = max(0, limit - head_size - 80)  # 80 chars for marker
                    if tail_size > 0:
                        text = (
                            text[:head_size]
                            + f"\n\n... [TRUNCATED {len(text) - limit} chars] ...\n\n"
                            + text[-tail_size:]
                        )
                    else:
                        text = text[:limit]
                parts.append(f"[Output from {dep_id}]:\n{text}")
        return "\n\n".join(parts) if parts else ""

    # ─────────────────────────────────────────
    # Status & resume
    # ─────────────────────────────────────────

    def _determine_final_status(self, state: ProjectState) -> ProjectStatus:
        # Check budget / time first — these override empty-results SYSTEM_FAILURE
        # so that a run halted by budget exhaustion or timeout is correctly labelled.
        budget_exhausted = state.budget.remaining_usd <= 0
        time_ok = state.budget.time_remaining()

        if budget_exhausted:
            return ProjectStatus.BUDGET_EXHAUSTED
        if not time_ok:
            return ProjectStatus.TIMEOUT

        if not state.results:
            return ProjectStatus.SYSTEM_FAILURE

        # COMPLETED or DEGRADED both count as "passed" for final status
        all_passed = all(
            r.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED)
            for r in state.results.values()
        )

        degraded_heavy = any(
            r.degraded_fallback_count > r.iterations * 0.5
            for r in state.results.values()
            if r.iterations > 0
        )

        det_ok = all(r.deterministic_check_passed for r in state.results.values())

        if all_passed and det_ok and not degraded_heavy:
            return ProjectStatus.SUCCESS
        else:
            return ProjectStatus.PARTIAL_SUCCESS

    async def _resume_project(self, state: ProjectState) -> ProjectState:
        """
        Resume from last checkpoint.
        FIX #7: Restore persisted budget (spent_usd, phase_spent) instead of
        creating a fresh Budget. Only reset start_time for the new session.
        """
        # Restore budget state from checkpoint
        self.budget.spent_usd = state.budget.spent_usd
        self.budget.phase_spent = dict(state.budget.phase_spent)
        # Reset start_time so the new session gets fresh wall-clock tracking
        self.budget.start_time = time.time()

        logger.info(
            f"Restored budget: ${self.budget.spent_usd:.4f} already spent, "
            f"${self.budget.remaining_usd:.4f} remaining"
        )

        self.results = dict(state.results)
        remaining = [
            tid for tid in state.execution_order
            if tid not in self.results or
            self.results[tid].status in (TaskStatus.PENDING, TaskStatus.FAILED)
        ]
        if remaining:
            logger.info(f"Resuming: {len(remaining)} tasks remaining")
            for task_id in remaining:
                if task_id in state.tasks:
                    result = await self._execute_task(state.tasks[task_id])
                    self.results[task_id] = result
                    state.results[task_id] = result

        state.status = self._determine_final_status(state)
        return state

    def _make_state(self, project_desc: str, criteria: str,
                     tasks: dict[str, Task],
                     status: ProjectStatus = ProjectStatus.PARTIAL_SUCCESS,
                     execution_order: Optional[list[str]] = None,
                     ) -> ProjectState:
        return ProjectState(
            project_description=project_desc,
            success_criteria=criteria,
            budget=self.budget,
            tasks=tasks,
            results=dict(self.results),
            api_health={m.value: h for m, h in self.api_health.items()},
            status=status,
            execution_order=execution_order if execution_order is not None else list(tasks.keys()),
        )

    def _log_summary(self, state: ProjectState):
        logger.info("=" * 60)
        logger.info(f"PROJECT STATUS: {state.status.value}")
        logger.info(f"Budget: ${self.budget.spent_usd:.4f} / ${self.budget.max_usd}")
        logger.info(f"Time: {self.budget.elapsed_seconds:.1f}s / {self.budget.max_time_seconds}s")
        for tid, result in state.results.items():
            logger.info(
                f"  {tid}: score={result.score:.3f} status={result.status.value} "
                f"model={result.model_used.value} iters={result.iterations} "
                f"cost=${result.cost_usd:.4f}"
            )
        logger.info("=" * 60)
