"""
ProjectRunner — Milestone 3 of ARCHITECTURE_COMPLIANCE_PLAN.md
===============================================================
Owns ``run_project`` and ``dry_run`` coordination logic extracted from
``engine.py``.

M3 change: the ``host=Orchestrator`` back-reference is replaced by two
injected objects:

  - ``callables: ProjectRunnerCallables`` — all execution callbacks
  - ``run_state: ProjectRunState``        — shared mutable run metadata

This makes ProjectRunner independently testable without an Orchestrator.

Design
------
- Named deps (state_mgr, budget, event_bus, etc.) are injected explicitly.
- Lifecycle: honours ``run_state.entered`` to decide whether to close
             connections in the finally clause (BUG-003 fix preserved).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

from ..models import ProjectState, ProjectStatus, TaskStatus, TaskType
from ..resilience import RetryTemplate
from .project_runner_deps import ProjectRunnerCallables, ProjectRunState
from .unattended_guard import RunContext, UnattendedGuard

logger = logging.getLogger("orchestrator")


def _slugify(text: str, max_words: int = 5, max_len: int = 50) -> str:
    """Convert a project description into a filesystem-friendly slug.

    Example: "Build a REST API with FastAPI" → "build-a-rest-api-with"
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    words = slug.split("-")
    trimmed = "-".join(words[:max_words])
    return trimmed[:max_len].rstrip("-")


def _detect_primary_language(description: str) -> str:
    """Heuristic to detect the primary output language from project description.

    Checks for keywords that suggest HTML/CSS/frontend vs Python/backend.
    Returns a language string suitable for ``target_language`` on Tasks.
    """
    desc_lower = description.lower()

    # Strong HTML/frontend signals
    html_keywords = {
        "landing page",
        "website",
        "web page",
        "static site",
        "html",
        "single-page",
        "landing",
        "frontend",
        "ui components",
    }
    for kw in html_keywords:
        if kw in desc_lower:
            return "html"

    # CSS-specific signals
    css_keywords = {
        "css",
        "stylesheet",
        "responsive design",
        "layout",
        "tailwind",
        "bootstrap",
        "scss",
        "sass",
    }
    for kw in css_keywords:
        if kw in desc_lower:
            return "css"

    # JavaScript signals
    js_keywords = {
        "javascript",
        "dynamic",
        "interactive",
        "react",
        "vue",
        "angular",
        "svelte",
        "next.js",
        "node.js",
    }
    for kw in js_keywords:
        if kw in desc_lower:
            return "javascript"

    # Default to python for backend/data projects
    python_keywords = {
        "api",
        "backend",
        "rest",
        "fastapi",
        "flask",
        "database",
        "sql",
        "migration",
        "endpoint",
        "microservice",
        "cli",
        "library",
        "package",
    }
    for kw in python_keywords:
        if kw in desc_lower:
            return "python"

    return ""


class ProjectRunner:
    """Coordinates ``run_project`` and ``dry_run`` on behalf of Orchestrator.

    All execution primitives are supplied via ``ProjectRunnerCallables``;
    this class owns the sequencing, event emission, and connection-lifecycle
    decisions.
    """

    def __init__(
        self,
        callables: ProjectRunnerCallables,
        run_state: ProjectRunState,
        state_mgr: Any,
        budget: Any,
        event_bus: Any,
        resumption_svc: Any,
        dashboard_bridge: Any,
        git_bridge: Any,
        generator: Any,
        meta_v2: Any,
        cache: Any,
        api_health: dict,  # type: ignore[type-arg]
        budget_hierarchy: Any = None,
    ) -> None:
        self._callables = callables
        self._run_state = run_state
        self._state_mgr = state_mgr
        self._budget = budget
        self._event_bus = event_bus
        self._resumption_svc = resumption_svc
        self._dashboard_bridge = dashboard_bridge
        self._git_bridge = git_bridge
        self._generator = generator
        self._meta_v2 = meta_v2
        self._cache = cache
        self._api_health = api_health
        self._budget_hierarchy = budget_hierarchy

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def run_project(
        self,
        project_description: str,
        success_criteria: str,
        project_id: str = "",
        app_profile: Any = None,
        analyze_on_complete: bool = False,
        output_dir: Path | None = None,
        precomposed_tasks: dict[str, Any] | None = None,
        constitution: Any = None,
    ) -> ProjectState:
        """Decompose project → execute tasks → return final ProjectState.

        When ``precomposed_tasks`` is provided (a ``dict[str, Task]``), the
        decomposition phase is **skipped** — used by the ``--from-speckit``
        path where tasks were already parsed from Spec-Kit artifacts.

        Preserves all behaviours from the original engine.py method:
        - Resume detection from PARTIAL_SUCCESS checkpoint
        - Assumption surfacing (Karpathy pattern)
        - Architecture rules generation
        - Dashboard / event-bus notifications
        - Git commit on completion
        - Meta-optimisation V2 callback
        - BUG-003: connection lifecycle (close only when not in context manager)
        """
        # Tracer is optional (tracing may be disabled via FeatureFlags)
        from ..tracing import get_tracer

        tracer = get_tracer()
        with tracer.start_as_current_span("run_project") as span:
            span.set_attribute("project.description", project_description[:200])
            if not project_id:
                # Generate a human-readable project_id: slug from description + short hash
                slug = _slugify(project_description)
                short_hash = hashlib.md5(
                    f"{project_description[:100]}{time.time()}".encode(),
                    usedforsecurity=False,
                ).hexdigest()[:6]
                project_id = f"{slug}-{short_hash}" if slug else short_hash

            # Publish project_id to run_state so downstream callbacks can read it
            self._run_state.project_id = project_id

            # Honour a per-run constitution (--from-speckit path): swap it into
            # the already-wired ConstitutionGate pipeline stage so protect_paths/
            # forbidden_imports/require_tests are enforced for precomposed tasks.
            constitution_gate = getattr(self._callables, "constitution_gate", None)
            if constitution is not None and constitution_gate is not None:
                constitution_gate.set_constitution(constitution)
            elif constitution is not None:
                logger.warning(
                    "constitution provided but no ConstitutionGate is wired — "
                    "protect_paths/forbidden_imports/require_tests will NOT be enforced"
                )

            # ENH-4: pre-flight unattended guard — fail closed before any work starts
            import sys

            _daily = None
            if self._budget_hierarchy is not None:
                _daily = getattr(self._budget_hierarchy, "_org_max", None)
            _hitl = getattr(self, "_hitl", None)
            _has_checkpoint = _hitl is not None and _hitl.has_real_channel()
            UnattendedGuard.validate(
                RunContext(
                    budget=self._budget,
                    daily_cap_usd=_daily,
                    max_retries=getattr(self._run_state, "max_retries", None),
                    has_checkpoint=_has_checkpoint,
                    is_unattended=not sys.stdin.isatty(),
                )
            )

            logger.info("Starting project %s", project_id)
            logger.info(
                "Budget: $%s, %ss",
                self._budget.max_usd,
                self._budget.max_time_seconds,
            )

            try:
                # ── Resume detection ────────────────────────────────────
                existing = await self._state_mgr.load_project(project_id)
                if existing and existing.status == ProjectStatus.PARTIAL_SUCCESS:
                    logger.info("Resuming project %s from checkpoint", project_id)
                    state = await self._resumption_svc.resume(existing)
                    await self._state_mgr.save_project(project_id, state)
                    self._callables.log_summary(state)
                    return state  # type: ignore[no-any-return]

                # ── Phase 0: Architecture rules ──────────────────────────
                architecture_rules = await self._callables.generate_architecture_rules(
                    project_description, success_criteria, output_dir
                )
                # Publish to run_state so Orchestrator can still read it
                self._run_state.architecture_rules = architecture_rules

                # ── Phase 1: Decompose ───────────────────────────────────
                if precomposed_tasks is not None:
                    # Spec-Kit (or other) pre-composed path — skip LLM decomposition
                    tasks = precomposed_tasks
                    logger.info(
                        "Using %d pre-composed tasks (decomposition skipped)",
                        len(tasks),
                    )
                else:
                    # Surface hidden assumptions (Karpathy pattern)
                    try:
                        from ..assumption_gate import surface_assumptions

                        report = await surface_assumptions(
                            project_description, self._callables.client
                        )
                        if report.has_ambiguity:
                            logger.info(
                                "Assumptions surfaced: %d assumptions, %d questions",
                                len(report.assumptions),
                                len(report.clarification_questions),
                            )
                            project_description = (
                                f"{project_description}\n\n{report.to_prompt_context()}"
                            )
                    except ImportError:
                        pass

                    # Build project context from heuristic language detection
                    try:
                        from ..project_mgmt.context import ProjectContext as _PContext

                        _primary_lang = _detect_primary_language(project_description)
                        _project_context = _PContext(
                            project_type=(
                                "frontend"
                                if _primary_lang in ("html", "css", "javascript")
                                else "backend"
                            ),
                            tech_stack=[_primary_lang] if _primary_lang else [],
                        )
                    except ImportError:
                        _project_context = None

                    gen_result = await self._generator.decompose(
                        project_description,
                        success_criteria,
                        app_profile=app_profile,
                        policy=RetryTemplate.DECOMPOSE.to_policy(),
                        project_context=_project_context,
                    )
                    if not gen_result.succeeded:
                        logger.error("Decomposition failed: %s", gen_result.error)
                        return self._callables.make_state(  # type: ignore[no-any-return]
                            project_description,
                            success_criteria,
                            {},
                            ProjectStatus.SYSTEM_FAILURE,
                        )
                    tasks = gen_result.tasks
                    if not tasks:
                        return self._callables.make_state(  # type: ignore[no-any-return]
                            project_description,
                            success_criteria,
                            {},
                            ProjectStatus.SYSTEM_FAILURE,
                        )

                    # ── Inject target_language into tasks that lack it ─────
                    _primary_lang = _detect_primary_language(project_description)
                    if _primary_lang and _primary_lang not in ("", "python"):
                        injected = 0
                        for task in tasks.values():
                            if (
                                not getattr(task, "target_language", "")
                                and task.type == TaskType.CODE_GEN
                            ):
                                task.target_language = _primary_lang
                                task.prompt = (
                                    f"[LANGUAGE: {_primary_lang}] {task.prompt}\n\n"
                                    f"IMPORTANT: Output MUST be valid {_primary_lang} code. "
                                    f"Do NOT generate Python. Use the named-block format "
                                    f"for separate files (**filename.ext** + code block)."
                                )
                                injected += 1
                        if injected:
                            logger.info(
                                "Injected target_language=%s into %d tasks",
                                _primary_lang,
                                injected,
                            )

                # Topological sort
                execution_order = self._callables.topological_sort(tasks)
                logger.info("Execution order: %s", execution_order)

                # Initial state snapshot for dashboard
                initial_state = self._callables.make_state(
                    project_description,
                    success_criteria,
                    tasks,
                    execution_order=execution_order,
                )

                # Dashboard notification
                logger.debug("Notifying dashboard of project start...")
                self._dashboard_bridge.on_project_start(
                    project_id, initial_state, architecture_rules
                )
                logger.debug("Dashboard notification complete")

                # ProjectStarted event
                if self._event_bus:
                    from ..unified_events.core import ProjectStartedEvent

                    logger.debug("Publishing ProjectStarted event...")
                    await self._event_bus.publish(
                        ProjectStartedEvent(
                            aggregate_id=project_id,
                            project_id=project_id,
                            description=project_description[:200],
                            budget=self._budget.max_usd,
                        )
                    )
                    logger.debug("ProjectStarted event published")

                # ── Phases 2-5: Execute ──────────────────────────────────
                logger.info("Starting task execution...")
                state = await self._callables.execute_all(
                    tasks, execution_order, project_description, success_criteria
                )
                logger.info("Task execution completed.")

                # Final status and persistence
                state.execution_order = execution_order
                state.status = self._callables.determine_final_status(state)
                await self._state_mgr.save_project(project_id, state)
                self._callables.log_summary(state)

                # Meta-optimisation V2
                if self._meta_v2:
                    from ..meta_integration import on_project_completed

                    await on_project_completed(self._meta_v2, state, run_optimization=True)

                # Git commit (P3-5 GitBridge, null-safe)
                self._git_bridge.commit_project(
                    project_name=project_description[:50],
                    total_tasks=len(tasks),
                    total_cost=self._budget.spent_usd,
                    elapsed_time=self._budget.elapsed_seconds,
                )

                # ProjectCompleted event
                if self._event_bus:
                    from ..unified_events.core import ProjectCompletedEvent

                    results = self._run_state.results
                    completed_count = sum(
                        1 for r in results.values() if r.status != TaskStatus.FAILED
                    )
                    failed_count = sum(1 for r in results.values() if r.status == TaskStatus.FAILED)
                    await self._event_bus.publish(
                        ProjectCompletedEvent(
                            aggregate_id=project_id,
                            project_id=project_id,
                            status=state.status.value,
                            total_cost=self._budget.spent_usd,
                            tasks_completed=completed_count,
                            tasks_failed=failed_count,
                        )
                    )

                # Optional post-project analysis
                if analyze_on_complete and output_dir:
                    await self._callables.analyze_completed_project(state, output_dir)

                return state  # type: ignore[no-any-return]

            finally:
                # BUG-003 FIX: close connections only when NOT inside an async
                # context manager.  When entered=True, __aexit__ owns cleanup
                # and runs after run_job()'s post-run operations.
                if not self._run_state.entered:
                    await self._state_mgr.close()
                    await self._cache.close()

    async def run_job(self, spec: Any) -> Any:
        """
        Policy-driven entry point. Accepts a JobSpec that bundles project
        description, success criteria, budget, quality targets, and policies.

        Orchestrator-level state mutation (budget assignment, policy set,
        quality mode, max parallel tasks) is expected to happen BEFORE this
        call. This method handles the lifecycle: warm-start → preflight
        BudgetEnforcer → run_project → charge → flush telemetry.

        Args:
            spec: A JobSpec-like object with project_description, success_criteria,
                  budget, job_id, team, max_parallel_tasks, quality_mode,
                  and policy_set attributes.
        """
        # Warm-start: blend historical profiles before execution
        if self._callables.warm_start_fn is not None:
            await self._callables.warm_start_fn()

        # Extract once; both the pre-flight check and charge path need them.
        job_id = getattr(spec, "job_id", "") or ""
        team = getattr(spec, "team", "") or ""

        # BudgetHierarchy pre-flight check via BudgetEnforcer
        from .budget_enforcer import BudgetEnforcer

        BudgetEnforcer.enforce_hierarchy_job(
            self._budget_hierarchy, job_id, team, spec.budget.max_usd
        )
        try:
            state = await self.run_project(
                project_description=spec.project_description,
                success_criteria=spec.success_criteria,
            )
        except Exception:
            # BUG-001 FIX: release reservation on failure
            if self._budget_hierarchy is not None:
                self._budget_hierarchy.release_reservation(job_id, team)
            raise
        # Charge actual spend to BudgetHierarchy
        if self._budget_hierarchy is not None:
            # T1-C5: use spent_usd directly. Deriving this from
            # `max_usd - remaining_usd` silently clamps at max_usd whenever the
            # run overspends its own per-run cap, because remaining_usd floors
            # at 0.0 — the excess would never reach the cross-run hierarchy.
            actual_spend = self._budget.spent_usd
            BudgetEnforcer.enforce_hierarchy_job(
                self._budget_hierarchy, job_id, team, spec.budget.max_usd, actual_spend
            )
        # Persist telemetry snapshots for all models used this run (fire-and-forget)
        if self._callables.flush_telemetry_fn is not None:
            await self._callables.flush_telemetry_fn(job_id)

        return state

    async def dry_run(
        self,
        project_description: str,
        success_criteria: str,
    ) -> Any:
        """Decompose and build an execution plan WITHOUT running any tasks.

        Makes one real API call (decomposition) then stops.
        Returns an ``ExecutionPlan`` that can be printed with ``plan.render()``.
        """
        from ..operations.dry_run import (
            _DEFAULT_TOKENS,
            _TOKEN_ESTIMATES,
            ExecutionPlan,
            TaskPlan,
        )
        from ..models import ROUTING_TABLE, estimate_cost

        # Surface assumptions (Karpathy pattern)
        try:
            from ..assumption_gate import surface_assumptions

            report = await surface_assumptions(project_description, self._callables.client)
            if report.has_ambiguity:
                logger.info(
                    "Assumptions surfaced: %d assumptions, %d questions",
                    len(report.assumptions),
                    len(report.clarification_questions),
                )
                project_description = f"{project_description}\n\n{report.to_prompt_context()}"
        except ImportError:
            pass

        gen_result = await self._generator.decompose(
            project_description,
            success_criteria,
            policy=RetryTemplate.DECOMPOSE.to_policy(),
        )
        tasks = gen_result.tasks if gen_result.succeeded else {}

        # Validate budget sufficiency
        if tasks and self._budget:
            is_sufficient, warning = self._budget.validate_sufficient_for_tasks(len(tasks))
            if not is_sufficient:
                logger.warning(warning)

        if not tasks:
            return ExecutionPlan(
                project_description=project_description,
                success_criteria=success_criteria,
            )

        levels = self._callables.topological_levels(tasks)
        level_index: dict[str, int] = {
            tid: lvl_idx for lvl_idx, lvl_tasks in enumerate(levels) for tid in lvl_tasks
        }

        task_plans: list[Any] = []
        total_cost = 0.0

        for tid, task in tasks.items():
            model_list = ROUTING_TABLE.get(task.type, [])
            available = [m for m in model_list if self._api_health.get(m, True)]
            primary = available[0] if available else (model_list[0] if model_list else None)

            in_tokens, out_tokens = _TOKEN_ESTIMATES.get(task.type.value, _DEFAULT_TOKENS)
            cost = estimate_cost(primary, in_tokens, out_tokens) if primary else 0.0
            total_cost += cost

            task_plans.append(
                TaskPlan(
                    task_id=tid,
                    task_type=task.type.value,
                    prompt_preview=(
                        task.prompt[:80].replace("\n", " ") + "…"
                        if len(task.prompt) > 80
                        else task.prompt
                    ),
                    dependencies=[str(d) for d in task.dependencies],
                    parallel_level=level_index.get(tid, 0),
                    primary_model=primary.value if primary else "unknown",
                    estimated_cost_usd=round(cost, 6),
                    acceptance_threshold=task.acceptance_threshold,
                    max_iterations=task.max_iterations,
                )
            )

        task_plans.sort(key=lambda t: (t.parallel_level, t.task_id))

        return ExecutionPlan(
            project_description=project_description,
            success_criteria=success_criteria,
            tasks=task_plans,
            parallel_levels=levels,
            estimated_total_cost=round(total_cost, 6),
            num_parallel_levels=len(levels),
        )
