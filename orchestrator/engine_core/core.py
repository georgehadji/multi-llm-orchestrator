"""
Orchestrator Core — Main Control Loop (Facade)
===============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Main orchestration facade that coordinates all engine components.
This is the refactored, decomposed version of the monolithic engine.py.

Part of Engine Decomposition (Phase 1)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from ..cache import DiskCache
    from ..models import Budget, ProjectState, ProjectStatus, Task, TaskResult
    from ..state import StateManager

logger = logging.getLogger(__name__)


class OrchestratorCore:
    """
    Main orchestration facade.

    Coordinates all engine components:
    - TaskExecutor: Individual task execution
    - CritiqueCycle: Generate→critique→revise pipeline
    - FallbackHandler: Model health & circuit breaker
    - BudgetEnforcer: Budget monitoring
    - DependencyResolver: DAG resolution

    Public API:
    - run_project(): Main entry point
    - run_project_streaming(): Streaming execution
    - run_job(): Policy-driven execution
    """

    def __init__(
        self,
        budget: Budget | None = None,
        cache: DiskCache | None = None,
        state_manager: StateManager | None = None,
        max_concurrency: int = 3,
        max_parallel_tasks: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize orchestrator core.

        Args:
            budget: Budget constraints
            cache: Disk cache for results
            state_manager: State persistence
            max_concurrency: Max concurrent API calls
            max_parallel_tasks: Max parallel tasks
            **kwargs: Additional configuration
        """
        from ..api_clients import UnifiedClient
        from ..cache import DiskCache
        from ..cost_optimization import OptimizationConfig, get_optimization_config
        from ..models import Budget, Model
        from ..semantic_cache import SemanticCache
        from ..state import StateManager

        # Initialize with defaults
        self.budget = budget or Budget()
        self.cache = cache or DiskCache()
        self.state_mgr = state_manager or StateManager()

        # API client
        self.client = UnifiedClient(
            cache=self.cache,
            max_concurrency=max_concurrency,
        )

        # Optimization config
        self.optim_config: OptimizationConfig = get_optimization_config()

        # Cache optimizer (optional)
        self._cache_optimizer = None
        try:
            from ..cache_optimizer import CacheConfig, CacheOptimizer
            self._cache_optimizer = CacheOptimizer(CacheConfig(
                l1_max_size=200,
                l1_ttl_seconds=3600,
                l2_ttl_hours=48,
                l3_quality_threshold=0.85,
                track_stats=True,
            ))
        except ImportError:
            pass

        # Semantic cache
        self._semantic_cache = SemanticCache(quality_threshold=0.85)

        # Initialize engine components
        self._init_engine_components(
            max_parallel_tasks=max_parallel_tasks,
            **kwargs
        )

        # State tracking
        self.results: dict[str, TaskResult] = {}
        self._results_lock: Any = None  # asyncio.Lock created at runtime
        self._project_id: str = ""

        # Model health
        self.api_health: dict[Model, bool] = dict.fromkeys(Model, True)

        logger.info(f"OrchestratorCore initialized: max_parallel={max_parallel_tasks}")

    def _init_engine_components(self, max_parallel_tasks: int, **kwargs: Any) -> None:
        """Initialize all engine core components."""
        from ..adaptive_router import AdaptiveRouter

        # Fallback handler (no dependencies)
        adaptive_router = AdaptiveRouter()
        self._fallback_handler = FallbackHandler(adaptive_router=adaptive_router)

        # Budget enforcer (depends on budget)
        budget_hierarchy = kwargs.get('budget_hierarchy')
        cost_predictor = kwargs.get('cost_predictor')
        self._budget_enforcer = BudgetEnforcer(
            budget=self.budget,
            budget_hierarchy=budget_hierarchy,
            cost_predictor=cost_predictor,
        )

        # Dependency resolver (no dependencies)
        self._dependency_resolver = DependencyResolver(
            context_truncation_limit=kwargs.get('context_truncation_limit', 40000)
        )

        # Critique cycle (depends on client)
        self._critique_cycle = CritiqueCycle(
            client=self.client,
            max_iterations=kwargs.get('max_iterations', 5),
            enable_streaming=kwargs.get('enable_streaming', False),
        )

        # Task executor (depends on all above)
        self._task_executor = TaskExecutor(
            client=self.client,
            cache=self.cache,
            semantic_cache=self._semantic_cache,
            cache_optimizer=self._cache_optimizer,
            optim_config=self.optim_config,
            critique_cycle=self._critique_cycle,
            fallback_handler=self._fallback_handler,
            budget_enforcer=self._budget_enforcer,
            dependency_resolver=self._dependency_resolver,
        )

        self._max_parallel_tasks = max_parallel_tasks

        logger.debug("Engine components initialized")

    async def run_project(
        self,
        project_description: str,
        success_criteria: str,
        output_dir: Path | None = None,
    ) -> ProjectState:
        """
        Run complete project execution.

        Args:
            project_description: Project specification
            success_criteria: Completion criteria
            output_dir: Optional output directory

        Returns:
            ProjectState with results
        """
        import asyncio

        from .models import ProjectStatus

        self._results_lock = asyncio.Lock()

        logger.info(f"Starting project: {project_description[:100]}...")

        # Phase 1: Decompose project into tasks
        tasks = await self._decompose_project(
            project_description, success_criteria
        )

        if not tasks:
            logger.error("Project decomposition failed")
            return self._make_state(
                project_description, success_criteria, {},
                ProjectStatus.SYSTEM_FAILURE
            )

        # Phase 2: Build dependency graph and execution order
        self._dependency_resolver.build_dependency_graph(tasks)
        execution_order = self._dependency_resolver.topological_sort(tasks)

        # Phase 3: Execute tasks
        await self._execute_all_tasks(tasks, execution_order)

        # Phase 4: Determine final status
        state = self._make_state(
            project_description,
            success_criteria,
            tasks,
            status=self._determine_final_status(),
            execution_order=execution_order,
        )

        # Phase 5: Save state checkpoint
        await self.state_mgr.save_checkpoint(state)

        # Phase 6: Log summary
        self._log_summary(state)

        return state

    async def _decompose_project(
        self,
        project_description: str,
        success_criteria: str,
    ) -> dict[str, Task]:
        """
        Decompose project specification into tasks.

        Args:
            project_description: Project spec
            success_criteria: Completion criteria

        Returns:
            Dictionary of task_id → Task
        """
        from .models import Task, TaskType

        logger.info("Decomposing project into tasks...")

        # Use LLM to decompose
        decomposition_prompt = (
            f"Decompose the following project into discrete tasks:\n\n"
            f"Project: {project_description}\n"
            f"Success Criteria: {success_criteria}\n\n"
            f"Return a JSON array of tasks with:\n"
            f"- id: unique task identifier\n"
            f"- type: one of {', '.join([t.value for t in TaskType])}\n"
            f"- prompt: detailed task instruction\n"
            f"- dependencies: list of task IDs this depends on\n"
        )

        try:
            response = await self.client.call_with_retry(
                model=self._fallback_handler.select_model(TaskType.DECOMPOSITION) or Model.GPT_4O_MINI,
                prompt=decomposition_prompt,
                max_tokens=4000,
                timeout=120,
            )

            # Parse JSON response
            import json
            tasks_data = json.loads(response.text)

            tasks = {}
            for task_data in tasks_data:
                task = Task(
                    id=task_data['id'],
                    type=TaskType(task_data['type']),
                    prompt=task_data['prompt'],
                    dependencies=task_data.get('dependencies', []),
                )
                tasks[task.id] = task

            logger.info(f"Decomposed into {len(tasks)} tasks")
            return tasks

        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            # Fallback: create simple single task
            return {
                "task_001": Task(
                    id="task_001",
                    type=TaskType.CODE_GEN,
                    prompt=f"Implement: {project_description}",
                    dependencies=[],
                )
            }

    async def _execute_all_tasks(
        self,
        tasks: dict[str, Task],
        execution_order: list[str],
    ) -> None:
        """
        Execute all tasks in dependency order.

        Args:
            tasks: All tasks
            execution_order: Ordered list of task IDs
        """
        for task_id in execution_order:
            task = tasks[task_id]

            # Check if dependencies are satisfied
            if not self._dependency_resolver.is_dependency_satisfied(task_id, self.results):
                logger.warning(f"Skipping {task_id}: dependencies not satisfied")
                continue

            # Check budget before task
            budget_ok, time_ok = self._budget_enforcer.check_budget(task)
            if not budget_ok or not time_ok:
                logger.warning(f"Budget/time exhausted before {task_id}")
                break

            # Execute task
            result = await self._task_executor.execute_task(task, tasks, self.results)

            # Store result
            async with self._results_lock:
                self.results[task_id] = result

            # Record cost
            self._budget_enforcer.record_cost(task_id, result.cost_usd)

            # Mark dependency complete
            self._dependency_resolver.mark_task_complete(task_id)

    def _determine_final_status(self) -> ProjectStatus:
        """Determine final project status from results."""
        from .models import ProjectStatus, TaskStatus

        if not self.results:
            return ProjectStatus.SYSTEM_FAILURE

        # Check budget and time
        budget_ok = self.budget.remaining_usd > 0
        time_ok = self.budget.time_remaining()

        if not budget_ok:
            return ProjectStatus.BUDGET_EXHAUSTED
        if not time_ok:
            return ProjectStatus.TIMEOUT

        # Check if all tasks completed
        all_completed = all(
            r.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED)
            for r in self.results.values()
        )

        if all_completed:
            # Check for degraded quality
            degraded_heavy = any(
                r.degraded_fallback_count > r.iterations * 0.5
                for r in self.results.values()
                if r.iterations > 0
            )

            det_ok = all(r.deterministic_check_passed for r in self.results.values())

            if det_ok and not degraded_heavy:
                return ProjectStatus.SUCCESS
            elif det_ok:
                return ProjectStatus.COMPLETED_DEGRADED
            else:
                return ProjectStatus.COMPLETED_DEGRADED

        return ProjectStatus.PARTIAL_SUCCESS

    def _make_state(
        self,
        project_desc: str,
        criteria: str,
        tasks: dict[str, Task],
        status: ProjectStatus,
        execution_order: list[str] | None = None,
    ) -> ProjectState:
        """Create ProjectState from current execution state."""
        from .models import ProjectState

        return ProjectState(
            project_description=project_desc,
            success_criteria=criteria,
            budget=self.budget,
            tasks=tasks,
            results=dict(self.results),
            api_health={m.value: h for m, h in self.api_health.items()},
            status=status,
            execution_order=execution_order or list(tasks.keys()),
        )

    def _log_summary(self, state: ProjectState):
        """Log project summary."""
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


# Import for backward compatibility
from .budget_enforcer import BudgetEnforcer
from .critique_cycle import CritiqueCycle
from .dependency_resolver import DependencyResolver
from .fallback_handler import FallbackHandler
from .task_executor import TaskExecutor

__all__ = ['OrchestratorCore']
