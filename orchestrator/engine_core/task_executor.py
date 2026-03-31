"""
Task Executor — Task Execution Logic
=====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles individual task execution, coordinating generation, validation,
and result aggregation.

Part of Engine Decomposition (Phase 1) - Extracted from engine.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..models import Model, TaskResult, TaskType

if TYPE_CHECKING:
    from ..api_clients import UnifiedClient
    from ..budget_enforcer import BudgetEnforcer
    from ..cache import DiskCache
    from ..cost_optimization import CacheOptimizer, OptimizationConfig
    from ..semantic_cache import SemanticCache
    from .critique_cycle import CritiqueCycle, CritiqueState
    from .dependency_resolver import DependencyResolver
    from .fallback_handler import FallbackHandler
    from .models import Task

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for task execution."""

    task: Task
    primary_model: Model
    reviewer_model: Model | None
    dependency_context: str = ""
    cached_result: dict | None = None
    full_prompt: str = ""


class TaskExecutor:
    """
    Executes individual tasks with validation and error handling.

    Responsibilities:
    1. Gather dependency context
    2. Check cache for reusable results
    3. Build execution prompt
    4. Run critique cycle
    5. Validate output
    6. Build task result

    Execution Flow:
    1. Check dependencies → 2. Check cache → 3. Execute → 4. Validate → 5. Return
    """

    def __init__(
        self,
        client: UnifiedClient,
        cache: DiskCache,
        semantic_cache: SemanticCache,
        cache_optimizer: CacheOptimizer | None,
        optim_config: OptimizationConfig,
        critique_cycle: CritiqueCycle,
        fallback_handler: FallbackHandler,
        budget_enforcer: BudgetEnforcer,
        dependency_resolver: DependencyResolver,
    ):
        """
        Initialize task executor.

        Args:
            client: API client for LLM calls
            cache: Disk cache for results
            semantic_cache: Semantic similarity cache
            cache_optimizer: Multi-level cache optimizer
            optim_config: Optimization configuration
            critique_cycle: Critique cycle runner
            fallback_handler: Model health & fallback
            budget_enforcer: Budget enforcement
            dependency_resolver: Dependency management
        """
        self.client = client
        self.cache = cache
        self.semantic_cache = semantic_cache
        self.cache_optimizer = cache_optimizer
        self.optim_config = optim_config
        self.critique_cycle = critique_cycle
        self.fallback_handler = fallback_handler
        self.budget_enforcer = budget_enforcer
        self.dependency_resolver = dependency_resolver

        # Lazy-loaded components
        self._tdd_generator = None
        self._has_tdd = False

        # Try to import TDD generator
        try:
            from .test_first_generator import TestFirstGenerator

            self._has_tdd = True
        except ImportError:
            pass

    async def execute_task(
        self,
        task: Task,
        all_tasks: dict[str, Task],
        results: dict[str, TaskResult],
    ) -> TaskResult:
        """
        Execute a single task.

        Args:
            task: Task to execute
            all_tasks: All tasks in project (for dependency resolution)
            results: Results from completed tasks

        Returns:
            TaskResult with execution outcome
        """
        logger.info(f"Executing task {task.id} ({task.type.value})")

        # Build execution context
        context = await self._build_execution_context(task, all_tasks, results)

        # Check if cache hit (no dependencies)
        if context.cached_result and not context.dependency_context:
            return self._build_cached_result(task, context)

        # Build full prompt with context
        context.full_prompt = self._build_full_prompt(task, context)

        # Get models for execution
        models = self.fallback_handler.get_available_models(task.type)
        if not models:
            return self._build_failure_result(task, "No models available for task type")

        context.primary_model = models[0]
        context.reviewer_model = self.fallback_handler.select_reviewer(
            context.primary_model, task.type
        )

        # Try TDD-first generation if enabled
        if self._has_tdd and self.optim_config.enable_tdd_first:
            tdd_result = await self._try_tdd_generation(task, context)
            if tdd_result:
                return tdd_result

        # Run critique cycle
        cycle_state = await self.critique_cycle.run_cycle(
            task=task,
            primary_model=context.primary_model,
            reviewer_model=context.reviewer_model,
            full_prompt=context.full_prompt,
        )

        # Build result from cycle state
        return self._build_result_from_cycle(task, cycle_state, context)

    async def _build_execution_context(
        self,
        task: Task,
        all_tasks: dict[str, Task],
        results: dict[str, TaskResult],
    ) -> ExecutionContext:
        """
        Build execution context for task.

        Args:
            task: Task to execute
            all_tasks: All tasks in project
            results: Completed task results

        Returns:
            ExecutionContext with gathered information
        """
        # Gather dependency context
        dependency_context = self.dependency_resolver.get_dependency_context(
            task, results, all_tasks
        )

        # Check cache (only if no dependencies)
        cached_result = None
        if not dependency_context and self.cache_optimizer:
            cached_result = await self.cache_optimizer.get(
                model=None,  # Will use default from routing
                prompt=task.prompt,
                max_tokens=task.max_output_tokens,
                task_type=task.type,
            )

        # Fallback to semantic cache
        if cached_result is None and not dependency_context:
            cached_output = self.semantic_cache.get_cached_pattern(task)
            if cached_output:
                cached_result = {
                    "response": cached_output,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "cost": 0.0,
                    "cached": True,
                }

        return ExecutionContext(
            task=task,
            primary_model=Model.GPT_4O_MINI,  # Placeholder
            reviewer_model=None,
            dependency_context=dependency_context,
            cached_result=cached_result,
        )

    def _build_full_prompt(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> str:
        """
        Build full prompt with context.

        Args:
            task: Task definition
            context: Execution context

        Returns:
            Complete prompt for LLM
        """
        full_prompt = task.prompt

        if context.dependency_context:
            # For code_review tasks, make context explicit
            if task.type == TaskType.CODE_REVIEW:
                full_prompt += (
                    f"\n\n--- SOURCE CODE TO REVIEW (from prior tasks) ---\n"
                    f"The following is the actual generated source code you must "
                    f"review. Do NOT claim the code was not provided.\n\n"
                    f"{context.dependency_context}"
                )
            else:
                full_prompt += (
                    f"\n\n--- CONTEXT FROM PRIOR TASKS ---\n" f"{context.dependency_context}"
                )

        return full_prompt

    async def _try_tdd_generation(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> TaskResult | None:
        """
        Try TDD-first generation.

        Args:
            task: Task to execute
            context: Execution context

        Returns:
            TaskResult if TDD succeeded, None otherwise
        """
        if not self._has_tdd or not self.optim_config.enable_tdd_first:
            return None

        if task.type != TaskType.CODE_GEN:
            return None

        logger.info(f"  {task.id}: Using TDD-first generation")

        try:
            # Lazy initialize TDD generator
            if self._tdd_generator is None:
                from .test_first_generator import TestFirstGenerator

                self._tdd_generator = TestFirstGenerator(
                    client=self.client,
                    sandbox=None,  # Optional sandbox
                    max_test_iterations=3,
                )

            tdd_result = await self._tdd_generator.generate_with_tests(
                task=task,
                project_context=context.dependency_context or "",
                model=context.primary_model,
            )

            if tdd_result.success:
                logger.info(
                    f"  {task.id}: TDD success - "
                    f"{tdd_result.test_result.tests_passed}/"
                    f"{tdd_result.test_result.tests_run} tests passed"
                )

                from .models import TaskStatus

                return TaskResult(
                    task_id=task.id,
                    output=tdd_result.implementation_code,
                    score=1.0 if tdd_result.test_result.passed else 0.8,
                    model_used=context.primary_model,
                    reviewer_model=None,
                    tokens_used={
                        "input": 0,
                        "output": len(tdd_result.implementation_code.split()),
                    },
                    iterations=tdd_result.iterations,
                    cost_usd=0.0,  # Would need to track from TDD
                    status=TaskStatus.COMPLETED if tdd_result.success else TaskStatus.DEGRADED,
                    critique=f"Tests: {tdd_result.test_result.tests_passed}/"
                    f"{tdd_result.test_result.tests_run} passed",
                    deterministic_check_passed=tdd_result.test_result.passed,
                    degraded_fallback_count=0,
                    attempt_history=[],
                    test_files={"test_main.py": tdd_result.test_spec.test_code},
                    tests_passed=tdd_result.test_result.tests_passed,
                    tests_total=tdd_result.test_result.tests_run,
                )

        except Exception as e:
            logger.warning(f"  {task.id}: TDD generation failed: {e}, falling back to standard")

        return None

    def _build_cached_result(
        self,
        task: Task,
        context: ExecutionContext,
    ) -> TaskResult:
        """
        Build result from cache hit.

        Args:
            task: Task definition
            context: Execution context with cached result

        Returns:
            TaskResult from cache
        """
        from .models import TaskResult, TaskStatus

        cached = context.cached_result

        return TaskResult(
            task_id=task.id,
            output=cached["response"],
            score=0.85,  # Cached patterns meet quality threshold
            model_used=context.primary_model,
            reviewer_model=None,
            tokens_used={
                "input": cached.get("tokens_input", 0),
                "output": cached.get("tokens_output", 0),
            },
            iterations=0,
            cost_usd=cached.get("cost", 0.0),
            status=TaskStatus.COMPLETED,
            critique="",
            deterministic_check_passed=True,
            degraded_fallback_count=0,
            attempt_history=[],
        )

    def _build_result_from_cycle(
        self,
        task: Task,
        state: CritiqueState,
        context: ExecutionContext,
    ) -> TaskResult:
        """
        Build TaskResult from critique cycle state.

        Args:
            task: Task definition
            state: Critique cycle state
            context: Execution context

        Returns:
            TaskResult with execution results
        """
        from .models import TaskResult, TaskStatus

        # Determine status based on score
        if state.best_score >= 0.9:
            status = TaskStatus.COMPLETED
        elif state.best_score >= 0.7:
            status = TaskStatus.DEGRADED
        else:
            status = TaskStatus.FAILED

        return TaskResult(
            task_id=task.id,
            output=state.best_output,
            score=state.best_score,
            model_used=context.primary_model,
            reviewer_model=context.reviewer_model,
            tokens_used={
                "input": state.total_input_tokens,
                "output": state.total_output_tokens,
            },
            iterations=len(state.scores_history),
            cost_usd=state.total_cost,
            status=status,
            critique=state.best_critique,
            deterministic_check_passed=len(state.failed_validators) == 0,
            degraded_fallback_count=state.degraded_count,
            attempt_history=state.attempt_history,
        )

    def _build_failure_result(
        self,
        task: Task,
        reason: str,
    ) -> TaskResult:
        """
        Build failure result.

        Args:
            task: Task definition
            reason: Failure reason

        Returns:
            TaskResult with failure status
        """
        from .models import Model, TaskResult, TaskStatus

        return TaskResult(
            task_id=task.id,
            output="",
            score=0.0,
            model_used=Model.GPT_4O_MINI,
            reviewer_model=None,
            tokens_used={"input": 0, "output": 0},
            iterations=0,
            cost_usd=0.0,
            status=TaskStatus.FAILED,
            critique=reason,
            deterministic_check_passed=False,
            degraded_fallback_count=0,
            attempt_history=[],
        )
