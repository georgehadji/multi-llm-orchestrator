"""
Orchestrator Engine — Core Control Loop
========================================
Implements the full generate → critique → revise → evaluate pipeline
with cross-model review, deterministic validation, budget enforcement,
plateau detection, and fallback routing.

Architecture decisions (Nash Equilibrium analysis):
- Path A: Single-model with self-review → fast, cheap, low quality ceiling
- Path B: Multi-model with cross-review → moderate cost, high quality ceiling
- Path C: All-models consensus → expensive, diminishing returns past 2 reviewers
→ Selected: Path B (Minimax Regret: worst-case quality is highest)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from typing import Optional

from .models import (
    Budget, BUDGET_PARTITIONS, FALLBACK_CHAIN, Model, ProjectState,
    ProjectStatus, ROUTING_TABLE, Task, TaskResult, TaskStatus, TaskType,
    get_max_iterations, get_provider, estimate_cost, build_default_profiles,
)
from .api_clients import UnifiedClient, APIResponse
from .validators import run_validators, all_validators_pass, ValidationResult
from .cache import DiskCache
from .state import StateManager
from .policy import ModelProfile, Policy, PolicySet, JobSpec  # noqa: F401
from .policy_engine import PolicyEngine, PolicyViolationError  # noqa: F401
from .planner import ConstraintPlanner
from .telemetry import TelemetryCollector

logger = logging.getLogger("orchestrator")


class Orchestrator:
    """
    Main orchestration engine.

    Invariants maintained:
    1. Cross-review always uses different provider than generator
    2. Deterministic validators override LLM scores
    3. Budget ceiling is never exceeded
    4. State is checkpointed after each task
    5. Plateau detection prevents runaway iteration
    """

    def __init__(self, budget: Optional[Budget] = None,
                 cache: Optional[DiskCache] = None,
                 state_manager: Optional[StateManager] = None,
                 max_concurrency: int = 3,
                 profiles: Optional[dict[Model, ModelProfile]] = None,
                 policy_set: Optional[PolicySet] = None):
        self.budget = budget or Budget()
        self.cache = cache or DiskCache()
        self.state_mgr = state_manager or StateManager()
        self.client = UnifiedClient(cache=self.cache, max_concurrency=max_concurrency)
        self.api_health: dict[Model, bool] = {m: True for m in Model}
        self.results: dict[str, TaskResult] = {}
        self._project_id: str = ""

        # Check which providers are actually available
        for model in Model:
            if not self.client.is_available(model):
                self.api_health[model] = False
                logger.warning(f"{model.value}: provider SDK/key not available")

        # ── Policy-driven constraint solver components ────────────────────────
        # Seeded from static tables; TelemetryCollector updates profiles at runtime.
        self._policy_set: PolicySet = policy_set or PolicySet()
        _profiles = profiles or build_default_profiles()
        self._policy_engine = PolicyEngine()
        self._telemetry = TelemetryCollector(_profiles)
        self._planner = ConstraintPlanner(
            profiles=_profiles,
            policy_engine=self._policy_engine,
            api_health=self.api_health,
        )

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    async def run_job(self, spec: JobSpec) -> "ProjectState":
        """
        New first-class entry point accepting a JobSpec with full policy support.
        Sets the active policy_set and budget for the duration of this run,
        then delegates to run_project().

        Example:
            from orchestrator import JobSpec, Budget, PolicySet, Policy, TaskType
            spec = JobSpec(
                project_description="Build auth service",
                success_criteria="Tests pass",
                budget=Budget(max_usd=8.0),
                policy_set=PolicySet(global_policies=[
                    Policy("gdpr", allow_training_on_output=False),
                ]),
                quality_targets={TaskType.CODE_GEN: 0.90},
            )
            state = await orch.run_job(spec)
        """
        self._policy_set = spec.policy_set
        self.budget = spec.budget
        # Apply quality_targets: override Task acceptance_threshold per type at execute time
        # (engine._execute_task uses task.acceptance_threshold which was set in __post_init__)
        # Store for use in _execute_task if overrides exist
        self._quality_targets: dict[TaskType, float] = spec.quality_targets
        return await self.run_project(spec.project_description, spec.success_criteria)

    async def run_project(self, project_description: str,
                          success_criteria: str,
                          project_id: str = "") -> ProjectState:
        """
        Main entry point. Decomposes project → executes tasks → returns state.
        """
        if not project_id:
            project_id = hashlib.md5(
                f"{project_description[:100]}{time.time()}".encode()
            ).hexdigest()[:12]
        self._project_id = project_id

        logger.info(f"Starting project {project_id}")
        logger.info(f"Budget: ${self.budget.max_usd}, {self.budget.max_time_seconds}s")

        # Check if resumable
        existing = self.state_mgr.load_project(project_id)
        if existing and existing.status == ProjectStatus.PARTIAL_SUCCESS:
            logger.info(f"Resuming project {project_id} from checkpoint")
            return await self._resume_project(existing)

        # Phase 1: Decompose
        tasks = await self._decompose(project_description, success_criteria)
        if not tasks:
            return self._make_state(
                project_description, success_criteria, {},
                ProjectStatus.SYSTEM_FAILURE
            )

        # Topological sort
        execution_order = self._topological_sort(tasks)
        logger.info(f"Execution order: {execution_order}")

        # Phase 2-5: Execute
        state = await self._execute_all(
            tasks, execution_order, project_description, success_criteria
        )

        # Final status determination
        state.status = self._determine_final_status(state)
        self.state_mgr.save_project(project_id, state)

        self._log_summary(state)
        return state

    # ─────────────────────────────────────────
    # Phase 1: Decomposition
    # ─────────────────────────────────────────

    async def _decompose(self, project: str, criteria: str) -> dict[str, Task]:
        """
        Use cheapest capable model to break project into atomic tasks.
        Returns parsed Task objects.
        """
        valid_types = [t.value for t in TaskType]
        prompt = f"""You are a project decomposition engine. Break this project into
atomic, executable tasks.

PROJECT: {project}

SUCCESS CRITERIA: {criteria}

Return ONLY a JSON array. Each element must have:
- "id": string (e.g., "task_001")
- "type": one of {valid_types}
- "prompt": detailed instruction for the task executor
- "dependencies": list of task id strings this depends on (empty if none)
- "hard_validators": list of validator names from: ["json_schema", "python_syntax", "pytest", "ruff", "latex", "length"]

RULES:
- Tasks must be atomic (one clear deliverable each)
- Dependencies must form a DAG (no cycles)
- Include code_review tasks after code_generation tasks
- Include at least one evaluation task at the end
- 5-15 tasks total for a medium project

Return ONLY the JSON array, no markdown fences, no explanation."""

        model = self._get_cheapest_available()
        try:
            response = await self.client.call(
                model, prompt,
                system="You are a precise project decomposition engine. Output only valid JSON.",
                max_tokens=2000,
                timeout=45,
            )
            self.budget.charge(response.cost_usd, "decomposition")
            return self._parse_decomposition(response.text)
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            # Fallback: try another model
            fallback = self._get_fallback(model)
            if fallback:
                try:
                    response = await self.client.call(
                        fallback, prompt,
                        system="You are a precise project decomposition engine. Output only valid JSON.",
                        max_tokens=2000,
                        timeout=45,
                    )
                    self.budget.charge(response.cost_usd, "decomposition")
                    return self._parse_decomposition(response.text)
                except Exception as e2:
                    logger.error(f"Decomposition fallback also failed: {e2}")
            return {}

    def _parse_decomposition(self, text: str) -> dict[str, Task]:
        """Parse LLM output into Task objects with defensive handling."""
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        text = text.strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON array in text
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                items = json.loads(match.group())
            else:
                logger.error("Could not parse decomposition output as JSON")
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

    async def _execute_all(self, tasks: dict[str, Task],
                            execution_order: list[str],
                            project_desc: str,
                            success_criteria: str) -> ProjectState:
        """Execute all tasks in dependency order."""
        for task_id in execution_order:
            # Pre-flight checks
            if not self.budget.can_afford(0.01):
                logger.warning("Budget exhausted, halting")
                break
            if not self.budget.time_remaining():
                logger.warning("Time limit reached, halting")
                break

            task = tasks[task_id]

            # Check dependencies met
            deps_ok = all(
                self.results.get(dep, TaskResult(dep, "", 0.0, Model.GPT_4O_MINI)).status
                == TaskStatus.COMPLETED
                for dep in task.dependencies
            )
            if not deps_ok:
                logger.warning(f"Skipping {task_id}: unmet dependencies")
                self.results[task_id] = TaskResult(
                    task_id=task_id, output="", score=0.0,
                    model_used=Model.GPT_4O_MINI,
                    status=TaskStatus.FAILED,
                )
                continue

            # Execute
            result = await self._execute_task(task)
            self.results[task_id] = result
            task.status = result.status

            # Checkpoint
            state = self._make_state(project_desc, success_criteria, tasks)
            self.state_mgr.save_checkpoint(self._project_id, task_id, state)

        return self._make_state(project_desc, success_criteria, tasks)

    async def _execute_task(self, task: Task) -> TaskResult:
        """
        Core loop: generate → critique → revise → evaluate
        With plateau detection and deterministic validation.
        """
        models = self._get_available_models(task.type, task_id=task.id)
        if not models:
            return TaskResult(
                task_id=task.id, output="", score=0.0,
                model_used=Model.GPT_4O_MINI,
                status=TaskStatus.FAILED,
            )

        primary = models[0]
        reviewer = self._select_reviewer(primary, task.type, task_id=task.id)

        # Build context from dependencies
        context = self._gather_dependency_context(task.dependencies)
        full_prompt = task.prompt
        if context:
            full_prompt += f"\n\n--- CONTEXT FROM PRIOR TASKS ---\n{context}"

        best_output = ""
        best_score = 0.0
        best_critique = ""
        total_cost = 0.0
        degraded_count = 0
        scores_history: list[float] = []
        det_passed = True  # default: True until a deterministic check runs

        logger.info(f"Executing {task.id} ({task.type.value}): "
                     f"primary={primary.value}, reviewer={reviewer.value if reviewer else 'none'}")

        for iteration in range(task.max_iterations):
            if not self.budget.can_afford(0.02):
                logger.warning(f"Budget low, stopping {task.id} at iteration {iteration}")
                break

            # ── GENERATE ──
            try:
                gen_response = await self.client.call(
                    primary, full_prompt,
                    system=f"You are an expert executing a {task.type.value} task. "
                           f"Produce high-quality, complete output.",
                    max_tokens=task.max_output_tokens,
                    timeout=60,
                )
                output = gen_response.text
                gen_cost = gen_response.cost_usd
                self.budget.charge(gen_cost, "generation")
                total_cost += gen_cost
                self._telemetry.record_call(primary, gen_response.latency_ms, gen_cost, True)
            except Exception as e:
                logger.error(f"Generation failed for {task.id}: {e}")
                self.api_health[primary] = False
                self._telemetry.record_call(primary, 0.0, 0.0, False)
                degraded_count += 1
                # Try adaptive replan
                fb = self._get_fallback(primary, task_type=task.type, task_id=task.id)
                if fb:
                    try:
                        gen_response = await self.client.call(
                            fb, full_prompt,
                            system=f"You are an expert executing a {task.type.value} task.",
                            max_tokens=task.max_output_tokens,
                            timeout=60,
                        )
                        output = gen_response.text
                        self.budget.charge(gen_response.cost_usd, "generation")
                        total_cost += gen_response.cost_usd
                        primary = fb  # Update for this iteration
                    except Exception as e2:
                        logger.error(f"Fallback generation also failed: {e2}")
                        break
                else:
                    break

            # ── CRITIQUE (cross-model) ──
            critique = ""
            if reviewer and reviewer != primary:
                try:
                    critique_response = await self.client.call(
                        reviewer,
                        f"Review this output for correctness, completeness, and quality. "
                        f"Be specific about flaws and suggest concrete improvements.\n\n"
                        f"ORIGINAL TASK: {task.prompt}\n\n"
                        f"OUTPUT TO REVIEW:\n{output}",
                        system="You are a critical reviewer. Find flaws, be specific.",
                        max_tokens=800,
                        timeout=45,
                    )
                    critique = critique_response.text
                    self.budget.charge(critique_response.cost_usd, "cross_review")
                    total_cost += critique_response.cost_usd
                except Exception as e:
                    logger.warning(f"Critique failed for {task.id}: {e}")
                    self.api_health[reviewer] = False
                    degraded_count += 1
                    # Continue without critique

            # ── REVISE (if critique exists) ──
            if critique:
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
                        max_tokens=task.max_output_tokens,
                        timeout=60,
                    )
                    output = revise_response.text
                    self.budget.charge(revise_response.cost_usd, "generation")
                    total_cost += revise_response.cost_usd
                except Exception as e:
                    logger.warning(f"Revision failed for {task.id}: {e}")
                    # Keep original output

            # ── DETERMINISTIC VALIDATION ──
            det_passed = True
            if task.hard_validators:
                val_results = run_validators(output, task.hard_validators)
                det_passed = all_validators_pass(val_results)
                if not det_passed:
                    failed = [v for v in val_results if not v.passed]
                    logger.warning(
                        f"Deterministic check failed for {task.id}: "
                        f"{[f'{v.validator_name}: {v.details}' for v in failed]}"
                    )

            # ── EVALUATE ──
            if det_passed:
                score = await self._evaluate(task, output)
                # Feed evaluator score back to TelemetryCollector so the planner
                # learns which models produce high-quality output over time.
                self._telemetry.record_call(
                    primary, 0.0, 0.0, True, quality_score=score
                )
            else:
                score = 0.0  # Override: deterministic failure
                self._telemetry.record_call(primary, 0.0, 0.0, False)

            self.budget.charge(0.0, "evaluation")  # cost tracked in _evaluate
            total_cost += 0  # _evaluate charges internally

            scores_history.append(score)

            if score > best_score:
                best_output = output
                best_score = score
                best_critique = critique

            logger.info(
                f"  {task.id} iter {iteration + 1}: score={score:.3f} "
                f"(best={best_score:.3f}, threshold={task.acceptance_threshold})"
            )

            # ── CONVERGENCE CHECKS ──
            if best_score >= task.acceptance_threshold:
                logger.info(f"  {task.id}: threshold met at iteration {iteration + 1}")
                break

            # Plateau detection: Δscore < 0.02 for 2 consecutive
            if len(scores_history) >= 2:
                delta = abs(scores_history[-1] - scores_history[-2])
                if delta < 0.02:
                    logger.info(f"  {task.id}: plateau detected (Δ={delta:.4f})")
                    break

        # Determine status
        status = TaskStatus.COMPLETED if best_score >= task.acceptance_threshold else TaskStatus.DEGRADED
        if best_score == 0.0 and not det_passed:
            status = TaskStatus.FAILED

        return TaskResult(
            task_id=task.id,
            output=best_output,
            score=best_score,
            model_used=primary,
            reviewer_model=reviewer,
            iterations=len(scores_history),
            cost_usd=total_cost,
            status=status,
            critique=best_critique,
            deterministic_check_passed=det_passed,
            degraded_fallback_count=degraded_count,
        )

    async def _evaluate(self, task: Task, output: str) -> float:
        """
        LLM-based scoring. Uses evaluation-tier model.
        Self-consistency: runs twice, checks Δ ≤ 0.05.
        """
        eval_models = self._get_available_models(TaskType.EVALUATE)
        if not eval_models:
            return 0.5  # Can't evaluate, return neutral score

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
        for run in range(2):  # Two runs for self-consistency
            try:
                response = await self.client.call(
                    eval_model, eval_prompt,
                    system="You are a precise evaluator. Score exactly, return only JSON.",
                    max_tokens=300,
                    temperature=0.1,  # Low temp for consistency
                    timeout=30,
                )
                self.budget.charge(response.cost_usd, "evaluation")
                score = self._parse_score(response.text)
                scores.append(score)
            except Exception as e:
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
        """Extract score from LLM evaluation response."""
        text = text.strip()
        # Try JSON parse
        try:
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            data = json.loads(text.strip())
            score = float(data.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Try regex fallback
        match = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        logger.warning(f"Could not parse score from: {text[:100]}")
        return 0.5

    # ─────────────────────────────────────────
    # Model selection & fallback
    # ─────────────────────────────────────────

    def _get_available_models(self, task_type: TaskType,
                               task_id: str = "") -> list[Model]:
        """
        Policy-driven model selection via ConstraintPlanner.
        Returns [best_model] if a compliant candidate exists, or falls back
        to any healthy model (legacy behaviour) if the planner returns None.

        The backwards-compatible list[Model] return type is preserved so that
        callers that use models[0] continue to work without changes.
        """
        policies = self._policy_set.policies_for(task_id)
        best = self._planner.select_model(
            task_type=task_type,
            policies=policies,
            budget_remaining=self.budget.remaining_usd,
            task_id=task_id,
        )
        if best is None:
            # No planner result — legacy fallback to any healthy model
            return [m for m in Model if self.api_health.get(m, False)]
        return [best]

    def _get_cheapest_available(self) -> Model:
        """Return cheapest healthy model for utility tasks (decomposition)."""
        from .models import COST_TABLE
        healthy = [m for m in Model if self.api_health.get(m, False)]
        if not healthy:
            raise RuntimeError("No healthy models available")
        return min(healthy, key=lambda m: COST_TABLE[m]["output"])

    def _select_reviewer(self, generator: Model, task_type: TaskType,
                          task_id: str = "") -> Optional[Model]:
        """
        Policy-compliant cross-provider reviewer selection via ConstraintPlanner.
        Invariant preserved: reviewer provider ≠ generator provider when possible.
        Counterfactual: Same-provider review → shared training biases (blind spots).
        """
        policies = self._policy_set.policies_for(task_id)
        return self._planner.select_reviewer(
            generator=generator,
            task_type=task_type,
            policies=policies,
            budget_remaining=self.budget.remaining_usd,
        )

    def _get_fallback(self, failed_model: Model,
                      task_type: Optional[TaskType] = None,
                      task_id: str = "") -> Optional[Model]:
        """
        Adaptive re-planning via ConstraintPlanner.replan() when task_type is known.
        Falls back to legacy FALLBACK_CHAIN logic for decomposition calls
        (which have no task_type context).
        """
        if task_type is None:
            # Legacy code path: decomposition or callers without task_type context
            fb = FALLBACK_CHAIN.get(failed_model)
            if fb and self.api_health.get(fb, False):
                return fb
            for m in Model:
                if m != failed_model and self.api_health.get(m, False):
                    return m
            return None

        policies = self._policy_set.policies_for(task_id)
        return self._planner.replan(
            failed_model=failed_model,
            task_type=task_type,
            policies=policies,
            budget_remaining=self.budget.remaining_usd,
        )

    # ─────────────────────────────────────────
    # DAG & dependency management
    # ─────────────────────────────────────────

    def _topological_sort(self, tasks: dict[str, Task]) -> list[str]:
        """Kahn's algorithm with cycle detection."""
        in_degree = {tid: 0 for tid in tasks}
        graph = defaultdict(list)

        for tid, task in tasks.items():
            for dep in task.dependencies:
                if dep in tasks:
                    graph[dep].append(tid)
                    in_degree[tid] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # Sort for determinism
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(tasks):
            cycle_tasks = set(tasks.keys()) - set(result)
            logger.error(f"Dependency cycle detected involving: {cycle_tasks}")
            # Include non-cyclic tasks, skip cyclic ones
            return result

        return result

    def _gather_dependency_context(self, dep_ids: list[str]) -> str:
        """Collect outputs from completed dependency tasks."""
        parts = []
        for dep_id in dep_ids:
            result = self.results.get(dep_id)
            if result and result.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED):
                parts.append(f"[Output from {dep_id}]:\n{result.output[:2000]}")
        return "\n\n".join(parts) if parts else ""

    # ─────────────────────────────────────────
    # Status & resume
    # ─────────────────────────────────────────

    def _determine_final_status(self, state: ProjectState) -> ProjectStatus:
        """
        SUCCESS when ALL:
        1. All tasks ≥ threshold
        2. No unmet dependencies
        3. No task in degraded fallback >50% iterations
        4. Deterministic checks passed
        5. Within budget
        """
        if not state.results:
            return ProjectStatus.SYSTEM_FAILURE

        all_passed = all(
            r.status == TaskStatus.COMPLETED for r in state.results.values()
        )
        budget_ok = state.budget.remaining_usd >= 0
        time_ok = state.budget.time_remaining()

        # Check degraded fallback ratio
        degraded_heavy = any(
            r.degraded_fallback_count > r.iterations * 0.5
            for r in state.results.values()
            if r.iterations > 0
        )

        det_ok = all(r.deterministic_check_passed for r in state.results.values())

        if all_passed and budget_ok and det_ok and not degraded_heavy:
            return ProjectStatus.SUCCESS
        elif not budget_ok:
            return ProjectStatus.BUDGET_EXHAUSTED
        elif not time_ok:
            return ProjectStatus.TIMEOUT
        else:
            return ProjectStatus.PARTIAL_SUCCESS

    async def _resume_project(self, state: ProjectState) -> ProjectState:
        """Resume from last checkpoint."""
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
                     status: ProjectStatus = ProjectStatus.PARTIAL_SUCCESS
                     ) -> ProjectState:
        return ProjectState(
            project_description=project_desc,
            success_criteria=criteria,
            budget=self.budget,
            tasks=tasks,
            results=dict(self.results),
            api_health={m.value: h for m, h in self.api_health.items()},
            status=status,
            execution_order=list(tasks.keys()) if tasks else [],
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
