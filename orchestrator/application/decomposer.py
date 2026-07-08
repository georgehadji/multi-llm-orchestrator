"""
DecomposerService — stable project decomposition interface.
===========================================================
Wraps engine.py._decompose() via callback injection. Establishes the
decompose(project, criteria) -> GeneratorResult boundary.

Part of Application Layer (Phase 4) — Canonical location.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..api_clients import UnifiedClient
from ..crosscutting.config import flags
from ..exceptions import OrchestratorError, TaskError
from ..models import Model, Task, TaskType, VSConfig
from ..resilience import ResiliencePolicy
from ..tracing import Tracer
from ..resilience import ResiliencePolicy as _ResiliencePolicy
from ..project_context import ProjectContext as _ProjectContext

_service_logger = logging.getLogger("orchestrator.services.generator")

DecomposeFn = Callable[..., Awaitable[dict[str, Task]]]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DecomposerResult:
    """Outcome of a decomposition call."""

    tasks: dict[str, Task]
    wall_time_ms: float
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.tasks)

    @property
    def task_count(self) -> int:
        return len(self.tasks)


@dataclass
class DecomposerMetrics:
    """Monotonic counters for decomposition calls."""

    total_calls: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_tasks_generated: int = 0
    cumulative_wall_ms: float = 0.0

    def record(self, result: DecomposerResult) -> None:
        self.total_calls += 1
        self.cumulative_wall_ms += result.wall_time_ms
        if result.succeeded:
            self.total_succeeded += 1
            self.total_tasks_generated += result.task_count
        else:
            self.total_failed += 1

    def to_dict(self) -> dict[str, Any]:
        avg_ms = self.cumulative_wall_ms / self.total_calls if self.total_calls else 0.0
        return {
            "total_calls": self.total_calls,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "total_tasks_generated": self.total_tasks_generated,
            "avg_wall_ms": round(avg_ms, 1),
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DecomposerService:
    """
    Application-layer service for project decomposition.

    Usage:
        self._decomposer = DecomposerService(decompose_fn=self._decompose)
        result = await self._decomposer.decompose(project, criteria)
        if not result.succeeded:
            raise OrchestratorError(f"Decomposition failed: {result.error}")
        tasks = result.tasks
    """

    def __init__(
        self,
        decompose_fn: DecomposeFn,
        decompose_timeout: float | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._decompose_fn = decompose_fn
        self._decompose_timeout = decompose_timeout
        self._tracer = tracer
        self.metrics = DecomposerMetrics()
        self._lock = asyncio.Lock()

    @property
    def decompose_fn(self) -> Any:
        """Late-bound decompose_fn — allows container.wire_executor() to set."""
        return self._decompose_fn

    @decompose_fn.setter
    def decompose_fn(self, fn: Any) -> None:
        self._decompose_fn = fn

    @staticmethod
    def _safe_float(value: Any, default: float, min_val: float, max_val: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(min_val, min(max_val, parsed))

    @staticmethod
    def _safe_int(value: Any, default: int, min_val: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(min_val, parsed)

    async def decompose(
        self,
        project: str,
        criteria: str,
        policy: _ResiliencePolicy | None = None,
        project_context: "ProjectContext | None" = None,
        **kwargs: Any,
    ) -> DecomposerResult:
        """Decompose project into an ordered task dict. Never raises."""
        t0 = time.monotonic()

        if self._tracer is not None:
            with self._tracer.trace(
                "generator.decompose",
                {"project": project[:50], "criteria": criteria[:50]},
            ) as span:
                tasks, error = await self._run_with_guard(
                    project, criteria, policy, project_context=project_context, **kwargs
                )
                if error:
                    span.set_status("ERROR")
                    span.add_event("exception", {"exception.message": str(error)})
        else:
            tasks, error = await self._run_with_guard(
                project, criteria, policy, project_context=project_context, **kwargs
            )

        wall_ms = (time.monotonic() - t0) * 1000

        result = DecomposerResult(tasks=tasks or {}, wall_time_ms=wall_ms, error=error)

        async with self._lock:
            self.metrics.record(result)

        if error:
            _service_logger.warning("decompose FAILED in %.0fms: %s", wall_ms, error)
        else:
            _service_logger.debug(
                "decompose succeeded in %.0fms — %d tasks", wall_ms, result.task_count
            )

        return result

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.to_dict()

    async def _run_with_guard(
        self,
        project: str,
        criteria: str,
        policy: _ResiliencePolicy | None = None,
        project_context: _ProjectContext | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Task] | None, Exception | None]:
        try:
            decompose_kwargs = dict(kwargs)
            decompose_kwargs["policy"] = policy
            if project_context is not None:
                decompose_kwargs["project_context"] = project_context

            if self._decompose_timeout is not None:
                raw = await asyncio.wait_for(
                    self._decompose_fn(project, criteria, **decompose_kwargs),
                    timeout=self._decompose_timeout,
                )
            else:
                raw = await self._decompose_fn(project, criteria, **decompose_kwargs)
            return raw, None

        except asyncio.TimeoutError as exc:
            wrapped = OrchestratorError(
                f"Decomposition timed out after {self._decompose_timeout}s",
                cause=exc,
            )
            return None, wrapped

        except (OrchestratorError, TaskError) as exc:
            return None, exc

        except Exception as exc:
            wrapped = OrchestratorError(f"Unexpected decomposition error: {exc}", cause=exc)
            return None, wrapped


if TYPE_CHECKING:
    from ..project_context import ProjectContext
    from ..model_selector import ModelSelector

# Lazy import for VS-powered decomposition
_VS_SAMPLER_DECOMP = None
_VS_DECOMP_LOCK = __import__("threading").Lock()


def _get_vs_decomp_sampler(client: Any) -> Any:
    global _VS_SAMPLER_DECOMP
    if _VS_SAMPLER_DECOMP is None:
        with _VS_DECOMP_LOCK:
            if _VS_SAMPLER_DECOMP is None:
                from ..application.verbalized_sampling import VerbalizedSampler as _VS

                _VS_SAMPLER_DECOMP = _VS
    return _VS_SAMPLER_DECOMP(client=client)


logger = logging.getLogger(
    "orchestrator.engine_core.decomposer"
)  # Legacy — preserved for Decomposer class logs


class Decomposer:
    """Project decomposition into atomic tasks.

    Wraps the model call, JSON parsing, and partial-recovery logic
    that engine.py previously held inline.

    Args:
        client: UnifiedClient for LLM calls.
        selector: ModelSelector for task-specific model routing.
        tracer: Optional OpenTelemetry tracer for spans.
    """

    def __init__(
        self,
        client: UnifiedClient,
        selector: ModelSelector,
        tracer: Tracer | None = None,
    ) -> None:
        self._client = client
        self._selector = selector
        self._tracer = tracer

    async def decompose(
        self,
        project: str,
        criteria: str,
        project_id: str = "",
        app_profile: Any = None,
        project_context: ProjectContext | None = None,
        policy: ResiliencePolicy | None = None,
        api_health: dict[Model, bool] | None = None,
        output_dir: Path | None = None,
        record_failure_fn: Any = None,
        charge_fn: Any = None,
    ) -> dict[str, Task]:
        """Break a project description into an ordered dict of atomic tasks.

        Never raises — returns empty dict on irrecoverable failure.

        Args:
            project: Project description.
            criteria: Success criteria.
            project_id: Optional project identifier for context injection.
            app_profile: Optional AppProfile / ArchitectureDecision.
            project_context: Optional ProjectContext for cross-phase knowledge.
            policy: Resilience policy for retry behaviour.
            api_health: Optional model health map (from engine).
            output_dir: Optional output directory for file references.
            record_failure_fn: Optional callback for failure tracking.

        Returns:
            Ordered dict of task_id -> Task, or empty dict on failure.
        """
        valid_types = [t.value for t in TaskType]

        # Build optional app-context block injected into the prompt
        app_context_block = ""
        if app_profile is not None:
            try:
                from orchestrator.scaffold import _TEMPLATE_MAP
                from orchestrator.scaffold.templates import generic

                app_type = app_profile.app_type if hasattr(app_profile, "app_type") else "script"
                template_files = _TEMPLATE_MAP.get(app_type, generic.FILES)
                scaffold_list = "\n".join(f"  - {p}" for p in sorted(template_files))
                tech_stack_str = (
                    ", ".join(app_profile.tech_stack)
                    if hasattr(app_profile, "tech_stack") and app_profile.tech_stack
                    else "unknown"
                )

                # Build architecture block if ArchitectureDecision fields are present
                arch_block = ""
                if hasattr(app_profile, "structural_pattern") and app_profile.structural_pattern:
                    rationale_line = (
                        f"\n  Rationale:          {app_profile.rationale}"
                        if hasattr(app_profile, "rationale") and app_profile.rationale
                        else ""
                    )
                    topology = getattr(app_profile, "topology", "")
                    api_paradigm = getattr(app_profile, "api_paradigm", "")
                    data_paradigm = getattr(app_profile, "data_paradigm", "")
                    arch_block = f"""ARCHITECTURE DECISION:
  Structural pattern: {app_profile.structural_pattern}
  Topology:           {topology}
  API paradigm:       {api_paradigm}
  Data paradigm:      {data_paradigm}{rationale_line}

Each task MUST follow this architecture — do not invent an alternative structure.
"""

                app_context_block = f"""APP_TYPE: {app_type}
TECH_STACK: {tech_stack_str}
SCAFFOLD_FILES (already exist — fill or extend these):
{scaffold_list}
{arch_block}
Each task JSON element MUST also include:
  - "target_path": relative path to the file this task creates/modifies
  - "module_name": dotted Python module path matching target_path
  - "tech_context": brief note on libraries or patterns to use
"""
            except ImportError:
                logger.debug("scaffold module not available, skipping app context block")

                # Phase 5: inject project context
        if project_context is not None and not project_context.is_empty():
            ctx_str = project_context.to_system_prompt()
            if ctx_str:
                app_context_block = (app_context_block + "\n" + ctx_str).strip()

        # Build system and user prompts
        system = (
            "You are a senior software engineer and architect.\n"
            "Break down the given project specification into atomic, ordered tasks.\n"
            "Each task should be a single file or a tightly coupled pair of files.\n"
            "Dependencies MUST form a DAG — no circular dependencies.\n"
            f"Valid task types: {valid_types}\n"
        )

        prompt = f"PROJECT: {project}\n" f"SUCCESS CRITERIA: {criteria}\n\n" f"{app_context_block}"

        # Determine models to try
        models_to_try = self._get_decomposition_models(project, api_health)

        last_response_text: str | None = None

        for attempt, model in enumerate(models_to_try):
            model_name = model.value if hasattr(model, "value") else str(model)
            try:
                # ── VS multi-plan decomposition ──────────────────────────
                if flags.vs_decomposition and attempt == 0:
                    sampler = _get_vs_decomp_sampler(self._client)
                    vs_system = (
                        system + "\n"
                        "Generate 2 complete task plans. Each plan should be a "
                        "full task decomposition covering a different architectural "
                        "approach. Return them as separate JSON task arrays."
                    )
                    candidates = await sampler.sample(
                        prompt=prompt,
                        model=model,
                        cfg=VSConfig(k=2, temperature=0.3),
                        system_extra=vs_system,
                        max_tokens=8192 * 2,
                        timeout=160,
                    )
                    if candidates:
                        for c in candidates:
                            parsed = self._parse_decomposition(c.text)
                            if parsed:
                                logger.info(
                                    "VS decomposition: plan (prob=%.2f) with %d tasks",
                                    c.probability,
                                    len(parsed),
                                )
                                return parsed

                # Standard single-call decomposition
                response = await self._client.call(
                    model=model,
                    prompt=prompt,
                    system=system,
                    max_tokens=8192,
                    temperature=0.3,
                    timeout=160,
                    retries=2,
                )
                last_response_text = response.text

                parsed = self._parse_decomposition(response.text)
                if parsed:
                    if charge_fn is not None:
                        try:
                            await charge_fn(getattr(response, "cost_usd", 0.0))
                        except Exception:
                            logger.debug("charge_fn failed for decomposition", exc_info=True)
                    logger.info(
                        "Decomposition succeeded on attempt %d with %s (%d tasks)",
                        attempt + 1,
                        model_name,
                        len(parsed),
                    )
                    return parsed

                # Try partial recovery
                partial = self._try_parse_partial_json_array(response.text)
                if partial:
                    recovered = self._repair_partial_tasks(partial)
                    if recovered:
                        logger.info(
                            "Partial recovery succeeded with %s (%d tasks)",
                            model_name,
                            len(recovered),
                        )
                        return recovered

                logger.warning(
                    "Decomposition attempt %d with %s produced unparseable JSON",
                    attempt + 1,
                    model_name,
                )

            except (json.JSONDecodeError, ValueError) as e:
                raw_preview = (last_response_text or "N/A")[:300]
                logger.warning(
                    "Decomposition attempt %d with %s failed (JSON parse): %s",
                    attempt + 1,
                    model_name,
                    e,
                )
                logger.warning(f"  Raw response (first 300 chars): {raw_preview}...")
                if record_failure_fn:
                    await record_failure_fn(model, error=e)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    "Decomposition attempt %d with %s failed: %s",
                    attempt + 1,
                    model_name,
                    e,
                )
                if record_failure_fn:
                    await record_failure_fn(model, error=e)

        logger.error("All decomposition attempts failed")
        return {}

    def _try_parse_partial_json_array(self, text: str) -> list[Any] | None:
        """Attempt to parse a potentially truncated JSON array.

        When LLM responses get cut off mid-stream, we may have valid JSON objects
        at the start but missing the closing brackets. This method tries multiple
        strategies to recover as much data as possible.

        Args:
            text: Potentially truncated JSON array text

        Returns:
            List of parsed objects if recovery succeeds, None otherwise.
        """
        text = text.strip()
        if not text:
            return None

        # Strategy 1: Try to close a JSON array
        if text.startswith("["):
            # Count opening and closing brackets
            open_count = text.count("[")
            close_count = text.count("]")
            if open_count > close_count:
                # Add missing closing brackets
                fixed = text + "]" * (open_count - close_count)
                try:
                    result = json.loads(fixed)
                    if isinstance(result, list):
                        logger.debug(
                            "Partial recovery: added %d closing bracket(s)",
                            open_count - close_count,
                        )
                        return result
                except json.JSONDecodeError:
                    pass

        # Strategy 2: Extract individual JSON objects using regex
        objects = []
        pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue

        if objects:
            logger.info("Recovered %d task objects via pattern extraction", len(objects))
            return objects

        return None

    def _repair_partial_tasks(
        self, partial_objects: list[dict[str, Any]]
    ) -> dict[str, Task] | None:
        """Convert partially recovered JSON objects into Tasks.

        Args:
            partial_objects: List of dicts recovered from partial JSON parsing.

        Returns:
            Dict of task_id -> Task if any valid tasks found, None otherwise.
        """
        tasks: dict[str, Task] = {}
        for obj in partial_objects:
            task_id = obj.get("id", "") or obj.get("task_id", "") or obj.get("name", "")
            task_type_str = obj.get("type", "") or obj.get("task_type", "") or ""
            prompt = obj.get("prompt", "") or obj.get("description", "")

            if task_id and prompt and task_type_str:
                try:
                    task_type = TaskType(task_type_str)
                except ValueError:
                    continue

                tasks[task_id] = Task(
                    id=task_id,
                    type=task_type,
                    prompt=prompt,
                    context=obj.get("context", obj.get("dependencies_context", "")),
                    dependencies=obj.get("dependencies", []),
                    acceptance_threshold=float(obj.get("acceptance_threshold", 0.85)),
                    max_iterations=int(obj.get("max_iterations", 3)),
                    target_path=obj.get("target_path", ""),
                    module_name=obj.get("module_name", ""),
                    tech_context=obj.get("tech_context", ""),
                )

        return tasks if tasks else None

    def _parse_decomposition(self, text: str) -> dict[str, Task]:
        """Parse LLM output into Task objects with defensive handling.

        P2-1 OPTIMIZATION: Uses json5 library for robust JSON parsing that handles:
        - Trailing commas
        - Single-quoted strings
        - Comments
        - Unquoted keys

        This replaces the multi-pass regex approach with a single robust parse.

        Args:
            text: Raw LLM response text.

        Returns:
            Dict of task_id -> Task on success, empty dict on failure.
        """
        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop the opening fence line
            lines = lines[1:]
            # Drop closing fence if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if not text:
            logger.warning("Empty response after stripping fences")
            return {}

        # Try json5 first (more lenient parsing)
        tasks_json = None
        try:
            import json5  # noqa: F401

            try:
                tasks_json = json5.loads(text)
                if isinstance(tasks_json, dict) and "tasks" in tasks_json:
                    tasks_json = tasks_json["tasks"]
            except Exception:
                logger.debug("json5 parse failed, falling back to standard json")
                tasks_json = None
        except ImportError:
            logger.debug("json5 not available, using standard json")

        # Fallback to standard json
        if tasks_json is None:
            try:
                tasks_json = json.loads(text)
                if isinstance(tasks_json, dict) and "tasks" in tasks_json:
                    tasks_json = tasks_json["tasks"]
            except json.JSONDecodeError:
                logger.debug("Standard JSON parse failed")
                return {}

        if not isinstance(tasks_json, list):
            logger.warning("Parsed JSON is not a list of tasks")
            return {}

        tasks: dict[str, Task] = {}
        for item in tasks_json:
            if not isinstance(item, dict):
                continue

            task_id = item.get("id")
            task_type = item.get("type")
            prompt = item.get("prompt") or item.get("description", "")

            if not task_id or not task_type or not prompt:
                continue

            # Normalize task type
            if isinstance(task_type, str):
                try:
                    task_type_enum = TaskType(task_type)
                except ValueError:
                    # Try common mappings
                    type_mapping = {
                        "code_gen": TaskType.CODE_GEN,
                        "code_generation": TaskType.CODE_GEN,
                        "code_review": TaskType.CODE_REVIEW,
                        "complex_reasoning": TaskType.REASONING,
                        "creative_writing": TaskType.WRITING,
                        "data_extraction": TaskType.DATA_EXTRACT,
                        "summarization": TaskType.SUMMARIZE,
                        "evaluation": TaskType.EVALUATE,
                    }
                    mapped = type_mapping.get(task_type.lower().replace(" ", "_"))
                    if mapped is None:
                        continue
                    task_type_enum = mapped
            else:
                continue

            # Extract dependencies
            deps = item.get("dependencies", [])
            if isinstance(deps, str):
                deps = [deps] if deps else []

            tasks[task_id] = Task(
                id=task_id,
                type=task_type_enum,
                prompt=prompt,
                context=item.get("context", ""),
                dependencies=deps,
                acceptance_threshold=DecomposerService._safe_float(
                    item.get("acceptance_threshold", 0.85), default=0.85, min_val=0.0, max_val=1.0
                ),
                max_iterations=DecomposerService._safe_int(
                    item.get("max_iterations", 3), default=3, min_val=1
                ),
                target_path=item.get("target_path", ""),
                module_name=item.get("module_name", ""),
                tech_context=item.get("tech_context", ""),
            )

        if not tasks:
            logger.warning("No valid tasks could be parsed from LLM response")
            return {}

        logger.debug("Parsed %d tasks from decomposition response", len(tasks))
        return tasks

    def _get_decomposition_models(
        self,
        project_description: str,
        api_health: dict[Model, bool] | None = None,
    ) -> list[Any]:
        """Get prioritized list of models for decomposition."""
        from ..models import FALLBACK_CHAIN, Model as _M

        primary = self._selector.decomposition_model(project_description)
        fallback = FALLBACK_CHAIN.get(primary, primary)
        models = [primary, fallback]
        if _M.QWEN_3_6_FLASH not in models:
            models.append(_M.QWEN_3_6_FLASH)
        if _M.XIAOMI_MIMO_V2_FLASH not in models:
            models.append(_M.XIAOMI_MIMO_V2_FLASH)
        return models
