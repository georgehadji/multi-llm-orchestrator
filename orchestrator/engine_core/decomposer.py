"""
Decomposer — Project decomposition into atomic tasks
======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles the full decomposition lifecycle: model selection, prompt
construction, LLM call, JSON parsing, and partial-result recovery.

Extracted from engine.py Phase 1: Decomposition (Strangler Fig pattern).
Orchestrator._decompose() delegates to this class.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..api_clients import UnifiedClient
from ..exceptions import OrchestratorError
from ..models import Model, Task, TaskType
from ..resilience import ResiliencePolicy, RetryTemplate
from ..tracing import Tracer

if TYPE_CHECKING:
    from ..project_context import ProjectContext
    from ..model_selector import ModelSelector

logger = logging.getLogger("orchestrator.engine_core.decomposer")


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
                    arch_block = f"""ARCHITECTURE DECISION:
  Structural pattern: {app_profile.structural_pattern}
  Topology:           {app_profile.topology}
  API paradigm:       {app_profile.api_paradigm}
  Data paradigm:      {app_profile.data_paradigm}{rationale_line}

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

        prompt = (
            "PROJECT: {project}\n" "SUCCESS CRITERIA: {criteria}\n\n" "{app_context_block}"
        ).format(project=project, criteria=criteria, app_context_block=app_context_block)

        # Determine models to try
        models_to_try = self._get_decomposition_models(project, api_health)

        last_response_text: str | None = None

        for attempt, model in enumerate(models_to_try):
            model_name = model.value if hasattr(model, "value") else str(model)
            try:
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
            except (Exception, asyncio.CancelledError) as e:
                logger.error(
                    "Decomposition attempt %d with %s failed: %s",
                    attempt + 1,
                    model_name,
                    e,
                )
                if record_failure_fn:
                    await record_failure_fn(model, error=e)

        logger.error("All decomposition attempts failed")
        raise OrchestratorError(
            "Project decomposition failed: unable to parse LLM response after multiple attempts. "
            "The model may be experiencing issues or the project description may be too complex. "
            "Try simplifying the project description or using a different model."
        )

    def _try_parse_partial_json_array(self, text: str) -> list | None:
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

    def _repair_partial_tasks(self, partial_objects: list[dict]) -> dict[str, Task] | None:
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
                acceptance_threshold=float(item.get("acceptance_threshold", 0.85)),
                max_iterations=int(item.get("max_iterations", 3)),
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
        api_health: list = None,
    ) -> list:
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
