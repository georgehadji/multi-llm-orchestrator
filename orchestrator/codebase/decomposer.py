"""
CodebaseDecomposer — Plan modifications to an existing codebase
=================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Phase 3 of the Codebase-Aware Orchestrator enhancement.
Given an objective and codebase context, produces a task plan
for modifying an existing codebase.

Optimization: Decomposer Task Consolidation (4.4)
- Post-decomposition stage merges MODIFY_FILE tasks targeting the same file
- Reduces orchestration overhead and critique cycles significantly
"""

from __future__ import annotations

import json
import logging

from ..api_clients import UnifiedClient
from ..codebase_context import CodebaseContext
from ..models import Model, Task, TaskType

logger = logging.getLogger("orchestrator.codebase_decomposer")


class CodebaseDecomposer:
    """Generate a plan of modification tasks from an objective + codebase context.

    Example:
        Objective: "Add JWT authentication"
        -> Task 1: INSTALL_DEP "python-jose[cryptography]"
        -> Task 2: MODIFY_FILE src/auth.py (insert auth middleware)
        -> Task 3: MODIFY_FILE src/routes/users.py (add login endpoint)
        -> Task 4: CODE_GEN tests/test_auth.py (new test file)
    """

    def __init__(
        self,
        client: UnifiedClient,
        model: Model = Model.GPT_4O_MINI,
    ) -> None:
        self._client = client
        self._model = model

    async def decompose(
        self,
        objective: str,
        context: CodebaseContext,
    ) -> dict[str, Task]:
        """Produce a task plan for the given objective.

        Args:
            objective: What to do (e.g. "Add JWT authentication").
            context: CodebaseContext with reader output.

        Returns:
            Dict of task_id -> Task with modification instructions.
        """
        # Build the LLM prompt with codebase context
        codebase_prompt = context.to_llm_prompt(objective=objective)
        context_prompt = (
            f"You are a senior software engineer modifying an existing codebase.\\n\\n"
            f"{codebase_prompt}\\n\\n"
            f"## Objective\\n{objective}\\n\\n"
            f"## Task\\n"
            f"Given the codebase context above, produce a list of tasks to {objective}.\\n\\n"
            f"For each task, return JSON with:\\n"
            f"- `id`: unique task identifier\\n"
            f"- `type`: one of {[t.value for t in TaskType]}\\n"
            f"- `prompt`: detailed instructions for what to do\\n"
            f"- `target_path`: relative file path (for MODIFY_FILE, CODE_GEN, DELETE_FILE)\\n"
            f'- `modification_strategy`: one of "replace", "insert", "patch" (for MODIFY_FILE)\\n'
            f"- `dependencies_to_install`: list of pip package names (for INSTALL_DEP)\\n"
            f"- `dependencies`: list of task IDs this task depends on\\n"
            f"\\n"
            f'Return ONLY valid JSON with a "tasks" key containing the list. '
            f"Priority: first install dependencies, then modify/create files, then add tests."
        )

        try:
            response = await self._client.call(
                model=self._model,
                system=(
                    "You are an expert software engineer. Generate a precise, "
                    "actionable modification plan for an existing codebase."
                ),
                prompt=context_prompt,
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            logger.error("CodebaseDecomposer LLM call failed: %s", exc)
            # Fallback: return a single CODE_GEN task from the objective
            return self._fallback_plan(objective, None)

        return self._parse_response(response.text, objective)

    def _parse_response(self, text: str, objective: str) -> dict[str, Task]:
        """Parse LLM JSON response into Task objects."""
        import re

        # Strip markdown fences
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return self._fallback_plan(objective, "JSON parse error")
            else:
                return self._fallback_plan(objective, "No JSON found")

        tasks_raw = data.get("tasks", data) if isinstance(data, dict) else data
        if isinstance(tasks_raw, dict):
            tasks_raw = [tasks_raw]

        tasks: dict[str, Task] = {}
        for item in tasks_raw if isinstance(tasks_raw, list) else []:
            if not isinstance(item, dict):
                continue
            task_id = item.get("id", f"task_{len(tasks) + 1:03d}")
            task_type_str = item.get("type", "code_generation")
            prompt = item.get("prompt", item.get("description", ""))

            if not prompt:
                continue

            # Normalize task type
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.CODE_GEN

            deps = item.get("dependencies", [])
            if isinstance(deps, str):
                deps = [deps] if deps else []

            install_deps = item.get("dependencies_to_install", [])
            if isinstance(install_deps, str):
                install_deps = [install_deps] if install_deps else []

            task = Task(
                id=task_id,
                type=task_type,
                prompt=prompt,
                dependencies=deps,
                target_path=item.get("target_path", ""),
                max_output_tokens=4096,
            )
            # Set dynamic attributes used by CodebaseWriter
            object.__setattr__(
                task, "modification_strategy", item.get("modification_strategy", "replace")
            )
            object.__setattr__(task, "dependencies_to_install", install_deps)

            tasks[task_id] = task

        if not tasks:
            return self._fallback_plan(objective, "No tasks parsed")

        # Optimization 4.4: Consolidate MODIFY_FILE tasks by target_path
        tasks = self._consolidate_tasks(tasks)

        return tasks

    @staticmethod
    def _consolidate_tasks(tasks: dict[str, Task]) -> dict[str, Task]:
        """Merge MODIFY_FILE tasks that target the same file into one task.

        When multiple modification tasks target the same file, this reduces
        orchestration overhead by combining their prompts into a single task.

        Args:
            tasks: Dict of task_id -> Task to consolidate.

        Returns:
            Consolidated dict of task_id -> Task.
        """
        from collections import defaultdict

        # Group MODIFY_FILE tasks by target_path
        grouped: dict[str, list[Task]] = defaultdict(list)
        non_modify: dict[str, Task] = {}
        for tid, task in tasks.items():
            if task.type == TaskType.MODIFY_FILE and task.target_path:
                grouped[task.target_path].append(task)
            else:
                non_modify[tid] = task

        if not grouped:
            return tasks  # Nothing to consolidate

        consolidated: dict[str, Task] = dict(non_modify)
        for path, sub_tasks in grouped.items():
            if len(sub_tasks) == 1:
                consolidated[sub_tasks[0].id] = sub_tasks[0]
                continue

            # Merge multiple tasks for the same file into one
            combined_prompt_lines: list[str] = []
            all_deps: list[str] = []
            for i, st in enumerate(sub_tasks, 1):
                combined_prompt_lines.append(f"{i}. {st.prompt}")
                all_deps.extend(st.dependencies if isinstance(st.dependencies, list) else [])

            merged_id = f"consolidated_{path.replace('/', '_').replace('.', '_')}"
            # Use modification_strategy of the first task, fall back to "replace"
            strategy = getattr(sub_tasks[0], "modification_strategy", "replace") or "replace"

            merged = Task(
                id=merged_id,
                type=TaskType.MODIFY_FILE,
                prompt="\n".join(combined_prompt_lines),
                target_path=path,
                dependencies=list(set(all_deps)),  # deduplicate
                max_output_tokens=max(st.max_output_tokens for st in sub_tasks),
            )
            # Set dynamic attributes used by CodebaseWriter
            object.__setattr__(merged, "modification_strategy", strategy)

            logger.info(
                "Consolidated %d MODIFY_FILE tasks for %s into %s",
                len(sub_tasks),
                path,
                merged_id,
            )
            consolidated[merged_id] = merged

        return consolidated

    def _fallback_plan(self, objective: str, error: str | None = None) -> dict[str, Task]:
        """Fallback when LLM decomposition fails."""
        logger.warning("CodebaseDecomposer fallback: %s", error or "unknown error")
        return {
            "task_001": Task(
                id="task_001",
                type=TaskType.CODE_GEN,
                prompt=f"Implement the following: {objective}",
                dependencies=[],
                max_output_tokens=4096,
            )
        }
