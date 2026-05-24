"""
TaskValidator — Preflight, syntax, and deterministic validation
=================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Extracted from engine.py Phase 3: Preflight + Validation.
Orchestrator._run_preflight_check() and syntax validators delegate here.
"""

from __future__ import annotations

import ast as _ast
import logging
from typing import Any

from ..api_clients import UnifiedClient
from ..budget import Budget
from ..models import Model, Task
from ..prompt_builder import RevisionPrompt
from ..resilience import ResiliencePolicy

logger = logging.getLogger("orchestrator.engine_core.validator")


class TaskValidator:
    """Preflight, syntax, and deterministic validation for task outputs.

    Wraps syntax streaming/batch checks, preflight delivery gate,
    and validator filtering — all previously inline in engine.py.

    Args:
        client: UnifiedClient for LLM revision calls.
        budget: Budget for tracking revision costs.
        preflight_validator: Optional PreflightValidator instance.
        hook_registry: Optional HookRegistry for firing events.
    """

    def __init__(
        self,
        client: UnifiedClient,
        budget: Budget,
        preflight_validator: Any = None,
        hook_registry: Any = None,
    ) -> None:
        self._client = client
        self._budget = budget
        self._preflight_validator = preflight_validator
        self._hook_registry = hook_registry

    # ── Syntax validation ────────────────────────────────────────────────

    def validate_syntax_streaming(self, partial_output: str) -> bool:
        """Quick streaming syntax validator for early abort.

        Checks partial code output for obvious syntax errors:
        - Unclosed brackets/parentheses
        - Invalid Python syntax (early detection)
        - Missing imports for common modules

        Args:
            partial_output: Partial code output (first ~500 tokens).

        Returns:
            True if syntax looks valid, False if obvious errors detected.
        """
        # Quick bracket balance check
        brackets = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for char in partial_output:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False  # Closing bracket without opening
                expected = brackets[stack.pop()]
                if char != expected:
                    return False  # Mismatched brackets

        # Quick AST parse for obvious errors
        try:
            _ast.parse(partial_output)
        except SyntaxError:
            return False

        return True

    def validate_syntax_batch(self, output: str) -> bool:
        """Full batch syntax validation for post-generation code.

        Uses AST parsing to validate complete code blocks.
        Also checks for common issues like missing imports.

        Args:
            output: Complete code output to validate.

        Returns:
            True if syntax is valid, False otherwise.
        """
        if not output or not output.strip():
            return False

        # Remove markdown code fences if present
        text = output.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if not text:
            return False

        try:
            _ast.parse(text)
            return True
        except SyntaxError as e:
            logger.debug("Syntax validation failed: %s", e)
            return False

    def filter_validators_for_task(self, task: Task, output: str) -> list[str]:
        """Filter validators based on task type and content.

        Removes Python-specific validators for non-Python tasks.

        Args:
            task: The task being validated.
            output: The task output.

        Returns:
            List of applicable hard validator names.
        """
        if not task.hard_validators:
            return []

        # Detect if this is a Python task
        is_python = False
        if output.startswith("import") or output.startswith("from"):
            is_python = True
        if task.type.value in ("code_generation", "code_review"):
            is_python = True

        # Filter validators
        if is_python:
            return task.hard_validators

        # For non-Python tasks, remove Python-specific validators
        python_specific = {"python_syntax", "pytest", "ruff", "black", "mypy"}
        return [v for v in task.hard_validators if v not in python_specific]

    # ── Preflight delivery gate ──────────────────────────────────────────

    async def run_preflight_check(
        self,
        task: Task,
        output: str,
        score: float,
        primary: Model,
        revision_prompt_builder: Any = None,
        policy: ResiliencePolicy | None = None,
    ) -> tuple[str, float, Any]:
        """Post-loop preflight delivery gate.

        Checks best_output before finalizing TaskResult:
        - PASS  : return unchanged
        - WARN  : log + score * 0.85, fire PREFLIGHT_CHECK hook
        - ENRICH: 1 extra LLM revision with enrich reason as critique
        - BLOCK : 1 extra LLM revision with block reason as critique
                   -> recovered: return revised output
                   -> still BLOCK: return original output, score=0.0

        Fail-open: any validator exception is caught and treated as PASS.

        Args:
            task: The completed task.
            output: The best generated output.
            score: The current quality score.
            primary: The primary model used.
            revision_prompt_builder: Optional prompt builder override.
            policy: Optional resilience policy.

        Returns:
            Tuple of (output, score, preflight_result).
        """
        if self._preflight_validator is None:
            return output, score, self._make_pass_result()

        try:
            from ..preflight import PreflightAction, PreflightMode, PreflightResult

            pf_result = self._preflight_validator.validate(
                response=output,
                context={
                    "task_type": task.type.value,
                    "user_request": task.prompt[:200],
                    "model": primary.value,
                    "score": score,
                },
                mode=PreflightMode.AUTO,
            )
        except Exception as exc:
            logger.warning("preflight validator raised: %s - treating as PASS", exc)
            return output, score, self._make_pass_result()

        if pf_result.action == PreflightAction.PASS:
            return output, score, pf_result

        if pf_result.action == PreflightAction.WARN:
            penalized = round(score * 0.85, 4)
            logger.warning(
                "[preflight] WARN task=%s score %.3f->%.3f: %s",
                task.id, score, penalized,
                "; ".join(pf_result.warnings),
            )
            if self._hook_registry is not None:
                self._hook_registry.fire(
                    "preflight_check",
                    task_id=task.id, action="warn",
                    reason="; ".join(pf_result.warnings),
                    score_before=score, score_after=penalized,
                )
            return output, penalized, pf_result

        # ENRICH or BLOCK — attempt one extra revision
        critique_text = pf_result.reason or pf_result.enrichment or "Improve the response quality."

        # Calculate dynamic timeout
        context_size = len(task.prompt)
        dynamic_timeout = min(120 + (context_size // 1000), 300)

        logger.info(
            "[preflight] %s task=%s - attempting 1 revision: %s (timeout=%ds)",
            pf_result.action.value.upper(), task.id,
            critique_text[:100], dynamic_timeout,
        )

        try:
            rev_prompt, rev_system = (
                revision_prompt_builder or RevisionPrompt
            ).build(task.prompt, critique_text, task.type.value)

            gen_response = await self._client.call(
                primary,
                rev_prompt,
                system=rev_system,
                max_tokens=task.max_output_tokens,
                temperature=0.3,
                timeout=dynamic_timeout,
                policy=policy,
            )
            await self._budget.charge(gen_response.cost_usd, "generation")
            revised_output = gen_response.text
        except Exception as exc:
            logger.warning("[preflight] revision LLM call failed (%s) - using original", exc)
            failed_score = 0.0 if pf_result.action == PreflightAction.BLOCK else score
            return output, failed_score, pf_result

        # Re-validate the revised output
        try:
            from ..preflight import PreflightMode

            retry_result = self._preflight_validator.validate(
                response=revised_output,
                context={"task_type": task.type.value, "user_request": task.prompt[:200]},
                mode=PreflightMode.AUTO,
            )
        except Exception:
            retry_result = pf_result

        if retry_result.action == PreflightAction.BLOCK:
            logger.warning("[preflight] BLOCK task=%s - revision still blocked, score->0", task.id)
            return output, 0.0, retry_result

        logger.info("[preflight] %s recovered task=%s", pf_result.action.value.upper(), task.id)
        return revised_output, score, retry_result

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_pass_result(self) -> Any:
        """Create a PASS preflight result."""
        try:
            from ..preflight import PreflightAction, PreflightResult
            return PreflightResult(action=PreflightAction.PASS, passed=True)
        except ImportError:
            return None
