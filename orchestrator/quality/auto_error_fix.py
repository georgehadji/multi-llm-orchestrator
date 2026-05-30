"""
AutoErrorFix — Diagnose, fix, and validate errors in generated code.
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 1 (Wave 3: V5 Auto-Error Fix Button).
Wraps the existing `deployment_feedback.py` diagnosis and
`validators.py` validation into a single "fix this error" loop.

Usage:
    fixer = AutoErrorFixer(client)
    result = await fixer.diagnose_and_fix(
        code="def broken( pass",
        error_output="SyntaxError: invalid syntax",
        task=task,
    )
    if result.fixed:
        print(f"Fixed: {result.fixed_code}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # FIXED: from ..models import Task
    from ..models import Task

    # FIXED: from ..infrastructure.llm_client import UnifiedClient
    from ...infrastructure.llm_client import UnifiedClient

logger = logging.getLogger(__name__)


_FIX_PROMPT = """You are a senior debugging engineer. Fix the following code error.

ORIGINAL CODE:
```python
{code}
```

ERROR OUTPUT:
{error_output}

TASK CONTEXT:
Prompt: {task_prompt}
Target: {task_target}

Instructions:
1. Identify the root cause of the error
2. Produce the minimal fix (only change what's broken)
3. Return the COMPLETE fixed code, not just the diff

Output exactly this JSON:
{{"root_cause": "explanation", "fix_description": "what changed", "fixed_code": "full fixed code here"}}"""


@dataclass
class FixResult:
    """Result of an auto-fix attempt."""

    fixed: bool
    original_code: str
    fixed_code: str
    root_cause: str = ""
    fix_description: str = ""
    attempts: int = 0
    validation_passed: bool = False


class AutoErrorFixer:
    """Diagnoses errors in generated code and produces fixes.

    Wraps the existing validation pipeline with a diagnose→fix→validate loop.
    """

    def __init__(self, client: UnifiedClient | None = None):
        self._client = client

    async def diagnose_and_fix(
        self,
        code: str,
        error_output: str,
        task: Task | None = None,
        max_attempts: int = 2,
    ) -> FixResult:
        """Diagnose an error and attempt to fix it.

        Args:
            code: The original code that produced the error
            error_output: The error message / stack trace
            task: Optional task context
            max_attempts: Maximum fix attempts

        Returns:
            FixResult with the (possibly fixed) code
        """
        task_prompt = getattr(task, "prompt", "unknown") if task else "unknown"
        task_target = getattr(task, "target_path", "unknown") if task else "unknown"

        for attempt in range(1, max_attempts + 1):
            try:
                result = await self._attempt_fix(code, error_output, task_prompt, task_target)
                if result.fixed:
                    logger.info(
                        f"Auto-fix succeeded on attempt {attempt}/{max_attempts}: "
                        f"{result.fix_description}"
                    )
                    result.attempts = attempt
                    return result
            except Exception as exc:
                logger.warning(f"Auto-fix attempt {attempt} failed: {exc}")

        # All attempts failed — return original code
        logger.warning(f"Auto-fix failed after {max_attempts} attempts")
        return FixResult(
            fixed=False,
            original_code=code,
            fixed_code=code,
            root_cause="Unable to fix automatically",
            attempts=max_attempts,
        )

    async def _attempt_fix(
        self,
        code: str,
        error_output: str,
        task_prompt: str,
        task_target: str,
    ) -> FixResult:
        """Make a single fix attempt."""
        if not self._client:
            return FixResult(
                fixed=False,
                original_code=code,
                fixed_code=code,
                root_cause="No LLM client available for auto-fix",
            )

        import json

        prompt = _FIX_PROMPT.format(
            code=code,
            error_output=error_output[:2000],
            task_prompt=task_prompt,
            task_target=task_target,
        )

        response = await self._client.call(
            model=None,
            prompt=prompt,
            system="You are a precise debugging engineer. Return only valid JSON.",
            max_tokens=2000,
            temperature=0.2,
            timeout=60,
        )

        parsed = self._parse_fix_response(response.text)
        fixed_code = parsed.get("fixed_code", code)

        # Validate the fix
        validation_passed = await self._validate_fix(fixed_code)

        return FixResult(
            fixed=validation_passed,
            original_code=code,
            fixed_code=fixed_code,
            root_cause=parsed.get("root_cause", ""),
            fix_description=parsed.get("fix_description", ""),
            validation_passed=validation_passed,
        )

    @staticmethod
    def _parse_fix_response(text: str) -> dict:
        """Parse the fix response from the LLM."""
        import json
        import re

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {
            "root_cause": "Unable to parse fix response",
            "fix_description": "",
            "fixed_code": "",
        }

    @staticmethod
    async def _validate_fix(code: str) -> bool:
        """Validate that the fixed code is syntactically correct."""
        import ast

        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def extract_error_from_output(output: str) -> str:
        """Extract the most relevant error from validation output.

        Given a full pytest/validation output, returns the key error message.
        """
        import re

        # Try to find a traceback or error line
        patterns = [
            r"(SyntaxError:.*?)(?:\n|$)",
            r"(TypeError:.*?)(?:\n|$)",
            r"(ValueError:.*?)(?:\n|$)",
            r"(AssertionError:.*?)(?:\n|$)",
            r"(FAILED.*?)(?:\n|$)",
            r"(E\s+.*?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1).strip()

        # Fallback: first 500 chars
        return output[:500].strip()
