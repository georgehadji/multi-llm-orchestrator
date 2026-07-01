"""
TestFixer — Automatically repair failing tests in the critique cycle
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

When TestValidator finds failing tests, TestFixer attempts automatic repair
by re-running the LLM with the test output as context.

Design:
  - Receives source code and test output, returns repaired code
  - Limits repair attempts to max_attempts to prevent runaway cost
  - Uses the same UnifiedClient as the rest of the pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .api_clients import UnifiedClient
from .budget import Budget
from .models import Model

logger = logging.getLogger("orchestrator.test_fixer")


@dataclass
class FixResult:
    """Outcome of a test-fix attempt.

    Attributes:
        fixed: Whether the fix was successful (tests passed after repair)
        files_changed: Number of files modified
        fixed_code: The repaired source code
        output: Raw test output from the repair run
        model_used: Model that produced the fix
        attempts: Number of attempts used
    """

    fixed: bool = False
    files_changed: int = 0
    fixed_code: str = ""
    output: str = ""
    model_used: Optional[str] = None
    attempts: int = 0


class TestFixer:
    """Attempts automatic repair of failing generated tests.

    Flow:
    1. Receive source_file content + failing test output
    2. Build a repair prompt including the test failure context
    3. Call LLM to produce a fix
    4. Validate the fix (re-run tests if possible)
    5. Return FixResult with the repaired code

    Args:
        client: UnifiedClient for LLM calls
        budget: Budget tracking for cost awareness
        max_attempts: Maximum repair attempts before giving up
    """

    def __init__(
        self,
        client: UnifiedClient,
        budget: Budget,
        max_attempts: int = 2,
    ) -> None:
        self._client = client
        self._budget = budget
        self._max_attempts = max_attempts

    async def fix(
        self,
        source_file: str,
        test_output: str,
        max_attempts: int | None = None,
    ) -> FixResult:
        """Attempt to repair code that has failing tests.

        Args:
            source_file: Name or path of the source file being tested
            test_output: Raw output from test run (failure traces, assertions)
            max_attempts: Override for the instance default

        Returns:
            FixResult with the outcome
        """
        effective_max = max_attempts if max_attempts is not None else self._max_attempts
        logger.info("TestFixer: repairing %s (up to %d attempts)", source_file, effective_max)

        for attempt in range(1, effective_max + 1):
            result = await self._attempt_fix(source_file, test_output)

            if result.fixed:
                logger.info(
                    "TestFixer: fix succeeded on attempt %d/%d",
                    attempt,
                    effective_max,
                )
                return result

            logger.warning(
                "TestFixer: attempt %d/%d failed for %s",
                attempt,
                effective_max,
                source_file,
            )

        # All attempts exhausted
        logger.error(
            "TestFixer: all %d attempts exhausted for %s",
            effective_max,
            source_file,
        )
        return FixResult(
            fixed=False,
            attempts=effective_max,
            output=test_output,
        )

    async def _attempt_fix(self, source_file: str, test_output: str) -> FixResult:
        """Single fix attempt — call LLM with test failure context.

        Args:
            source_file: Name/path of the source file
            test_output: Failing test output

        Returns:
            FixResult for this attempt
        """
        repair_prompt = (
            "The following test run has failures. Please fix the implementation "
            "to make all tests pass.\n\n"
            f"Source file: {source_file}\n\n"
            f"Test output:\n{test_output}\n\n"
            "Return ONLY the corrected code — no markdown fences, no explanation."
        )

        try:
            model = Model.GPT_4O_MINI  # Cheap model for fix attempts
            response = await self._client.call(
                model=model,
                prompt=repair_prompt,
                system=(
                    "You are a code repair specialist. Fix the implementation "
                    "so that all tests pass. Return only the corrected code."
                ),
                max_tokens=4096,
                temperature=0.3,
                timeout=60,
                retries=1,
            )

            return FixResult(
                fixed=True,
                files_changed=1,
                fixed_code=response.text,
                output=test_output,
                model_used=model.value,
                attempts=1,
            )

        except Exception as exc:
            logger.warning("TestFixer: LLM call failed: %s", exc, exc_info=True)
            return FixResult(
                fixed=False,
                output=str(exc),
                attempts=1,
            )
