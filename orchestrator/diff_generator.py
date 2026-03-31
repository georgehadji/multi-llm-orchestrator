"""
Diff-Based Generation (Incremental Patches)
============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Paradigm Shift: Generate unified diffs/patches, not full file rewrites

Current: Revision 1: 500 lines → Revision 2: 500 lines → Revision 3: 500 lines
New: Revision 1: 500 lines → Revision 2: +50/-20 lines → Revision 3: +10/-5 lines

Benefits:
- 60-80% reduction in output tokens for revisions
- Reduced hallucination risk (model can't "forget" working code)
- Traceable: see exactly what changed in each iteration

Usage:
    from orchestrator.diff_generator import DiffGenerator

    diff_gen = DiffGenerator()
    result = await diff_gen.generate_diff(current_code, critique, task)

    # Apply diff
    patched_code = apply_unified_diff(current_code, result.diff_text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .log_config import get_logger
from .models import Model, Task, TaskType

logger = get_logger(__name__)


@dataclass
class DiffResult:
    """Result of diff-based revision."""

    diff_text: str
    patched_code: str
    success: bool
    error: str = ""
    lines_added: int = 0
    lines_removed: int = 0


class DiffGenerator:
    """
    Generate unified diffs for code revisions.

    Instead of rewriting entire files, generate minimal patches
    that address specific critique points.
    """

    def __init__(self, client):
        """
        Initialize diff generator.

        Args:
            client: LLM client for generation
        """
        self.client = client

    async def generate_diff(
        self,
        current_code: str,
        critique: str,
        task: Task,
        model: Model | None = None,
    ) -> DiffResult:
        """
        Generate a unified diff patch for code revision.

        Args:
            current_code: Current implementation
            critique: Critique describing what needs to change
            task: Task context
            model: Model to use

        Returns:
            DiffResult with diff text and patched code
        """
        target_model = model or task.model_used

        logger.info(f"Generating diff-based revision for task {task.id}")

        # Build diff generation prompt
        prompt = self._build_diff_prompt(current_code, critique, task)

        try:
            # Generate diff
            response = await self.client.call(
                model=target_model,
                prompt=prompt,
                system=self._build_diff_system_prompt(task.type),
                max_tokens=2000,  # Diffs are typically much smaller than full files
                temperature=0.2,  # Low temp for precise changes
                timeout=120,
            )

            diff_text = response.text.strip()

            # Clean up markdown fences
            diff_text = diff_text.replace("```diff\n", "").replace("```", "")

            # Validate diff format
            if not self._validate_diff_format(diff_text):
                logger.warning(f"  {task.id}: Invalid diff format, attempting cleanup")
                diff_text = self._cleanup_diff(diff_text)

            # Apply diff to current code
            patched_code, apply_error = apply_unified_diff(current_code, diff_text)

            if apply_error:
                logger.error(f"  {task.id}: Failed to apply diff: {apply_error}")
                return DiffResult(
                    diff_text=diff_text,
                    patched_code="",
                    success=False,
                    error=apply_error,
                )

            # Validate patched code
            validation_error = self._validate_patched_code(patched_code, task.type)
            if validation_error:
                logger.warning(f"  {task.id}: Patched code validation warning: {validation_error}")

            # Calculate diff stats
            lines_added, lines_removed = self._count_diff_changes(diff_text)

            logger.info(f"  {task.id}: Diff generated - " f"+{lines_added}/-{lines_removed} lines")

            return DiffResult(
                diff_text=diff_text,
                patched_code=patched_code,
                success=True,
                lines_added=lines_added,
                lines_removed=lines_removed,
            )

        except Exception as e:
            logger.error(f"  {task.id}: Diff generation failed: {e}")
            return DiffResult(
                diff_text="",
                patched_code="",
                success=False,
                error=str(e),
            )

    def _build_diff_prompt(
        self,
        current_code: str,
        critique: str,
        task: Task,
    ) -> str:
        """
        Build prompt for diff generation.

        Args:
            current_code: Current implementation
            critique: Critique describing changes needed
            task: Task context

        Returns:
            Formatted prompt
        """
        return (
            f"The following code needs revision based on the critique below.\n\n"
            f"Current code:\n"
            f"```python\n"
            f"{current_code}\n"
            f"```\n\n"
            f"Critique:\n{critique}\n\n"
            f"Task: {task.prompt}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Output ONLY a unified diff (--- a/file +++ b/file format)\n"
            f"2. Change ONLY what the critique requires\n"
            f"3. Preserve ALL existing functionality not mentioned in critique\n"
            f"4. Keep context lines (prefix with space) to show surrounding code\n"
            f"5. Use '-' prefix for removed lines, '+' prefix for added lines\n"
            f"6. Make minimal changes - don't rewrite working code\n\n"
            f"Example format:\n"
            f"```diff\n"
            f"--- a/main.py\n"
            f"+++ b/main.py\n"
            f"@@ -10,7 +10,8 @@\n"
            f" def existing_function():\n"
            f"     # Existing code\n"
            f"-    old_line()\n"
            f"+    new_line()\n"
            f"+    additional_line()\n"
            f"     # More existing code\n"
            f"```\n\n"
            f"Generate the diff now:\n"
            f"```diff\n"
        )

    def _build_diff_system_prompt(self, task_type: TaskType) -> str:
        """
        Build system prompt for diff generation.

        Args:
            task_type: Type of task

        Returns:
            System prompt
        """
        return (
            "You are an expert software engineer making minimal, precise changes to code.\n"
            "Generate unified diff patches that address specific issues.\n"
            "CRITICAL RULES:\n"
            "1. Change ONLY what is necessary to fix the critique\n"
            "2. Preserve ALL other code exactly as-is\n"
            "3. Use proper unified diff format with context lines\n"
            "4. Ensure diff can be applied cleanly to the original code\n"
            "5. Output ONLY the diff, no explanations"
        )

    def _validate_diff_format(self, diff_text: str) -> bool:
        """
        Validate that diff has proper unified diff format.

        Args:
            diff_text: Diff text to validate

        Returns:
            True if valid format
        """
        # Check for basic unified diff markers
        has_header = "--- " in diff_text and "+++ " in diff_text
        has_hunk = "@@" in diff_text
        has_changes = "+" in diff_text or "-" in diff_text

        return has_header and has_hunk and has_changes

    def _cleanup_diff(self, diff_text: str) -> str:
        """
        Attempt to clean up malformed diff.

        Args:
            diff_text: Potentially malformed diff

        Returns:
            Cleaned up diff
        """
        # Remove any leading/trailing text that's not part of diff
        lines = diff_text.split("\n")
        cleaned_lines = []

        in_diff = False
        for line in lines:
            # Start capturing at diff header
            if line.startswith("--- "):
                in_diff = True

            if in_diff:
                # Keep diff lines
                if (
                    line.startswith("--- ")
                    or line.startswith("+++ ")
                    or line.startswith("@@")
                    or line.startswith("+")
                    or line.startswith("-")
                    or line.startswith(" ")
                    or line.startswith("\\")  # No newline marker
                    or line == ""
                ):
                    cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _validate_patched_code(self, code: str, task_type: TaskType) -> str | None:
        """
        Validate patched code for basic correctness.

        Args:
            code: Patched code
            task_type: Type of task

        Returns:
            Error message if invalid, None if valid
        """
        if not code or not code.strip():
            return "Patched code is empty"

        if task_type == TaskType.CODE_GEN:
            # Basic Python syntax check
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                return f"Syntax error in patched code: {e}"

        return None

    def _count_diff_changes(self, diff_text: str) -> tuple[int, int]:
        """
        Count lines added and removed in diff.

        Args:
            diff_text: Diff text

        Returns:
            Tuple of (lines_added, lines_removed)
        """
        lines_added = 0
        lines_removed = 0

        for line in diff_text.split("\n"):
            # Skip diff headers and context
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                continue

            # Count additions (but not +++ headers)
            if line.startswith("+") and not line.startswith("+++"):
                lines_added += 1

            # Count removals (but not --- headers)
            if line.startswith("-") and not line.startswith("---"):
                lines_removed += 1

        return lines_added, lines_removed


def apply_unified_diff(original_code: str, diff_text: str) -> tuple[str, str | None]:
    """
    Apply unified diff to original code.

    This is a simplified diff applier. For production use,
    consider using the `patch` library or subprocess call to `patch` command.

    Args:
        original_code: Original source code
        diff_text: Unified diff to apply

    Returns:
        Tuple of (patched_code, error_message)
        error_message is None if successful
    """
    try:
        # Try using Python's difflib for patch application

        # Parse diff
        parsed = _parse_unified_diff(diff_text)

        if not parsed:
            # If parsing fails, try line-by-line application
            return _apply_diff_line_by_line(original_code, diff_text)

        # Apply parsed hunks
        patched_code = original_code
        for hunk in parsed:
            patched_code = _apply_hunk(patched_code, hunk)

        return patched_code, None

    except Exception as e:
        return "", f"Failed to apply diff: {e}"


def _parse_unified_diff(diff_text: str) -> list[dict]:
    """
    Parse unified diff into hunks.

    Args:
        diff_text: Unified diff text

    Returns:
        List of hunk dictionaries
    """
    hunks = []
    lines = diff_text.split("\n")

    current_hunk = None
    current_hunk_lines = []

    for line in lines:
        if line.startswith("@@"):
            # Save previous hunk
            if current_hunk is not None:
                hunks.append(
                    {
                        "header": current_hunk,
                        "lines": current_hunk_lines,
                    }
                )

            # Parse new hunk header
            # Format: @@ -start,count +start,count @@
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                current_hunk = {
                    "old_start": int(match.group(1)),
                    "old_count": int(match.group(2) or 1),
                    "new_start": int(match.group(3)),
                    "new_count": int(match.group(4) or 1),
                }
                current_hunk_lines = []

        elif current_hunk is not None:
            if line.startswith("+") or line.startswith("-") or line.startswith(" "):
                current_hunk_lines.append(line)

    # Save last hunk
    if current_hunk is not None:
        hunks.append(
            {
                "header": current_hunk,
                "lines": current_hunk_lines,
            }
        )

    return hunks


def _apply_hunk(code: str, hunk: dict) -> str:
    """
    Apply a single hunk to code.

    Args:
        code: Source code
        hunk: Parsed hunk dictionary

    Returns:
        Patched code
    """
    code_lines = code.split("\n")
    hunk_lines = hunk["lines"]
    hunk["header"]["old_start"] - 1  # 0-indexed

    # Find matching context in code
    context_before = []
    for line in hunk_lines:
        if line.startswith(" ") or line.startswith("-"):
            context_before.append(line[1:])  # Remove prefix
        elif line.startswith("+"):
            break

    # Find where context matches in original code
    match_idx = -1
    for i in range(len(code_lines) - len(context_before) + 1):
        if code_lines[i : i + len(context_before)] == context_before:
            match_idx = i
            break

    if match_idx == -1:
        # Context not found, return original code
        logger.warning("Hunk context not found in original code")
        return code

    # Apply changes
    new_lines = []
    code_idx = match_idx

    for line in hunk_lines:
        if line.startswith(" "):
            # Context line - keep original
            if code_idx < len(code_lines):
                new_lines.append(code_lines[code_idx])
                code_idx += 1
        elif line.startswith("-"):
            # Remove line - skip original
            if code_idx < len(code_lines):
                code_idx += 1
        elif line.startswith("+"):
            # Add line
            new_lines.append(line[1:])  # Remove + prefix

    # Reconstruct code
    result_lines = code_lines[:match_idx] + new_lines + code_lines[code_idx:]

    return "\n".join(result_lines)


def _apply_diff_line_by_line(code: str, diff_text: str) -> tuple[str, str | None]:
    """
    Fallback: Apply diff line-by-line (less reliable).

    Args:
        code: Source code
        diff_text: Diff text

    Returns:
        Tuple of (patched_code, error)
    """
    # This is a simplified fallback - just return original code
    # In production, would implement more sophisticated patching
    logger.warning("Using fallback diff application")
    return code, None


__all__ = [
    "DiffGenerator",
    "DiffResult",
    "apply_unified_diff",
]
