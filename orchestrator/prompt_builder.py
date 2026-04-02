"""
orchestrator/prompt_builder.py
──────────────────────────────
Single source of truth for all LLM prompt templates used by the orchestrator.

Rules:
  - No I/O, no asyncio, no engine imports.
  - All public methods are static — no instance state.
  - Returns plain strings or (user_prompt, system_prompt) tuples.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.models import AttemptRecord


class DecompositionPrompt:
    """Builds the project decomposition prompt sent to the decomposition model."""

    @staticmethod
    def build(
        project: str,
        criteria: str,
        app_context_block: str,
        valid_types: list[str],
    ) -> str:
        return (
            f"You are a project decomposition engine. Break this project into\n"
            f"atomic, executable tasks.\n"
            f"\n"
            f"PROJECT: {project}\n"
            f"\n"
            f"SUCCESS CRITERIA: {criteria}\n"
            f"{app_context_block}"
            f"Return ONLY a JSON array. Each element must have:\n"
            f'- "id": string (e.g., "task_001")\n'
            f"- \"type\": one of {valid_types}\n"
            f'- "prompt": detailed instruction for the task executor. For code_generation tasks, MUST include:\n'
            f"  - Code must be THOROUGHLY COMMENTED\n"
            f"  - EVERY file MUST start with: /** Author: Georgios-Chrysovalantis Chatzivantsidis */\n"
            f'- "dependencies": list of task id strings this depends on (empty if none)\n'
            f'- "hard_validators": list of validator names \u2014 ONLY use these for code tasks:\n'
            f'  - "python_syntax": only for code_generation tasks that produce Python code\n'
            f'  - "json_schema": only for tasks that must return valid JSON\n'
            f'  - "pytest": only for code_generation tasks with runnable tests\n'
            f'  - "ruff": only for code_generation tasks requiring lint checks\n'
            f'  - "latex": only for tasks producing LaTeX documents\n'
            f'  - "length": for tasks requiring minimum/maximum output length\n'
            f"  - Use [] (empty list) for non-code tasks (reasoning, writing, analysis, evaluation)\n"
            f"\n"
            f"RULES:\n"
            f"- Tasks must be atomic (one clear deliverable each)\n"
            f"- Dependencies must form a DAG (no cycles)\n"
            f"- Include code_review tasks after code_generation tasks\n"
            f"- Include at least one evaluation task at the end\n"
            f"- 5-15 tasks total for a medium project\n"
            f"- Do NOT add hard_validators to reasoning, writing, analysis, or evaluation tasks\n"
            f"\n"
            f"Return ONLY the JSON array, no markdown fences, no explanation."
        )


class SystemPrompt:
    """Builds the system prompt for task execution, varying by quality mode."""

    @staticmethod
    def build(task_type: str = "", mode: str = "standard") -> str:
        if mode == "production":
            return SystemPrompt._production(task_type)
        return SystemPrompt._standard()

    @staticmethod
    def _standard() -> str:
        return (
            "You are an expert software engineer executing a task. "
            "Produce high-quality, complete output. "
            "Follow best practices and ensure all code is valid and runnable."
        )

    @staticmethod
    def _production(task_type: str = "") -> str:
        base = (
            "You are a senior software engineer delivering production-grade output. "
            "Requirements:\n"
            "1. Full type annotations on every function and class.\n"
            "2. Comprehensive error handling and input validation.\n"
            "3. Unit tests for every public function (pytest style).\n"
            "4. Docstrings on every module, class, and public function.\n"
            "5. Logging via the standard library logger (not print).\n"
            "6. No TODOs, no placeholder implementations.\n"
            "7. Follow SOLID principles and keep cyclomatic complexity \u2264 10.\n"
            "8. Include a brief inline comment for any non-obvious logic.\n"
        )
        if task_type in ("code_gen", "code_generation"):
            base += (
                "9. Return ONLY raw code \u2014 no markdown fences, no prose outside code.\n"
                "10. Code must pass mypy --strict.\n"
            )
        return base


class DeltaPrompt:
    """
    Builds the enriched retry prompt after a failed validation attempt.

    Security: XML-style tags delimit the feedback block so adversarial LLM
    output cannot escape it with injected sentinels or tag sequences.
    All user-controlled fields are sanitised before embedding.
    """

    @staticmethod
    def build(original_prompt: str, record: "AttemptRecord") -> str:
        def _sanitize(text: str) -> str:
            """Strip XML delimiters and the plain sentinel from user-supplied data."""
            text = text.replace("<ORCHESTRATOR_FEEDBACK>", "")
            text = text.replace("</ORCHESTRATOR_FEEDBACK>", "")
            text = text.replace("PREVIOUS ATTEMPT FAILED:", "[PREVIOUS ATTEMPT]:")
            return text

        safe_reason = _sanitize(record.failure_reason)
        safe_validators = [_sanitize(v) for v in record.validators_failed]
        validators_str = ", ".join(safe_validators) if safe_validators else "none"

        snippet_section = ""
        if record.output_snippet:
            safe_snippet = _sanitize(record.output_snippet)
            snippet_section = f"\n- Output snippet: {safe_snippet}"

        additional_guidance = ""
        if "F821" in record.failure_reason or "Undefined name" in record.failure_reason:
            additional_guidance = (
                "\n\n\u26a0\ufe0f IMPORT ERROR DETECTED: You used a name without importing it first.\n"
                "FIX: Add the required import statement at the TOP of your code.\n"
                "Example: 'from nba_api.stats.endpoints import playerdashboardbyyearoveryear'\n"
                "Example: 'from requests import RequestException'\n"
                "Check ALL function/class names used and ensure they are imported."
            )
        elif "F401" in record.failure_reason or "imported but unused" in record.failure_reason:
            additional_guidance = (
                "\n\n\u26a0\ufe0f UNUSED IMPORT DETECTED: Remove imports you don't use.\n"
                "FIX: Either remove the unused import OR use the imported name in your code."
            )
        elif "E402" in record.failure_reason or "import not at top" in record.failure_reason:
            additional_guidance = (
                "\n\n\u26a0\ufe0f IMPORT POSITION ERROR: Move all imports to the TOP of the file.\n"
                "FIX: Place all import statements before any code (functions, classes, etc.)."
            )
        elif (
            "unterminated triple-quoted string" in record.failure_reason
            or "Syntax error" in record.failure_reason
        ):
            additional_guidance = (
                "\n\n\u26a0\ufe0f SYNTAX ERROR DETECTED: Unclosed string literal or code structure issue.\n"
                "FIX:\n"
                '1. Check ALL triple-quoted strings ("""...""") are properly CLOSED\n'
                "2. Ensure all parentheses (), brackets [], and braces {} are matched\n"
                "3. Verify all string literals have matching opening and closing quotes\n"
                "4. Check that if/else/for/while blocks have proper indentation\n"
                "5. Run your code through a Python syntax checker BEFORE submitting\n"
                '\nCRITICAL: Every opening """ must have a closing """ on a later line!'
            )
        elif "invalid-syntax" in record.failure_reason:
            additional_guidance = (
                "\n\n\u26a0\ufe0f SYNTAX ERROR DETECTED: Invalid Python code structure.\n"
                "FIX:\n"
                "1. Check for missing colons after if/for/while/def/class statements\n"
                "2. Ensure proper indentation (use spaces, not tabs)\n"
                "3. Verify all parentheses, brackets, and braces are properly matched\n"
                "4. Check that string literals are properly quoted and closed"
            )

        return (
            f"{original_prompt}\n\n"
            f"<ORCHESTRATOR_FEEDBACK>\n"
            f"PREVIOUS ATTEMPT FAILED:\n"
            f"- Attempt: {record.attempt_num}\n"
            f"- Model: {record.model_used}\n"
            f"- Reason: {safe_reason}\n"
            f"- Validators failed: {validators_str}"
            f"{snippet_section}"
            f"{additional_guidance}\n\n"
            f"Please correct specifically: {safe_reason}\n"
            f"</ORCHESTRATOR_FEEDBACK>"
        )


class CritiquePrompt:
    """Builds critique/review prompts for cross-model review and quality scoring."""

    @staticmethod
    def build(task_prompt: str, output: str) -> tuple[str, str]:
        """
        Cross-model review prompt (engine.py usage).

        Returns:
            (user_prompt, system_prompt) tuple.
        """
        user_prompt = (
            f"Review this output for correctness, completeness, and quality. "
            f"Be specific about flaws and suggest concrete improvements.\n\n"
            f"ORIGINAL TASK: {task_prompt}\n\n"
            f"OUTPUT TO REVIEW:\n{output}"
        )
        system_prompt = "You are a critical reviewer. Find flaws, be specific."
        return user_prompt, system_prompt

    @staticmethod
    def build_score(
        task_prompt: str,
        output: str,
        task_type_value: str = "",
    ) -> str:
        """
        Scoring critique prompt (engine_core/critique_cycle.py usage).
        Returns a user prompt string; caller supplies the system prompt.
        """
        if task_type_value == "code_review":
            return (
                f"Review the following code for quality, correctness, and best practices.\n"
                f'Provide a score from 0.0 to 1.0 in JSON format: {{"score": 0.85, "reasoning": "..."}}\n\n'
                f"Original requirement:\n{task_prompt}\n\n"
                f"Code to review:\n```\n{output}\n```"
            )
        return (
            f"Review the following output for quality and correctness.\n"
            f'Provide a score from 0.0 to 1.0 in JSON format: {{"score": 0.85, "reasoning": "..."}}\n\n'
            f"Original prompt:\n{task_prompt}\n\n"
            f"Generated output:\n```\n{output}\n```"
        )


class RevisionPrompt:
    """Builds the revision prompt when a critique requires rework."""

    @staticmethod
    def build(
        task_prompt: str,
        critique_text: str,
        task_type_value: str = "",
    ) -> tuple[str, str]:
        """
        Returns:
            (user_prompt, system_prompt) tuple.
        """
        user_prompt = (
            f"{task_prompt}\n\n"
            f"[Revision required] {critique_text}\n"
            f"Please revise your previous response to address the above."
        )
        if task_type_value:
            system_prompt = (
                f"You are an expert executing a {task_type_value} task. "
                f"Produce high-quality, complete output."
            )
        else:
            system_prompt = "You are an expert. Produce high-quality, complete output."
        return user_prompt, system_prompt
