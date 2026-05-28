"""
orchestrator/output/code_cleaner.py
====================================
Post-processing for LLM code output: removes markdown fences,
placeholder comments, and normalises whitespace.

Extracted from engine.py as a pure function (P3-7 of REFACTORING_PLAN_V7.md).
"""

from __future__ import annotations

import re

from ..models import TaskType

# Regex patterns for placeholder comments that LLMs commonly emit
_PLACEHOLDER_PATTERNS: list[str] = [
    r"//\s*[Aa]dd\s+(?:content|code|your|more|placeholder).*?\n",
    r"//\s*[Rr]eplace\s+this.*?(?:\n|$)",
    r"//\s*[Tt]ODO:.*?(?:\n|$)",
    r"//\s*[Ff]IXME:.*?(?:\n|$)",
    r"/\*\s*[Aa]dd\s+(?:content|code|your).*?\*/",
    r"/\*\s*[Rr]eplace\s+this.*?\*/",
    r"<!--\s*[Aa]dd\s+(?:content|code|your).*?-->",
    r"<!--\s*[Rr]eplace\s+this.*?-->",
    r"#\s*[Aa]dd\s+(?:content|code|your|more).*?(?:\n|$)",
    r"#\s*[Rr]eplace\s+this.*?(?:\n|$)",
]


def clean_code_output(text: str, task_type: TaskType) -> str:
    """Post-process code output to remove common LLM artifacts.

    Strips:
    - Markdown fences (```language...```)
    - Placeholder comments ("// Add content here", "# TODO: …")
    - Runs of 3+ blank lines (collapsed to 2)

    Non-CODE_GEN tasks are returned unchanged.

    Args:
        text: Raw LLM output.
        task_type: The task type that produced the output.

    Returns:
        Cleaned output, or the original text for non-code tasks.
    """
    if task_type != TaskType.CODE_GEN:
        return text

    # Remove markdown code fences
    text = re.sub(r"^```\w*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)

    # Remove placeholder / explanatory comment patterns
    for pattern in _PLACEHOLDER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
