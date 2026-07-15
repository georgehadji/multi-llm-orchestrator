"""
Python AST Verifier — checks syntactic validity via ``ast.parse``,
without executing the code.

Safe: no ``exec`` / ``eval``, no subprocess, no I/O.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import ClassVar

from orchestrator.models import TaskType, Verdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _extract_code_blocks(response: str) -> list[str]:
    """Extract Python code blocks from a response.

    Looks for fenced code blocks with python/py language tag,
    or standalone shebang scripts.  Falls back to the cleanest
    python-looking indented block when no fences are found.
    """
    blocks: list[str] = []

    # Fenced blocks: ```python ... ```
    pattern = re.compile(
        r"```(?:py|python)\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    blocks.extend(m.group(1).strip() for m in pattern.finditer(response))

    # Bare blocks (no language tag): ``` ... ```
    if not blocks:
        bare = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
        blocks.extend(m.group(1).strip() for m in bare.finditer(response))

    # Shebang scripts
    if not blocks:
        shebang = re.compile(r"^#!.*python.*\n((?:.+\n?)*)", re.MULTILINE)
        blocks.extend(m.group(1).strip() for m in shebang.finditer(response))

    return blocks


def _score_from_ast(tree: ast.AST, source: str) -> tuple[float, list[str]]:
    """Heuristic quality score based on AST structure, not execution.

    Bonus signals for:
    - Has a function definition  (+0.10)
    - Has a class definition      (+0.10)
    - Has a docstring             (+0.05)
    - Has type annotations        (+0.05)
    - Has an async function       (+0.05)
    - Has a ``if __name__`` guard (+0.05)
    - Lines of code > 10          (+0.05)
    """
    signals: list[str] = ["ast_valid"]
    score_bonus = 0.0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if "func_def" not in signals:
                signals.append("func_def")
                score_bonus += 0.10
            if node.returns is not None:
                if "type_annotation" not in signals:
                    signals.append("type_annotation")
                    score_bonus += 0.05
        if isinstance(node, ast.AsyncFunctionDef):
            if "async_func" not in signals:
                signals.append("async_func")
                score_bonus += 0.05
        if isinstance(node, ast.ClassDef):
            if "class_def" not in signals:
                signals.append("class_def")
                score_bonus += 0.10
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and "docstring" not in signals:
                signals.append("docstring")
                score_bonus += 0.05

    # Guard clause
    if any(isinstance(node, ast.If) and _is_name_main_guard(node) for node in ast.walk(tree)):
        signals.append("name_main_guard")
        score_bonus += 0.05

    # Length bonus
    lines = source.count("\n") + 1
    if lines > 10:
        score_bonus += 0.05
        signals.append("long_enough")

    score = min(1.0, 0.5 + score_bonus)
    return score, signals


def _is_name_main_guard(node: ast.If) -> bool:
    """Detect ``if __name__ == "__main__":`` idiom."""
    try:
        left = node.test.left  # type: ignore[union-attr]
        right = node.test.comparators[0]  # type: ignore[union-attr]
        return (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
        )
    except (AttributeError, IndexError):
        return False


# ─────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────


class PythonASTVerifier:
    """Verify Python code via AST parse (safe — no exec).

    Supports two modes:
    1. **Code-extraction mode** (default): extracts fenced code
       blocks from a prose response.
    2. **Raw mode**: assumes the entire response is Python source.

    Usage:
        verifier = PythonASTVerifier()
        verdict = await verifier.verify(prompt="Write a function", response="def foo(): pass", ...)
    """

    CODE_TASK_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "code_generation",
            "code_review",
        }
    )

    def __init__(self, *, extract_blocks: bool = True) -> None:
        """Initialize verifier.

        Args:
            extract_blocks: When True, extract code from fenced blocks.
                Set to False when the response *is* the code.
        """
        self._extract_blocks = extract_blocks

    async def verify(
        self,
        *,
        prompt: str,
        response: str,
        task_type: TaskType,
    ) -> Verdict:
        """Check Python AST validity of *response*."""
        # Only applicable for code tasks
        if task_type.value not in self.CODE_TASK_TYPES:
            return Verdict(
                passed=True,
                score=0.5,
                signals=("not_applicable",),
                detail=f"PythonASTVerifier skipped for {task_type.value}",
            )

        sources: list[str] = []
        if self._extract_blocks:
            sources = _extract_code_blocks(response)
        if not sources:
            # Try the whole response as Python
            sources = [response]

        all_signals: list[str] = []
        max_score = 0.0
        errors: list[str] = []

        for i, source in enumerate(sources):
            try:
                tree = ast.parse(source)
                score, sigs = _score_from_ast(tree, source)
                all_signals.extend(sigs)
                max_score = max(max_score, score)
            except SyntaxError as exc:
                msg = f"Block {i}: SyntaxError at line {exc.lineno}: {exc.msg}"
                errors.append(msg)
                all_signals.append("syntax_error")

        if errors and max_score == 0:
            # No valid block found
            return Verdict(
                passed=False,
                score=0.0,
                signals=tuple(all_signals),
                detail="; ".join(errors),
            )

        # At least some valid code
        passed = max_score >= 0.5  # any valid AST means "passed"
        return Verdict(
            passed=passed,
            score=max_score,
            signals=tuple(all_signals),
            detail="; ".join(errors) if errors else "AST parse OK",
        )
