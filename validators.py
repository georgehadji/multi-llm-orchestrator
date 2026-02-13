"""
Deterministic Validators
========================
Non-negotiable checks that override LLM-based scoring.
If deterministic check fails → score = 0.0 regardless of LLM evaluation.

Counterfactual: Without deterministic validators → vulnerability Ψ:
LLM evaluator gives high score to syntactically broken code or
structurally invalid data, causing cascading downstream failures.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orchestrator.validators")


class ValidationResult:
    __slots__ = ("passed", "details", "validator_name")

    def __init__(self, passed: bool, details: str = "", validator_name: str = ""):
        self.passed = passed
        self.details = details
        self.validator_name = validator_name


def validate_json_schema(output: str, schema: Optional[dict] = None) -> ValidationResult:
    """Validate that output is valid JSON, optionally against a schema."""
    try:
        parsed = json.loads(output)
        if schema:
            import jsonschema
            jsonschema.validate(instance=parsed, schema=schema)
        return ValidationResult(True, "Valid JSON", "json_schema")
    except json.JSONDecodeError as e:
        return ValidationResult(False, f"Invalid JSON: {e}", "json_schema")
    except Exception as e:
        return ValidationResult(False, f"Schema validation failed: {e}", "json_schema")


def validate_python_syntax(output: str) -> ValidationResult:
    """Check Python code compiles without syntax errors."""
    try:
        # Extract code from markdown fences if present
        code = _extract_code_block(output, "python")
        compile(code, "<orchestrator_check>", "exec")
        return ValidationResult(True, "Syntax OK", "python_syntax")
    except SyntaxError as e:
        return ValidationResult(False, f"Syntax error: {e}", "python_syntax")


def validate_pytest(output: str, test_code: str = "",
                    timeout: int = 30) -> ValidationResult:
    """
    Run pytest on generated code. Writes to temp dir, executes, checks exit code.
    Requires pytest installed in environment.
    """
    code = _extract_code_block(output, "python")
    if not code.strip():
        return ValidationResult(False, "No Python code found in output", "pytest")

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "generated_code.py"
        code_path.write_text(code)

        if test_code:
            test_path = Path(tmpdir) / "test_generated.py"
            test_content = f"from generated_code import *\n\n{test_code}"
            test_path.write_text(test_content)
        else:
            # If no explicit tests, at least check import
            test_path = Path(tmpdir) / "test_import.py"
            test_path.write_text(
                "def test_import():\n    import generated_code\n"
            )

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", str(tmpdir), "-v", "--tb=short"],
                capture_output=True, text=True, timeout=timeout,
                cwd=tmpdir,
            )
            if result.returncode == 0:
                return ValidationResult(True, result.stdout[-500:], "pytest")
            else:
                return ValidationResult(
                    False,
                    f"Tests failed (exit {result.returncode}):\n{result.stdout[-500:]}\n{result.stderr[-300:]}",
                    "pytest"
                )
        except subprocess.TimeoutExpired:
            return ValidationResult(False, f"Pytest timed out after {timeout}s", "pytest")
        except FileNotFoundError:
            return ValidationResult(False, "pytest not found in PATH", "pytest")


def validate_ruff(output: str, timeout: int = 15) -> ValidationResult:
    """Run ruff linter on generated Python code."""
    code = _extract_code_block(output, "python")
    if not code.strip():
        return ValidationResult(False, "No Python code found", "ruff")

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["ruff", "check", f.name, "--select=E,F"],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return ValidationResult(True, "No lint errors", "ruff")
            else:
                return ValidationResult(
                    False,
                    f"Lint issues:\n{result.stdout[-500:]}",
                    "ruff"
                )
        except FileNotFoundError:
            logger.warning("ruff not installed, skipping lint check")
            return ValidationResult(True, "ruff not available, skipped", "ruff")
        except subprocess.TimeoutExpired:
            return ValidationResult(False, "ruff timed out", "ruff")


def validate_latex(output: str, timeout: int = 30) -> ValidationResult:
    """Check LaTeX compiles (requires pdflatex)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "check.tex"
        tex_path.write_text(output)
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 str(tex_path)],
                capture_output=True, text=True, timeout=timeout,
                cwd=tmpdir,
            )
            if result.returncode == 0:
                return ValidationResult(True, "LaTeX compiled", "latex")
            else:
                return ValidationResult(
                    False,
                    f"LaTeX compilation failed:\n{result.stderr[-500:]}",
                    "latex"
                )
        except FileNotFoundError:
            return ValidationResult(True, "pdflatex not available, skipped", "latex")
        except subprocess.TimeoutExpired:
            return ValidationResult(False, "LaTeX compilation timed out", "latex")


def validate_length_bounds(output: str, min_chars: int = 10,
                           max_chars: int = 50000) -> ValidationResult:
    """Check output is within reasonable length bounds."""
    length = len(output.strip())
    if length < min_chars:
        return ValidationResult(
            False, f"Output too short ({length} < {min_chars})", "length"
        )
    if length > max_chars:
        return ValidationResult(
            False, f"Output too long ({length} > {max_chars})", "length"
        )
    return ValidationResult(True, f"Length OK ({length} chars)", "length")


# ─────────────────────────────────────────────
# Validator registry
# ─────────────────────────────────────────────

VALIDATORS = {
    "json_schema": validate_json_schema,
    "python_syntax": validate_python_syntax,
    "pytest": validate_pytest,
    "ruff": validate_ruff,
    "latex": validate_latex,
    "length": validate_length_bounds,
}


def run_validators(output: str, validator_names: list[str],
                   **kwargs) -> list[ValidationResult]:
    """Run all specified validators. Returns list of results."""
    results = []
    for name in validator_names:
        fn = VALIDATORS.get(name)
        if fn:
            try:
                result = fn(output, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(False, f"Validator crash: {e}", name))
        else:
            logger.warning(f"Unknown validator: {name}")
    return results


def all_validators_pass(results: list[ValidationResult]) -> bool:
    return all(r.passed for r in results)


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _extract_code_block(text: str, language: str = "python") -> str:
    """Extract first code block from markdown-fenced output."""
    import re
    # Try fenced block first
    pattern = rf"```{language}\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    # Try generic fence
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    # Assume entire output is code
    return text
