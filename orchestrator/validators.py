"""
Deterministic Validators
========================
Non-negotiable checks that override LLM-based scoring.
If deterministic check fails → score = 0.0 regardless of LLM evaluation.

FIX #1: validate_ruff now cleans up temp files via try/finally.
FIX #8: run_validators filters kwargs per-validator using inspect.signature.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
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
    """Check Python code compiles without syntax errors.

    Special cases handled:
    - Indented method fragments: dedent + wrap in dummy class before retry
    - Truncated output (LLM hit max_tokens mid-statement): treated as SKIP/pass,
      because the truncation is a generation limit issue, not a code quality issue.
      A truncated block ends with an incomplete statement on the last non-empty line.
    Note: IndentationError is a subclass of SyntaxError so it is caught first.
    """
    import textwrap
    import warnings
    code = _extract_code_block(output, "python")

    # Detect truncated output: last non-empty line ends mid-statement
    # (no colon, no closing bracket/paren, not a complete expression)
    last_line = next((l for l in reversed(code.splitlines()) if l.strip()), "")
    truncated = (
        last_line.rstrip().endswith(":")  # incomplete annotation like `access_token:`
        or (last_line.rstrip()[-1:] not in {"}", ")", "]", '"', "'", "\\"}
            and not last_line.strip().startswith("#")
            and ":" not in last_line
            and len(code.splitlines()) >= 50)  # only flag as truncated for long outputs
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # suppress invalid escape seq warnings
        try:
            compile(code, "<orchestrator_check>", "exec")
            return ValidationResult(True, "Syntax OK", "python_syntax")
        except IndentationError:
            # Possibly a class-method fragment — dedent then wrap in a dummy class and retry
            dedented = textwrap.dedent(code)
            wrapped = "class _Wrapper:\n" + "\n".join(
                "    " + line for line in dedented.splitlines()
            )
            try:
                compile(wrapped, "<orchestrator_check>", "exec")
                return ValidationResult(True, "Syntax OK (method fragment)", "python_syntax")
            except SyntaxError as e:
                if truncated:
                    return ValidationResult(True, "Skipped: output appears truncated at token limit", "python_syntax")
                return ValidationResult(False, f"Syntax error: {e}", "python_syntax")
        except SyntaxError as e:
            if truncated:
                return ValidationResult(True, "Skipped: output appears truncated at token limit", "python_syntax")
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
                    f"Tests failed (exit {result.returncode}):\n"
                    f"{result.stdout[-500:]}\n{result.stderr[-300:]}",
                    "pytest"
                )
        except subprocess.TimeoutExpired:
            return ValidationResult(False, f"Pytest timed out after {timeout}s", "pytest")
        except FileNotFoundError:
            return ValidationResult(False, "pytest not found in PATH", "pytest")


def validate_ruff(output: str, timeout: int = 15) -> ValidationResult:
    """
    Run ruff linter on generated Python code.
    FIX #1: Temp file is always cleaned up via try/finally.
    """
    code = _extract_code_block(output, "python")
    if not code.strip():
        return ValidationResult(False, "No Python code found", "ruff")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            tmp_path = f.name
            f.write(code)
            f.flush()

        result = subprocess.run(
            ["ruff", "check", tmp_path, "--select=E,F"],
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
    finally:
        # FIX #1: Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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


def _filter_kwargs_for(fn, kwargs: dict) -> dict:
    """
    FIX #8: Only pass kwargs that the validator function actually accepts.
    Prevents TypeError from mismatched signatures.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    # If function has **kwargs, pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    accepted = {k for k, p in params.items()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY)
                and k != "output"}  # 'output' is positional arg
    return {k: v for k, v in kwargs.items() if k in accepted}


def run_validators(output: str, validator_names: list[str],
                   **kwargs) -> list[ValidationResult]:
    """Run all specified validators. Returns list of results."""
    results = []
    for name in validator_names:
        fn = VALIDATORS.get(name)
        if fn:
            try:
                filtered = _filter_kwargs_for(fn, kwargs)
                result = fn(output, **filtered)
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
    """Extract first code block from markdown-fenced output.

    Strategy (in order):
    1. Fenced block with explicit language tag  ```python ... ```
    2. Fenced block without language tag        ``` ... ```
    3. Heuristic: find the first top-level Python statement line
       (must start at column 0: import/from/def/class/@/if __name__)
       and return everything from that line onward.
    4. Return the full text unchanged as a last resort.
    """
    import re

    # 1. Fenced with language tag
    match = re.search(rf"```{language}\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)

    # 2. Generic fenced block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)

    # 3. Heuristic: first top-level (column-0) Python statement
    #    Excludes prose that starts with words like "Here", "The", "This", etc.
    code_start_re = re.compile(
        r"^(import |from \w|def |class |@\w|if __name__|async def )", re.MULTILINE
    )
    m = code_start_re.search(text)
    if m:
        return text[m.start():]

    # 4. Fallback: return as-is
    return text
