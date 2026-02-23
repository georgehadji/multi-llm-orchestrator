"""
Deterministic Validators
========================
Author: Georgios-Chrysovalantis Chatzivantsidis
Non-negotiable checks that override LLM-based scoring.
If deterministic check fails → score = 0.0 regardless of LLM evaluation.

FIX #1: validate_ruff now cleans up temp files via try/finally.
FIX #8: run_validators filters kwargs per-validator using inspect.signature.
FEAT:   async_run_validators() offloads all subprocess validators to threads
        so the event loop is never blocked (pytest, ruff, latex).
"""

from __future__ import annotations

import asyncio
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
    - Truncated output (LLM hit max_tokens mid-statement): treated as FAIL so the
      orchestrator retries with a higher token budget. A truncated block ends with
      an incomplete statement on the last non-empty line.
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

    # Truncation check before syntax parsing: fail immediately so the engine
    # knows to retry with a higher max_output_tokens rather than silently passing
    # incomplete code through to execution/tests.
    if truncated:
        return ValidationResult(
            False,
            "Output appears truncated at token limit — retry with higher max_output_tokens",
            "python_syntax",
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
                return ValidationResult(False, f"Syntax error: {e}", "python_syntax")
        except SyntaxError as e:
            return ValidationResult(False, f"Syntax error: {e}", "python_syntax")


def validate_pytest(output: str, test_code: str = "",
                    timeout: int = 30) -> ValidationResult:
    """
    Run pytest on generated code. Writes to temp dir, executes, checks exit code.
    Requires pytest installed in environment.
    For async callers use async_run_validators() which offloads via asyncio.to_thread().

    When no explicit test_code is provided, the default "import generated_code" test
    is skipped for files with third-party top-level imports that are not installed
    in the orchestrator environment (e.g. fastapi, grpc, jwt).  In that case only
    python_syntax validation is meaningful; pytest would always fail with
    ModuleNotFoundError regardless of code quality, producing false negatives.
    """
    code = _extract_code_block(output, "python")
    if not code.strip():
        return ValidationResult(False, "No Python code found in output", "pytest")

    def _has_unavailable_imports(src: str) -> bool:
        """
        Return True if the code has top-level imports of packages that are not
        installed in the current Python environment.  Only top-level import and
        from-import statements are checked (not conditional or inline imports).
        """
        import importlib.util
        import re as _re
        # Match 'import X' and 'from X import Y' at column 0
        pattern = _re.compile(
            r"^(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", _re.MULTILINE
        )
        for m in pattern.finditer(src):
            top_pkg = m.group(1)
            spec = importlib.util.find_spec(top_pkg)
            if spec is None:
                return True
        return False

    def _run_sync() -> ValidationResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Always write with utf-8: LLM output may contain non-ASCII chars
            # that fail on Windows cp1252 default encoding.
            code_path = Path(tmpdir) / "generated_code.py"
            code_path.write_text(code, encoding="utf-8")

            if test_code:
                test_path = Path(tmpdir) / "test_generated.py"
                test_content = f"from generated_code import *\n\n{test_code}"
                test_path.write_text(test_content, encoding="utf-8")
            else:
                # Default: import-only smoke test.
                # Skip if the code has third-party imports not available here —
                # such a test would always fail with ModuleNotFoundError, not
                # because the code is wrong but because the environment is
                # isolated. python_syntax already covers structural correctness.
                if _has_unavailable_imports(code):
                    return ValidationResult(
                        True,
                        "Import smoke test skipped: code uses third-party packages "
                        "not installed in the validator environment — "
                        "python_syntax already verified structural correctness.",
                        "pytest",
                    )
                test_path = Path(tmpdir) / "test_import.py"
                test_path.write_text(
                    "# -*- coding: utf-8 -*-\n"
                    "def test_import():\n    import generated_code\n",
                    encoding="utf-8",
                )

            # Set PYTHONIOENCODING so pytest's own output is UTF-8 on Windows
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"  # Python 3.7+ UTF-8 mode

            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", str(tmpdir), "-v", "--tb=short"],
                    capture_output=True, text=True, timeout=timeout,
                    cwd=tmpdir, env=env,
                    encoding="utf-8", errors="replace",
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

    return _run_sync()


def validate_ruff(output: str, timeout: int = 15) -> ValidationResult:
    """
    Run ruff linter on generated Python code.
    FIX #1: Temp file is always cleaned up via try/finally.
    For async callers use async_run_validators() which offloads via asyncio.to_thread().
    """
    code = _extract_code_block(output, "python")
    if not code.strip():
        return ValidationResult(False, "No Python code found", "ruff")

    def _run_sync() -> ValidationResult:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", encoding="utf-8", delete=False
            ) as f:
                tmp_path = f.name
                f.write(code)
                f.flush()

            # Errors to ignore: LLM-generated code commonly triggers these
            # and ruff --fix cannot resolve them (they need structural changes):
            #   E501: line too long (cosmetic)
            #   E402: module-level import not at top (LLMs put imports after
            #         sys.path manipulation or __main__ guards — valid patterns)
            #   F401: imported but unused (LLMs import for availability checks
            #         via try/except — ruff removal breaks the logic)
            _IGNORE = "E501,E402,F401"

            # Pass 1: auto-fix all safe fixable issues in-place
            subprocess.run(
                ["ruff", "check", tmp_path, "--select=E,F",
                 f"--ignore={_IGNORE}", "--fix", "--unsafe-fixes"],
                capture_output=True, timeout=timeout,
            )

            # Pass 2: report any remaining errors
            result = subprocess.run(
                ["ruff", "check", tmp_path, "--select=E,F",
                 f"--ignore={_IGNORE}"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
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

    return _run_sync()


def validate_latex(output: str, timeout: int = 30) -> ValidationResult:
    """
    Check LaTeX compiles (requires pdflatex).
    For async callers use async_run_validators() which offloads via asyncio.to_thread().
    """
    def _run_sync() -> ValidationResult:
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

    return _run_sync()


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


# Subprocess-based validators that block the CPU — offloaded to threads
# when called from async context via async_run_validators().
_SUBPROCESS_VALIDATORS = {"pytest", "ruff", "latex"}


async def async_run_validators(output: str, validator_names: list[str],
                               **kwargs) -> list[ValidationResult]:
    """
    Async variant of run_validators().

    Pure-Python validators (json_schema, python_syntax, length) run inline.
    Subprocess validators (pytest, ruff, latex) are offloaded to a thread pool
    via asyncio.to_thread() so the event loop is never blocked.

    An overall async deadline is applied to each subprocess validator equal to
    the 'timeout' kwarg (default: 60 s) + a 5 s grace period.  If the
    thread does not return within that wall-clock window, the coroutine is
    cancelled and a FAIL result is returned so the engine can retry.
    """
    # Wall-clock deadline per subprocess call (subprocess timeout + 5 s grace)
    _subprocess_timeout = kwargs.get("timeout", 60)
    _async_deadline = _subprocess_timeout + 5

    results: list[ValidationResult] = []
    for name in validator_names:
        fn = VALIDATORS.get(name)
        if not fn:
            logger.warning(f"Unknown validator: {name}")
            continue
        filtered = _filter_kwargs_for(fn, kwargs)
        try:
            if name in _SUBPROCESS_VALIDATORS:
                coro = asyncio.to_thread(fn, output, **filtered)
                result = await asyncio.wait_for(coro, timeout=_async_deadline)
            else:
                result = fn(output, **filtered)
            results.append(result)
        except asyncio.TimeoutError:
            logger.warning(
                f"Validator '{name}' exceeded async deadline of {_async_deadline}s; "
                "marking as FAIL"
            )
            results.append(ValidationResult(
                False,
                f"Validator timed out after {_async_deadline}s",
                name,
            ))
        except Exception as e:
            results.append(ValidationResult(False, f"Validator crash: {e}", name))
    return results


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
