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
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("orchestrator.validators")


class ValidationResult:
    __slots__ = ("passed", "details", "validator_name")

    def __init__(self, passed: bool, details: str = "", validator_name: str = ""):
        self.passed = passed
        self.details = details
        self.validator_name = validator_name


def validate_json_schema(output: str, schema: dict | None = None) -> ValidationResult:
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
    truncated = last_line.rstrip().endswith(":") or (  # incomplete annotation like `access_token:`
        last_line.rstrip()[-1:] not in {"}", ")", "]", '"', "'", "\\"}
        and not last_line.strip().startswith("#")
        and ":" not in last_line
        and len(code.splitlines()) >= 50
    )  # only flag as truncated for long outputs

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
            # Provide detailed error context for common issues
            error_msg = f"Syntax error: {e.msg} (line {e.lineno})"

            # Add helpful context
            lines = code.split("\n")
            if e.lineno and 0 < e.lineno <= len(lines):
                error_line = lines[e.lineno - 1]
                error_context = f"\n\nProblematic line {e.lineno}:\n  {error_line}"

                # Specific guidance for common errors
                if "unterminated" in str(e.msg) or "string literal" in str(e.msg):
                    error_context += (
                        '\n\n💡 TIP: Check for unclosed triple-quoted strings (""").\n'
                        '   Every opening """ must have a matching closing """.'
                    )
                elif "EOF" in str(e.msg):
                    error_context += "\n\n💡 TIP: Unexpected end of file - check for unclosed parentheses, brackets, or strings."

            return ValidationResult(False, error_msg + error_context, "python_syntax")


def validate_pytest(output: str, test_code: str = "", timeout: int = 30) -> ValidationResult:
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
        pattern = _re.compile(r"^(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", _re.MULTILINE)
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
                    "# -*- coding: utf-8 -*-\n" "def test_import():\n    import generated_code\n",
                    encoding="utf-8",
                )

            # Set PYTHONIOENCODING so pytest's own output is UTF-8 on Windows
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"  # Python 3.7+ UTF-8 mode

            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", str(tmpdir), "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                    env=env,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode == 0:
                    return ValidationResult(True, result.stdout[-500:], "pytest")
                else:
                    return ValidationResult(
                        False,
                        f"Tests failed (exit {result.returncode}):\n"
                        f"{result.stdout[-500:]}\n{result.stderr[-300:]}",
                        "pytest",
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
        # No explicit Python block found — skip ruff rather than fail
        return ValidationResult(True, "No Python code block, ruff skipped", "ruff")
    
    # FIX: Detect JavaScript/TypeScript content and skip Python validation
    js_indicators = [
        "export default",
        "export const",
        "export function",
        "import {",
        "import *",
        "from '",
        'from "',
        "const ",
        "let ",
        "var ",
        "function ",
        "=> {",
        "React.",
        "react",
        "jsx",
        "</div>",
        "</span>",
        "</p>",
        "className=",
        "useState(",
        "useEffect(",
    ]
    code_sample = code[:1500]  # Check first 1500 chars
    if any(indicator in code_sample for indicator in js_indicators):
        logger.debug("JavaScript/TypeScript detected, skipping ruff validation")
        return ValidationResult(True, "JavaScript/TypeScript detected, ruff skipped", "ruff")

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
                [
                    "ruff",
                    "check",
                    tmp_path,
                    "--select=E,F",
                    f"--ignore={_IGNORE}",
                    "--fix",
                    "--unsafe-fixes",
                ],
                capture_output=True,
                timeout=timeout,
            )

            # Pass 2: report any remaining errors
            result = subprocess.run(
                ["ruff", "check", tmp_path, "--select=E,F", f"--ignore={_IGNORE}"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            if result.returncode == 0:
                return ValidationResult(True, "No lint errors", "ruff")
            else:
                return ValidationResult(False, f"Lint issues:\n{result.stdout[-500:]}", "ruff")
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
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(tex_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                )
                if result.returncode == 0:
                    return ValidationResult(True, "LaTeX compiled", "latex")
                else:
                    return ValidationResult(
                        False, f"LaTeX compilation failed:\n{result.stderr[-500:]}", "latex"
                    )
            except FileNotFoundError:
                return ValidationResult(True, "pdflatex not available, skipped", "latex")
            except subprocess.TimeoutExpired:
                return ValidationResult(False, "LaTeX compilation timed out", "latex")

    return _run_sync()


def validate_length_bounds(
    output: str, min_chars: int = 10, max_chars: int = 50000
) -> ValidationResult:
    """Check output is within reasonable length bounds."""
    length = len(output.strip())
    if length < min_chars:
        return ValidationResult(False, f"Output too short ({length} < {min_chars})", "length")
    if length > max_chars:
        return ValidationResult(False, f"Output too long ({length} > {max_chars})", "length")
    return ValidationResult(True, f"Length OK ({length} chars)", "length")


# HARDEN: Tool call validation — prevent hallucinated tool invocations
# These patterns indicate potentially dangerous LLM-generated commands
_SUSPICIOUS_PATTERNS = [
    # Shell execution
    (
        r"\b(os\.system|subprocess\.call|subprocess\.run|subprocess\.Popen)\s*\(",
        "potential shell execution",
    ),
    # Code evaluation
    (r"\b(eval|exec)\s*\(", "code evaluation"),
    # File system operations outside temp
    (r'\bopen\s*\(\s*["\']/(etc|usr|bin|sbin|root|home)', "system file access"),
    # Network calls
    (r"\b(urllib\.request|requests\.(get|post)|socket\.)", "network call"),
    # Import of dangerous modules
    (r"^\s*import\s+(os|subprocess|sys|socket|urllib)", "suspicious import"),
]


def validate_tool_safety(output: str) -> ValidationResult:
    """
    HARDEN: Validate that output doesn't contain hallucinated tool calls
    or potentially dangerous code patterns.

    This prevents:
    - Shell command injection
    - Code evaluation attacks
    - Unauthorized file system access
    - Unexpected network calls

    Note: This is a safety check, not a functionality check.
    Legitimate uses of these patterns should use the proper validators.
    """
    import re

    found_issues = []
    for pattern, description in _SUSPICIOUS_PATTERNS:
        if re.search(pattern, output, re.MULTILINE | re.IGNORECASE):
            found_issues.append(description)

    if found_issues:
        return ValidationResult(
            False,
            f"Potentially unsafe patterns detected: {', '.join(found_issues)}. "
            f"If these are intentional, use appropriate sandboxed validators.",
            "tool_safety",
        )

    return ValidationResult(True, "No unsafe patterns detected", "tool_safety")


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
    accepted = {
        k
        for k, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and k != "output"
    }  # 'output' is positional arg
    return {k: v for k, v in kwargs.items() if k in accepted}


def run_validators(output: str, validator_names: list[str], **kwargs) -> list[ValidationResult]:
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


async def async_run_validators(
    output: str, validator_names: list[str], **kwargs
) -> list[ValidationResult]:
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

    async def _run_one(name: str) -> ValidationResult | None:
        """Run a single validator and return its result, or None if unknown."""
        fn = VALIDATORS.get(name)
        if not fn:
            logger.warning(f"Unknown validator: {name}")
            return None
        filtered = _filter_kwargs_for(fn, kwargs)
        try:
            if name in _SUBPROCESS_VALIDATORS:
                coro = asyncio.to_thread(fn, output, **filtered)
                result = await asyncio.wait_for(coro, timeout=_async_deadline)
            else:
                result = fn(output, **filtered)
            return result
        except asyncio.TimeoutError:
            logger.warning(
                f"Validator '{name}' exceeded async deadline of {_async_deadline}s; "
                "marking as FAIL"
            )
            return ValidationResult(
                False,
                f"Validator timed out after {_async_deadline}s",
                name,
            )
        except Exception as e:
            return ValidationResult(False, f"Validator crash: {e}", name)

    outcomes = await asyncio.gather(*(_run_one(name) for name in validator_names))
    # Filter out None entries (unknown validators that were skipped)
    return [r for r in outcomes if r is not None]


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
        return text[m.start() :]

    # 4. Fallback: return as-is
    return text


def validate_file_has_content(output: str, min_lines: int = 5, min_code_lines: int = 3) -> ValidationResult:
    """Validate that generated file has actual implementation content.
    
    Checks:
    - File is not empty
    - Has minimum number of lines
    - Has actual code (not just comments/imports)
    - Has function or class definitions
    
    This prevents empty files or files with only imports from passing validation.
    """
    code = _extract_code_block(output, "python")
    lines = code.split("\n")
    
    # Check 1: Not empty
    if not code.strip():
        return ValidationResult(False, "File is empty - no code generated", "file_content")
    
    # Check 2: Minimum lines
    non_empty_lines = [l for l in lines if l.strip()]
    if len(non_empty_lines) < min_lines:
        return ValidationResult(
            False, 
            f"File too short ({len(non_empty_lines)} lines, min {min_lines}) - incomplete implementation", 
            "file_content"
        )
    
    # Check 3: Has actual code (not just comments)
    code_lines = []
    in_multiline_string = False
    for line in lines:
        stripped = line.strip()
        # Track multiline strings
        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') % 2 == 1 or stripped.count("'''") % 2 == 1:
                in_multiline_string = not in_multiline_string
            continue
        # Skip comments and docstrings
        if in_multiline_string or stripped.startswith("#"):
            continue
        if stripped:
            code_lines.append(stripped)
    
    if len(code_lines) < min_code_lines:
        return ValidationResult(
            False,
            f"Too little actual code ({len(code_lines)} code lines, min {min_code_lines}) - needs implementation",
            "file_content"
        )
    
    # Check 4: Has function or class definitions
    has_implementation = any(
        re.match(r"^(def |class |async def )", line)
        for line in code_lines
    )
    
    if not has_implementation:
        return ValidationResult(
            False,
            "No function or class definitions found - file lacks implementation",
            "file_content"
        )
    
    return ValidationResult(
        True,
        f"Content OK: {len(non_empty_lines)} lines, {len(code_lines)} code lines, has implementation",
        "file_content"
    )


def validate_simplicity(output: str, task_description: str = "") -> ValidationResult:
    """Karpathy Principle 2: Simplicity First — detect over-engineering.

    Detects:
    1. Abstract base classes with fewer than 2 concrete subclasses.
    2. Comments mentioning speculative flexibility (configurable, extensible).
    3. Lines-to-features ratio warnings (>200 lines for a single feature).
    4. Empty error handlers (bare 'except: pass').

    This is a soft check — flags issues but does not block execution.
    """
    code = _extract_code_block(output, "python") if output else output
    issues = []

    # Check 1: Single-implementation abstractions
    class_pattern = re.compile(r"class (\w+)\((?:ABC|metaclass=ABCMeta)\)", re.MULTILINE)
    for m in class_pattern.finditer(code):
        base = m.group(1)
        sub_pat = re.compile(rf"class \w+\({base}\)")
        subclass_count = len(sub_pat.findall(code))
        if subclass_count <= 1:
            issues.append(
                f"Abstract class '{base}' has only {subclass_count} concrete subclass(es) — "
                "consider simplifying into a single class"
            )

    # Check 2: Speculative flexibility comments
    speculative_patterns = [
        (r"#.*?\b(configurable|extensible|flexible)\b", "speculative flexibility comment"),
        (r"#.*?\b(might|could|may)\s+need\b", "speculative future-need comment"),
    ]
    for pattern, desc in speculative_patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        if matches:
            issues.append(f"{desc}: found {len(matches)} instance(s)")

    # Check 3: Lines-to-features ratio
    code_lines = [l for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(code_lines) > 200 and task_description:
        multi_file_signals = ["multiple", "several", "database", "migration", "full-stack"]
        if not any(s in task_description.lower() for s in multi_file_signals):
            issues.append(
                f"Output is {len(code_lines)} lines for a single feature — "
                "consider if this could be simplified to under 100 lines"
            )

    # Check 4: Empty error handlers
    empty_handler = re.compile(
        r"except[^:]*:\s*\n\s*(pass|logger\.(info|debug)\(['\"].*?['\"]\))"
    )
    if empty_handler.search(code):
        issues.append(
            "Empty error handler detected (bare 'pass' or silent log). "
            "Either handle the error properly or let it propagate."
        )

    if issues:
        return ValidationResult(
            False,
            f"Simplicity concerns ({len(issues)}):\n" + "\n".join(f"  - {i}" for i in issues),
            "simplicity",
        )

    return ValidationResult(True, "Simplicity check passed", "simplicity")


def validate_surgical_changes(
    output: str = "",
    diff: str = "",
    task_description: str = "",
    locked_files: list[str] | None = None,
) -> ValidationResult:
    """Karpathy Principle 3: Surgical Changes — detect scope creep.

    Detects:
    1. Changes to locked files.
    2. Style/formatting-only changes in untouched code sections.
    3. Removal of pre-existing code unrelated to the task.
    4. New imports not used in the added code.

    This is a soft check — flags issues but does not block execution.
    """
    if not diff:
        return ValidationResult(True, "No diff provided, skipping", "surgical_changes")

    issues: list[str] = []
    locked = set(locked_files or [])

    # Check 1: Locked files modified
    if locked:
        file_pattern = re.compile(r"^diff --git a/(.+) b/(.+)", re.MULTILINE)
        changed_files: set[str] = set()
        for m in file_pattern.finditer(diff):
            changed_files.add(m.group(1))
            changed_files.add(m.group(2))

        for locked_file in locked:
            if any(locked_file in f for f in changed_files):
                issues.append(f"Modified locked file: {locked_file}")

    # Check 2: Style-only changes
    diff_lines = diff.split("\n")
    style_change_count = 0
    for line in diff_lines:
        if not line.startswith(("+", "-")):
            continue
        # Skip actual code changes
        if line[1:].lstrip().startswith(("import ", "from ", "def ", "class ", "return ", "if ", "for ")):
            continue
        if line.startswith("+") and line[1:].lstrip().startswith(("#", "'''", '"""')):
            style_change_count += 1
        if line.startswith("-") and line[1:].lstrip().startswith(("#", "'''", '"""')):
            style_change_count += 1

    if style_change_count > 3:
        issues.append(
            f"Detected {style_change_count} style-only changes (comments, formatting) — "
            "remove style changes unrelated to the task"
        )

    # Check 3: Pre-existing code removed
    removed_defs = re.findall(r"^-\s*(?:def |class )(\w+)", diff, re.MULTILINE)
    if removed_defs and task_description:
        for removed in removed_defs:
            if removed.lower() not in task_description.lower():
                issues.append(
                    f"Removed pre-existing definition '{removed}' — "
                    "not mentioned in task description"
                )

    # Check 4: New imports not used
    new_imports = re.findall(r"^\+import (\w+)", diff, re.MULTILINE)
    if new_imports:
        added_lines = [l for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
        added_code = "\n".join(l[1:] for l in added_lines)
        for imp in new_imports:
            if imp not in added_code:
                issues.append(f"New import '{imp}' not used in added code — remove")

    if issues:
        return ValidationResult(
            False,
            f"Surgical change violations ({len(issues)}):\n" + "\n".join(f"  - {i}" for i in issues),
            "surgical_changes",
        )

    return ValidationResult(True, "Surgical change check passed", "surgical_changes")


def validate_no_error_placeholders(output: str) -> ValidationResult:
    """Validate that output is not an error placeholder.
    
    Detects patterns like:
    - "CODE GENERATION FAILED"
    - "Syntax error:"
    - "ERROR: The generated code failed"
    - raise RuntimeError(...)
    
    These indicate the code generation failed and should be retried.
    """
    error_patterns = [
        r"CODE GENERATION FAILED",
        r"ERROR:.*generated code failed",
        r"Syntax error:",
        r"raise RuntimeError\(.*Code generation failed",
        r"# .*?ERROR.*?\nraise ",
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return ValidationResult(
                False,
                f"Output is an error placeholder - code generation failed: {pattern}",
                "error_placeholder"
            )
    
    return ValidationResult(True, "No error placeholders detected", "error_placeholder")


# ─────────────────────────────────────────────
# Validator registry - MUST be at end after all functions defined
# ─────────────────────────────────────────────

VALIDATORS = {
    "json_schema": validate_json_schema,
    "python_syntax": validate_python_syntax,
    "file_content": validate_file_has_content,  # Check for empty/implementations
    "error_placeholder": validate_no_error_placeholders,  # Detect failed generation
    "pytest": validate_pytest,
    "ruff": validate_ruff,
    "latex": validate_latex,
    "length": validate_length_bounds,
    "tool_safety": validate_tool_safety,  # Prevent hallucinated tool calls
    "simplicity": validate_simplicity,  # Karpathy Principle 2: over-engineering detection
    "surgical_changes": validate_surgical_changes,  # Karpathy Principle 3: scope creep detection
}



# ─────────────────────────────────────────────
# Engine extraction helpers (Phase 4 of ENGINE_OPTIMIZATION_PLAN.md)
# ─────────────────────────────────────────────


def validate_syntax_streaming(partial_output: str) -> bool:
    """OPTIMIZATION: Streaming syntax validator for early abort.

    Checks partial code output for obvious syntax errors:
    - Unclosed brackets/parentheses
    - Invalid Python syntax (early detection)
    - Missing imports for common modules

    Args:
        partial_output: Partial code output (first ~500 tokens)

    Returns:
        True if syntax looks valid, False if obvious errors detected
    """
    import ast

    # Quick bracket balance check
    brackets = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for char in partial_output:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return False  # Unmatched closing bracket
            if brackets[stack.pop()] != char:
                return False  # Mismatched brackets

    # Try parsing as Python (may fail on incomplete code)
    try:
        # Only validate if we have a complete statement (ends with newline)
        if partial_output.strip().endswith(":") or partial_output.count("\n") < 2:
            return True  # Incomplete statement, can't validate yet

        ast.parse(partial_output)
        return True  # Valid syntax
    except SyntaxError as e:
        # Check if error is likely due to incompleteness vs actual error
        error_msg = str(e).lower()
        if "eof" in error_msg or "unexpected eof" in error_msg:
            return True  # Incomplete code, not necessarily wrong
        elif "invalid syntax" in error_msg:
            # Check if it's a common incomplete pattern
            if partial_output.rstrip().endswith((",", "\\", "...")):
                return True  # Likely continuation
            return False  # Actual syntax error
        return True  # Other errors, be lenient


async def validate_syntax_batch(output: str) -> bool:
    """Batch syntax validator for post-generation validation.

    Args:
        output: Complete code output

    Returns:
        True if syntax valid, False otherwise
    """
    import ast
    try:
        ast.parse(output)
        return True
    except SyntaxError:
        return False


def extract_function_name(code: str) -> str | None:
    """Extract the main function name from generated code.

    Args:
        code: Python source code

    Returns:
        Function name or None
    """
    import ast
    import re

    try:
        # Try AST parsing first
        tree = ast.parse(code)

        # Look for the first function definition
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip dunder methods
                if not node.name.startswith("__"):
                    return node.name

        # Fallback: Look for class __init__
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                return node.name

    except SyntaxError:
        # AST parsing failed, try regex
        match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
        if match:
            return match.group(1)

        # Try class name
        class_match = re.search(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
        if class_match:
            return class_match.group(1)

    return None


def filter_validators_for_task(task: object, output: str) -> list[str]:
    """Filter validators based on task type and content.

    Removes Python-specific validators for non-Python tasks.

    Args:
        task: Task object with .hard_validators, .prompt, .target_path attributes
        output: Task output string

    Returns:
        Filtered list of validator names
    """
    if not task.hard_validators:
        return []

    # Detect if this is a Python task
    # NOTE: "import " is intentionally excluded -- JS/TS also use ES module imports
    is_python_task = (
        "python" in task.prompt.lower()
        or ".py" in task.target_path.lower()
        or "flask" in task.prompt.lower()
        or "django" in task.prompt.lower()
        or "fastapi" in task.prompt.lower()
        or "def " in output[:500]  # Python function defs
    )

    # Detect if this is a web/JS/TS task (HTML/CSS/JS/TS/React/Vue)
    is_web_task = (
        "html" in task.prompt.lower()
        or "css" in task.prompt.lower()
        or "javascript" in task.prompt.lower()
        or "typescript" in task.prompt.lower()
        or "react" in task.prompt.lower()
        or "vue" in task.prompt.lower()
        or "angular" in task.prompt.lower()
        or "next.js" in task.prompt.lower()
        or " js " in task.prompt.lower()
        or task.prompt.lower().endswith(" js")
        or ".html" in task.target_path.lower()
        or ".css" in task.target_path.lower()
        or ".js" in task.target_path.lower()
        or ".ts" in task.target_path.lower()
        or ".tsx" in task.target_path.lower()
        or ".jsx" in task.target_path.lower()
        or "<!DOCTYPE" in output[:100]
        or "<html" in output[:100]
        or "function(" in output[:500]
        or "const " in output[:500]
        or "export default" in output[:1000]  # JS/TS module export
        or "export const" in output[:1000]  # JS/TS named export
        or "from 'react'" in output[:500]  # React import (single quotes)
        or 'from "react"' in output[:500]  # React import (double quotes)
    )

    if is_web_task or not is_python_task:
        # Remove Python-specific validators
        original = set(task.hard_validators)
        filtered = [
            v for v in task.hard_validators if v not in ("python_syntax", "ruff", "pytest")
        ]
        removed = original - set(filtered)
        if removed:
            import logging
            logger = logging.getLogger("orchestrator.validators")
            logger.info(
                f"Task {task.id}: skipped Python validators {removed} (non-Python content detected)"
            )
        return filtered

    return task.hard_validators
