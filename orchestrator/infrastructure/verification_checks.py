"""
Verification check adapters — concrete shell-based check implementations.

WBS-1: test, lint, type, build, artifact, and security check adapters.

Each factory function returns a ``VerificationCheck`` compatible with
the application-layer ``VerificationGate``.  These live in infrastructure
because they execute shell commands — application code should never import
from this module directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from orchestrator.application.verification_gate import VerificationCheck

logger = logging.getLogger("orchestrator.infrastructure.verification_checks")

CheckFn = Callable[[str], Awaitable[tuple[bool, str]]]
"""Async function (artifact: str) -> (passed: bool, reason: str)."""


def strip_code_block(artifact: str) -> str:
    """If the artifact is wrapped in a markdown code block, extract the inner content."""
    text = artifact.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 1 and lines[0].startswith("```"):
            start_idx = 1
            end_idx = len(lines)
            if lines[-1].startswith("```"):
                end_idx = len(lines) - 1
            else:
                for i in range(len(lines) - 1, 0, -1):
                    if lines[i].startswith("```"):
                        end_idx = i
                        break
            return "\n".join(lines[start_idx:end_idx])
    return artifact


def is_python_code(artifact: str) -> bool:
    """Detect if the artifact is Python code. Returns False for TS/JS, JSON, and CLI/shell scripts."""
    text = strip_code_block(artifact).strip()
    if not text:
        return False

    # Remove markdown code-blocks if present
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 1:
            lang = lines[0].replace("```", "").strip().lower()
            if lang in ("python", "py"):
                return True
            if lang in (
                "typescript",
                "ts",
                "javascript",
                "js",
                "html",
                "css",
                "json",
                "bash",
                "sh",
                "yaml",
                "yml",
                "markdown",
                "md",
            ):
                return False

    # Check for typescript/javascript/bash/json/html patterns
    non_python_patterns = [
        "import React",
        "import {",
        "export default",
        "export const",
        "export interface",
        "interface ",
        "const ",
        "let ",
        "type ",
        "npm ",
        "yarn ",
        "pnpm ",
        "npx ",
        "node ",
        "package.json",
        "tsconfig.json",
        "vite.config",
        "<div",
        "</",
        "import type",
        "as const",
    ]
    if any(p in text for p in non_python_patterns):
        return False

    # If it parses as JSON, it's not Python code (except maybe a trivial number or string)
    import json

    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            return False
        except Exception:
            pass

    # Try compile. If it compiles successfully as Python, then it is indeed Python.
    try:
        compile(text, "<verify>", "exec")
        return True
    except SyntaxError:
        # If compile failed, check for TS/JS patterns or CLI/shell scripts
        import re

        if re.search(r"===|=>|\bfunction\b", text):
            return False
        # If it starts with common bash commands:
        if text.startswith(("npm ", "cd ", "git ", "pip install ", "python -m ", "python3 ")):
            return False

    # A line ending in a bare "=" is a truncated assignment: the model was cut
    # off mid-statement. No prose ends a line that way, so this is safe to treat
    # as Python even though it does not compile. Without it, `x = ` was
    # classified "not Python" and the syntax check PASSED a truncated artifact —
    # exactly the silent failure the check exists to catch. Deliberately narrow:
    # matching a general `name = value` pattern would misclassify prose such as
    # "Total = 5 items" and reject valid non-code output.
    if re.search(r"^\s*[A-Za-z_][\w.\[\]'\"]*\s*=\s*$", text, re.M):
        return True

    # Check for standard Python markers:
    python_markers = ["def ", "class ", "import ", "from ", "print(", "#"]
    if any(m in text for m in python_markers):
        return True

    # Ambiguous assignment/expression fragments (e.g. "x = ") — treat as
    # Python so the syntax check can report the real SyntaxError instead
    # of silently passing broken code (pre-existing bug: is_python_code
    # returned False for "x = ", letting invalid Python through the gate).
    import re as _re

    if _re.search(r"\b\w+\s*=", text):
        return True

    return False


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _run_command(
    command: list[str],
    cwd: str | None = None,
    timeout: float = 30.0,
    input_text: str | None = None,
) -> tuple[int, str, str]:
    """Run a command in a subprocess and return (returncode, stdout, stderr).

    All check adapters delegate to this to ensure consistent error handling,
    timeout, and logging.  Uses text=False and decodes manually for Windows compat.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        )  # nosec B603 — list args prevent shell injection; stdin is LLM artifact
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=input_text.encode("utf-8") if input_text is not None else None),
            timeout=timeout,
        )
        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        return proc.returncode or 0, stdout, stderr
    except asyncio.TimeoutError:
        logger.warning("Command timed out after %ss: %s", timeout, " ".join(command))
        try:
            proc.kill()
        except Exception:
            pass  # best-effort cleanup; proc may already be dead
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("Subprocess %d refused to die after kill — continuing", proc.pid)
        return -1, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError:
        logger.warning("Command not found: %s", command[0])
        return -2, "", f"Command not found: {command[0]}"
    except Exception as exc:
        logger.error("Command failed unexpectedly: %s", exc)
        return -3, "", str(exc)


# ── Check Factories ────────────────────────────────────────────────────────────


def _make_syntax_check() -> VerificationCheck:
    """Check that Python code compiles (AST parse)."""

    async def _check(artifact: str) -> tuple[bool, str]:
        artifact = strip_code_block(artifact)
        if not is_python_code(artifact):
            return True, ""
        try:
            compile(artifact, "<verify>", "exec")
            return True, ""
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"

    return make_check_adapter(name="syntax", run=_check, command="compile(<verify>)")


def _make_lint_check() -> VerificationCheck:
    """Check artifact with ruff (if available)."""

    async def _check(artifact: str) -> tuple[bool, str]:
        artifact = strip_code_block(artifact)
        if not is_python_code(artifact):
            return True, ""
        returncode, stdout, stderr = await _run_command(
            [sys.executable, "-m", "ruff", "check", "--stdin-filename", "verify.py", "-"],
            input_text=artifact,
        )
        if returncode == 0:
            return True, ""
        if returncode == -2:
            return False, "ruff not installed"
        return False, stdout[:500] or stderr[:500]

    return make_check_adapter(
        name="lint", run=_check, command="ruff check --stdin-filename verify.py -"
    )


def _make_type_check() -> VerificationCheck:
    """Check artifact with mypy (if available)."""

    # Configurable timeout via ORCH_MYPY_TIMEOUT env var (default 30s)
    try:
        mypy_timeout = float(os.environ.get("ORCH_MYPY_TIMEOUT", "30.0"))
        if mypy_timeout <= 0 or mypy_timeout > 120:
            raise ValueError("out of bounds")
    except (ValueError, TypeError):
        mypy_timeout = 30.0
        logger.warning("Invalid ORCH_MYPY_TIMEOUT value, falling back to 30s")

    async def _check(artifact: str) -> tuple[bool, str]:
        artifact = strip_code_block(artifact)
        if not is_python_code(artifact):
            return True, ""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(artifact)
            tmp_path = f.name

        try:
            returncode, stdout, stderr = await _run_command(
                [sys.executable, "-m", "mypy", "--show-error-codes", tmp_path],
                timeout=mypy_timeout,
            )
            if returncode in (0, -2):  # 0 = clean, -2 = not found
                return returncode == 0, stdout[:500] if returncode != 0 else ""
            return False, stdout[:500] or stderr[:500]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return make_check_adapter(
        name="type_check", run=_check, command="mypy --show-error-codes <tempfile>"
    )


def _make_build_check() -> VerificationCheck:
    """Check artifact can be imported (executes import-level code in isolated subprocess).

    Uses asyncio.create_subprocess_exec in a temp directory instead of
    in-process exec() to avoid security risks (D-2 fix).
    """

    async def _check(artifact: str) -> tuple[bool, str]:
        artifact = strip_code_block(artifact)
        if not is_python_code(artifact):
            return True, ""
        try:
            import tempfile as _tf

            # Write artifact to a temp file and try to import it in a subprocess
            with _tf.TemporaryDirectory() as td:
                test_file = Path(td) / "_verify_target.py"
                test_file.write_text(artifact, encoding="utf-8")

                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    "import sys, os; "
                    "sys.path.insert(0, os.environ['_ORCH_VERIFY_DIR']); "
                    "import _verify_target",
                    env={**os.environ, "_ORCH_VERIFY_DIR": td},
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return False, "Import check timed out"

                if proc.returncode != 0:
                    stderr_text = stderr.decode("utf-8", errors="replace")
                    if "SyntaxError" in stderr_text:
                        return False, stderr_text.splitlines()[-1].strip()
                    return False, (
                        stderr_text.splitlines()[-1].strip() if stderr_text else "Import failed"
                    )

                return True, ""
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
        except ImportError as exc:
            return False, f"ImportError: {exc}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    return make_check_adapter(name="build", run=_check, command="subprocess_import(<verify>)")


def _make_security_check() -> VerificationCheck:
    """Basic security scan: detect common dangerous patterns.

    This is a lightweight static scan.  A full security check would use
    bandit or similar tool.  This catches the most obvious issues.
    """

    DANGEROUS_PATTERNS: list[tuple[str, str]] = [
        ("subprocess.call", "subprocess without shell=False not enforced"),
        ("eval(", "eval() can execute arbitrary code"),
        ("exec(", "exec() can execute arbitrary code"),
        ("__import__(", "dynamic import"),
        ("pickle.loads", "unsafe deserialization"),
        ("yaml.load(", "unsafe YAML load (use safe_load)"),
    ]

    async def _check(artifact: str) -> tuple[bool, str]:
        artifact = strip_code_block(artifact)
        findings: list[str] = []
        for pattern, description in DANGEROUS_PATTERNS:
            for i, line in enumerate(artifact.splitlines(), 1):
                if pattern in line:
                    # Skip comments
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    findings.append(f"L{i}: {description} ({pattern})")

        if findings:
            return False, "; ".join(findings[:5])
        return True, ""

    return make_check_adapter(name="security", run=_check, command="static-pattern-scan")


# ── VerificationCheckAdapter (factory for application-layer checks) ───────


def make_check_adapter(name: str, run: CheckFn, command: str | None = None) -> VerificationCheck:
    """Create an application-layer VerificationCheck from infrastructure code.

    This is the only bridge between infrastructure and application layers —
    it wraps a concrete check implementation into the CheckFn protocol
    expected by the verification gate.
    """
    from orchestrator.application.verification_gate import VerificationCheck

    return VerificationCheck(name=name, run=run, command=command)


# ── Test execution check (E-6) ──────────────────────────────────────────────


def _make_test_execution_check(
    runner=None, sandbox=None, timeout_s: float = 120.0
) -> VerificationCheck:
    """Create the WORKSPACE-scoped test-execution check (E-6).

    The check materializes nothing itself — it receives a pre-built
    ``Workspace`` (domain object) from the gate dispatch and executes the
    suite through the authoritative runner (F-6). A vacuous result (zero
    executed tests) is never a pass (D-7); collection errors are reported
    as infrastructure failures.

    Args:
        runner: Optional TestExecutorPort; defaults to the framework
            registry's pytest runner bound to the resolved sandbox.
        sandbox: Optional SandboxPort; defaults to the resolved tier.
        timeout_s: Suite timeout.

    Returns:
        A WORKSPACE-scoped VerificationCheck named ``test_execution``.
    """

    async def _check(artifact: str, workspace=None) -> tuple[bool, str]:
        if workspace is None:
            return False, "workspace required for test execution (E-6)"
        from ..domain.testing_models import TestStatus

        try:
            from .test_runners import get_runner

            effective_runner = runner
            if effective_runner is None:
                effective_runner = get_runner(workspace.framework, sandbox=sandbox)
            report = await effective_runner.run(workspace, timeout_s=timeout_s)
            # E-8: stamp structured metadata for the gate receipt.
            check.last_report = report  # type: ignore[attr-defined]
        except Exception as exc:
            return False, f"test execution failed: {exc}"

        if report.is_vacuous_result:
            return False, ("0 tests executed — a vacuous suite is never a pass (D-7)")
        if report.collection_errors:
            return False, "collection errors: " + "; ".join(report.collection_errors[:2])
        failed = sum(
            1 for o in report.outcomes if o.status in (TestStatus.FAILED, TestStatus.ERROR)
        )
        if not report.passed or failed:
            return False, f"{failed}/{report.executed} tests failed"
        return True, f"{report.executed} tests passed"

    from orchestrator.application.verification_gate import VerificationCheck
    from ..domain.testing_models import CheckScope

    check = VerificationCheck(
        name="test_execution",
        run=_check,
        command="sandboxed suite execution",
        scope=CheckScope.WORKSPACE,
    )
    return check


# ── Default Check Set ──────────────────────────────────────────────────────────


def default_checks() -> list[VerificationCheck]:
    """Return a list of all available check adapters.

    Lint and type checks are conditional on tool availability; syntax,
    build, security, and test-execution checks always work. The
    test-execution check is WORKSPACE-scoped: it emits NOT_RUN when no
    workspace is provided (E-6).
    """
    checks: list[VerificationCheck] = [
        _make_syntax_check(),
        _make_build_check(),
        _make_security_check(),
        _make_test_execution_check(),
    ]

    # Try adding lint check
    try:
        lint = _make_lint_check()
        checks.append(lint)
    except Exception:
        logger.debug("Lint check not available")

    # Try adding type check
    try:
        type_check = _make_type_check()
        checks.append(type_check)
    except Exception:
        logger.debug("Type check not available")

    return checks
