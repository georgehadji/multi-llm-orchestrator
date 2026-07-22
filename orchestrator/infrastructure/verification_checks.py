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
        try:
            compile(artifact, "<verify>", "exec")
            return True, ""
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"

    return make_check_adapter(name="syntax", run=_check, command="compile(<verify>)")


def _make_lint_check() -> VerificationCheck:
    """Check artifact with ruff (if available)."""

    async def _check(artifact: str) -> tuple[bool, str]:
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
    """Check artifact can be imported (executes import-level code in isolated namespace)."""

    async def _check(artifact: str) -> tuple[bool, str]:
        try:
            code = compile(artifact, "<verify>", "exec")
            ns: dict[str, object] = {}
            exec(code, ns)  # nosec B102 — isolated namespace (empty dict), no caller access
            return True, ""
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
        except ImportError as exc:
            return False, f"ImportError: {exc}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    return make_check_adapter(name="build", run=_check, command="exec(<verify>)")


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


# ── Default Check Set ──────────────────────────────────────────────────────────


def default_checks() -> list[VerificationCheck]:
    """Return a list of all available check adapters.

    Lint and type checks are conditional on tool availability; syntax,
    build, and security checks always work.
    """
    checks: list[VerificationCheck] = [
        _make_syntax_check(),
        _make_build_check(),
        _make_security_check(),
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
