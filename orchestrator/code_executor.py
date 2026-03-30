"""
Secure Code Executor
=====================
Author: Georgios-Chrysovalantis Chatzivantsidis

FIX #4: Mandatory sandbox for all code execution.

Usage:
    executor = CodeExecutor(require_sandbox=True)
    result = await executor.execute(code, language="python")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from .log_config import get_logger

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Code execution mode."""
    SANDBOX = "sandbox"  # Docker container (secure)
    LOCAL = "local"      # Direct execution (insecure)
    DISABLED = "disabled"  # No execution allowed


@dataclass
class ExecutionConfig:
    """Configuration for code execution."""
    # Security settings
    require_sandbox: bool = True  # FIX #4: Default to secure
    sandbox_image: str = "python:3.12-slim"
    sandbox_memory_limit: str = "256m"
    sandbox_timeout: int = 30

    # Network settings
    sandbox_network_disabled: bool = True  # No network access

    # Filesystem settings
    sandbox_readonly_root: bool = True
    allowed_paths: list[str] = field(default_factory=list)

    # Execution settings
    max_output_size: int = 1024 * 1024  # 1MB max output

    # Fallback behavior
    fail_if_sandbox_unavailable: bool = True  # FIX #4: Fail closed


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0
    sandbox_used: bool = False
    security_warnings: list[str] = field(default_factory=list)


class CodeExecutor:
    """
    Secure code executor with mandatory sandboxing.

    FIX #4: All code execution goes through sandbox by default.
    """

    def __init__(self, config: ExecutionConfig | None = None):
        """
        Initialize code executor.

        Args:
            config: Execution configuration (default: secure settings)
        """
        self.config = config or ExecutionConfig()
        self._sandbox = None

        # FIX #4: Validate security settings at startup
        self._validate_security_settings()

    def _validate_security_settings(self) -> None:
        """
        FIX #4: Validate security settings and warn about risks.
        """
        if not self.config.require_sandbox:
            logger.warning(
                "⚠️  SECURITY WARNING: Sandbox disabled. "
                "Generated code will execute directly on host. "
                "Set REQUIRE_SANDBOX=true for production."
            )

        if self.config.sandbox_network_disabled:
            logger.debug("Sandbox network isolation enabled")
        else:
            logger.warning(
                "⚠️  SECURITY WARNING: Sandbox has network access. "
                "Generated code can make outbound connections."
            )

    async def execute(
        self,
        code: str,
        language: str = "python",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Execute code securely.

        FIX #4: All execution goes through sandbox (if enabled).

        Args:
            code: Code to execute
            language: Programming language
            args: Command-line arguments
            env: Environment variables
            timeout: Execution timeout (seconds)

        Returns:
            ExecutionResult with output and security metadata
        """
        security_warnings = []

        # FIX #4: Check if execution is allowed
        if self.config.require_sandbox and not self._is_sandbox_available():
            if self.config.fail_if_sandbox_unavailable:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=(
                        "Code execution requires Docker sandbox. "
                        "Install Docker: https://docs.docker.com/get-docker/ "
                        "Or set REQUIRE_SANDBOX=false (not recommended)."
                    ),
                    security_warnings=[
                        "Sandbox unavailable - execution blocked"
                    ],
                )
            else:
                security_warnings.append(
                    "⚠️  FALLBACK: Executing without sandbox (insecure)"
                )
                logger.warning(
                    "Executing code without sandbox - security risk!"
                )

        # FIX #4: Use sandbox if available and required
        if self.config.require_sandbox and self._is_sandbox_available():
            return await self._execute_in_sandbox(
                code, language, args, env, timeout, security_warnings
            )
        else:
            # Fallback to local execution (insecure)
            return await self._execute_local(
                code, language, args, env, timeout, security_warnings
            )

    def _is_sandbox_available(self) -> bool:
        """Check if Docker sandbox is available."""
        try:
            import docker
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    async def _execute_in_sandbox(
        self,
        code: str,
        language: str,
        args: list[str] | None,
        env: dict[str, str] | None,
        timeout: int | None,
        security_warnings: list[str],
    ) -> ExecutionResult:
        """Execute code in Docker sandbox."""
        # Import sandbox module
        from .cost_optimization.docker_sandbox import DockerSandbox

        sandbox = DockerSandbox(
            image=self.config.sandbox_image,
            memory_limit=self.config.sandbox_memory_limit,
            timeout=timeout or self.config.sandbox_timeout,
        )

        # Prepare code files
        if language == "python":
            code_files = {"script.py": code}
            command = "python script.py"
        else:
            code_files = {"script": code}
            command = "./script"

        # Execute in sandbox
        result = await sandbox.execute(
            code_files=code_files,
            command=command,
            timeout=timeout or self.config.sandbox_timeout,
        )

        return ExecutionResult(
            success=result.return_code == 0,
            output=result.output,
            error=result.error,
            exit_code=result.return_code,
            execution_time_ms=result.execution_time * 1000,
            sandbox_used=True,
            security_warnings=security_warnings,
        )

    async def _execute_local(
        self,
        code: str,
        language: str,
        args: list[str] | None,
        env: dict[str, str] | None,
        timeout: int | None,
        security_warnings: list[str],
    ) -> ExecutionResult:
        """
        Execute code locally (INSECURE - fallback only).

        FIX #4: This is only used when sandbox disabled.
        """
        import asyncio
        import tempfile
        import time

        start_time = time.time()

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Execute
            cmd = ["python", temp_path] if language == "python" else [temp_path]

            if args:
                cmd.extend(args)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout or self.config.sandbox_timeout,
            )

            execution_time = (time.time() - start_time) * 1000

            return ExecutionResult(
                success=proc.returncode == 0,
                output=stdout.decode('utf-8', errors='replace'),
                error=stderr.decode('utf-8', errors='replace'),
                exit_code=proc.returncode or 0,
                execution_time_ms=execution_time,
                sandbox_used=False,
                security_warnings=security_warnings + [
                    "⚠️  Executed without sandbox (insecure)"
                ],
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timeout after {timeout}s",
                exit_code=-1,
                execution_time_ms=timeout * 1000,
                sandbox_used=False,
                security_warnings=security_warnings,
            )

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def get_metrics(self) -> dict:
        """Get executor metrics."""
        return {
            "sandbox_available": self._is_sandbox_available(),
            "require_sandbox": self.config.require_sandbox,
            "fail_if_unavailable": self.config.fail_if_sandbox_unavailable,
        }


__all__ = [
    "CodeExecutor",
    "ExecutionConfig",
    "ExecutionMode",
    "ExecutionResult",
]
