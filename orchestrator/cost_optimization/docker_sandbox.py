"""
Docker Sandbox Module
======================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements secure code execution in isolated Docker containers.

Features:
- Isolated container execution
- Resource limits (CPU, memory, network)
- Timeout enforcement
- Security isolation (no host access)

Usage:
    from orchestrator.cost_optimization import DockerSandbox

    sandbox = DockerSandbox()
    result = await sandbox.execute(
        code_files={"main.py": "print('hello')"},
        command="python main.py",
        timeout=30,
    )
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of sandbox execution."""

    return_code: int
    output: str
    error: str = ""
    timeout: bool = False
    memory_exceeded: bool = False
    execution_time: float = 0.0


@dataclass
class SandboxMetrics:
    """Metrics for sandbox execution."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    timeouts: int = 0
    memory_violations: int = 0
    security_violations: int = 0
    avg_execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "timeouts": self.timeouts,
            "memory_violations": self.memory_violations,
            "security_violations": self.security_violations,
            "avg_execution_time": self.avg_execution_time,
            "success_rate": self.successful_executions / max(1, self.total_executions),
        }


class DockerSandbox:
    """
    Secure code execution in isolated Docker containers.

    Usage:
        sandbox = DockerSandbox()
        result = await sandbox.execute(code_files, command, timeout=30)
    """

    # Default Docker image
    DEFAULT_IMAGE = "python:3.12-slim"

    # Resource limits
    DEFAULT_MEMORY_LIMIT = "256m"  # 256MB
    DEFAULT_CPU_QUOTA = 50000  # 50% of one CPU
    DEFAULT_CPU_PERIOD = 100000  # 100ms period
    DEFAULT_TIMEOUT = 30  # 30 seconds
    DEFAULT_NETWORK_DISABLED = True  # No network access

    def __init__(
        self,
        image: str | None = None,
        memory_limit: str | None = None,
        cpu_quota: int | None = None,
        timeout: int | None = None,
    ):
        """
        Initialize Docker sandbox.

        Args:
            image: Docker image to use
            memory_limit: Memory limit (e.g., "256m")
            cpu_quota: CPU quota (microseconds)
            timeout: Default timeout in seconds
        """
        self.image = image or self.DEFAULT_IMAGE
        self.memory_limit = memory_limit or self.DEFAULT_MEMORY_LIMIT
        self.cpu_quota = cpu_quota or self.DEFAULT_CPU_QUOTA
        self.cpu_period = self.DEFAULT_CPU_PERIOD
        self.default_timeout = timeout or self.DEFAULT_TIMEOUT
        self.network_disabled = self.DEFAULT_NETWORK_DISABLED

        self.metrics = SandboxMetrics()
        self._workspaces: dict[str, Path] = {}

        # Check if Docker is available
        self._docker_available: bool | None = None

    async def _check_docker(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            import docker

            client = docker.from_env()
            client.ping()
            self._docker_available = True
            logger.info("Docker is available")
            return True
        except Exception as e:
            logger.warning(f"Docker not available: {e}, falling back to subprocess")
            self._docker_available = False
            return False

    async def execute(
        self,
        code_files: dict[str, str],
        command: str,
        timeout: int | None = None,
        working_dir: str = "/app",
        environment: dict[str, str] | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute code in isolated Docker container.

        Args:
            code_files: Dictionary of filename -> content
            command: Command to execute
            timeout: Timeout in seconds
            working_dir: Working directory in container
            environment: Environment variables
            **kwargs: Additional Docker parameters

        Returns:
            ExecutionResult with output, errors, etc.
        """
        import time

        start_time = time.time()

        self.metrics.total_executions += 1
        timeout = timeout or self.default_timeout

        # Check Docker availability
        docker_available = await self._check_docker()

        if docker_available:
            try:
                import docker

                client = docker.from_env()

                # Create temporary workspace
                workspace = Path(tempfile.mkdtemp(prefix="sandbox_"))
                self._workspaces[str(workspace)] = workspace

                # Write code files
                for filename, content in code_files.items():
                    file_path = workspace / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)

                # Run container
                container = client.containers.run(
                    self.image,
                    command=f"bash -c '{command}'",
                    volumes={str(workspace): {"bind": working_dir, "mode": "rw"}},
                    working_dir=working_dir,
                    network_disabled=self.network_disabled,
                    mem_limit=self.memory_limit,
                    cpu_period=self.cpu_period,
                    cpu_quota=self.cpu_quota,
                    detach=True,
                    remove=False,  # We'll remove manually
                )

                try:
                    # Wait for completion with timeout
                    result = container.wait(timeout=timeout)
                    logs = container.logs().decode("utf-8", errors="replace")

                    execution_time = time.time() - start_time

                    # Update metrics
                    if result["StatusCode"] == 0:
                        self.metrics.successful_executions += 1
                    else:
                        self.metrics.failed_executions += 1

                    self.metrics.avg_execution_time = (
                        self.metrics.avg_execution_time * (self.metrics.total_executions - 1)
                        + execution_time
                    ) / self.metrics.total_executions

                    return ExecutionResult(
                        return_code=result["StatusCode"],
                        output=logs,
                        execution_time=execution_time,
                    )

                except Exception as e:
                    # Timeout or other error
                    container.kill()
                    execution_time = time.time() - start_time

                    if "timeout" in str(e).lower():
                        self.metrics.timeouts += 1
                        return ExecutionResult(
                            return_code=-1,
                            output="",
                            error=f"Execution timeout after {timeout}s",
                            timeout=True,
                            execution_time=execution_time,
                        )

                    self.metrics.failed_executions += 1
                    return ExecutionResult(
                        return_code=-1,
                        output="",
                        error=str(e),
                        execution_time=execution_time,
                    )

                finally:
                    # Cleanup container
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass

                    # FIX-OPT-003a: Use robust cleanup with retry logic
                    self._cleanup_workspace(workspace)

            except Exception as e:
                logger.error(f"Docker execution failed: {e}")
                self.metrics.failed_executions += 1
                return ExecutionResult(
                    return_code=-1,
                    output="",
                    error=f"Docker error: {str(e)}",
                )

        else:
            # FIX-OPT-001a: Fail-closed - no insecure fallback
            # Security > availability: Docker is required for safe code execution
            logger.error(
                "Docker not available. For security, code execution requires Docker. "
                "Install Docker: https://docs.docker.com/get-docker/"
            )
            self.metrics.failed_executions += 1
            return ExecutionResult(
                return_code=-1,
                output="",
                error="Docker not available. Code execution requires Docker for security isolation. Please install Docker.",
            )

    def _cleanup_workspace(self, workspace: Path, max_retries: int = 3) -> None:
        """
        FIX-OPT-003a: Cleanup workspace with retry logic.

        Args:
            workspace: Workspace path to cleanup
            max_retries: Maximum retry attempts
        """
        import shutil
        import time

        for attempt in range(max_retries):
            try:
                shutil.rmtree(workspace, ignore_errors=False)
                if str(workspace) in self._workspaces:
                    del self._workspaces[str(workspace)]
                logger.debug(f"Cleaned up workspace: {workspace}")
                return
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 0.1 * (2**attempt)
                    logger.warning(
                        f"Cleanup attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to cleanup workspace after {max_retries} attempts: {workspace} - {e}"
                    )
                    # Keep in orphaned workspaces for manual cleanup
                    self._workspaces[str(workspace)] = workspace

    async def cleanup(self) -> None:
        """Cleanup all workspaces."""
        import shutil

        for workspace_path in list(self._workspaces.values()):
            try:
                shutil.rmtree(workspace_path, ignore_errors=True)
            except Exception:
                pass
        self._workspaces.clear()
        logger.info("Sandbox cleanup complete")


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────


async def execute_in_sandbox(
    code_files: dict[str, str],
    command: str,
    timeout: int = 30,
) -> ExecutionResult:
    """
    Convenience function for sandbox execution.

    Args:
        code_files: Dictionary of filename -> content
        command: Command to execute
        timeout: Timeout in seconds

    Returns:
        ExecutionResult
    """
    sandbox = DockerSandbox()
    return await sandbox.execute(code_files, command, timeout)


__all__ = [
    "DockerSandbox",
    "ExecutionResult",
    "SandboxMetrics",
    "execute_in_sandbox",
]
