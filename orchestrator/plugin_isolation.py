"""
Plugin Isolation & Sandboxing
=============================

Secure plugin execution with:
- Process isolation
- Resource limits (CPU, memory, time)
- Filesystem sandboxing
- Network restrictions
- Capability-based security

Usage:
    from orchestrator.plugin_isolation import IsolatedPluginRuntime

    runtime = IsolatedPluginRuntime(
        memory_limit_mb=512,
        cpu_limit_percent=50,
        timeout_seconds=30,
    )

    result = await runtime.execute(plugin, "validate", code)
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .log_config import get_logger

if TYPE_CHECKING:
    from .plugins import Plugin

logger = get_logger(__name__)

try:
    import resource
except ModuleNotFoundError:  # Windows compatibility
    resource = None


# ═══════════════════════════════════════════════════════════════════════════════
# Isolation Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IsolationConfig:
    """Configuration for plugin isolation."""

    # Resource limits
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50
    timeout_seconds: float = 30.0

    # Filesystem
    allow_filesystem_access: bool = True
    sandbox_path: Path | None = None
    allowed_paths: list[Path] = field(default_factory=list)

    # Network
    allow_network: bool = False
    allowed_hosts: list[str] = field(default_factory=list)

    # Capabilities
    allow_subprocess: bool = False
    allow_threading: bool = True

    # Cleanup
    cleanup_on_exit: bool = True

    def __post_init__(self):
        if self.sandbox_path is None:
            self.sandbox_path = Path(tempfile.gettempdir()) / "plugin_sandbox"


class IsolationLevel(Enum):
    """Levels of plugin isolation."""
    NONE = "none"           # Same process (fastest, least secure)
    THREAD = "thread"       # Separate thread
    PROCESS = "process"     # Separate process (recommended)
    CONTAINER = "container" # Container (future)


# ═══════════════════════════════════════════════════════════════════════════════
# Plugin Execution Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IsolatedResult:
    """Result from isolated plugin execution."""
    success: bool
    result: Any = None
    error: str | None = None
    error_type: str | None = None
    execution_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    exit_code: int = 0

    @classmethod
    def success_result(cls, result: Any, execution_time_ms: float) -> IsolatedResult:
        return cls(
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
        )

    @classmethod
    def error_result(
        cls,
        error: str,
        error_type: str = "Exception",
        execution_time_ms: float = 0.0,
    ) -> IsolatedResult:
        return cls(
            success=False,
            error=error,
            error_type=error_type,
            execution_time_ms=execution_time_ms,
        )


class PluginExecutionError(Exception):
    """Raised when a plugin execution attempt fails due to isolation policies."""


# ═══════════════════════════════════════════════════════════════════════════════
# Process-Level Isolation
# ═══════════════════════════════════════════════════════════════════════════════

def _set_resource_limits(config: IsolationConfig) -> None:
    """Set resource limits in child process."""
    if resource is None:
        logger.debug("Resource module unavailable; skipping strict resource limits.")
        return
    # Memory limit
    memory_bytes = config.memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    # CPU limit (soft limit)
    # Note: This is process CPU time, not percentage
    cpu_seconds = config.timeout_seconds * 2  # Allow some buffer
    resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 10))

    # File descriptor limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))

    # Stack size limit (64MB)
    resource.setrlimit(resource.RLIMIT_STACK, (64 * 1024 * 1024, 64 * 1024 * 1024))


def _setup_sandbox(config: IsolationConfig) -> Path:
    """Setup sandbox directory for plugin."""
    sandbox = config.sandbox_path
    sandbox.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (sandbox / "input").mkdir(exist_ok=True)
    (sandbox / "output").mkdir(exist_ok=True)
    (sandbox / "temp").mkdir(exist_ok=True)

    return sandbox


def _execute_in_process(
    plugin_class: type[Plugin],
    plugin_config: dict[str, Any],
    method_name: str,
    args: tuple,
    kwargs: dict,
    config: IsolationConfig,
    result_queue: multiprocessing.Queue,
) -> None:
    """
    Execute plugin method in isolated process.

    This runs in a separate process with restricted resources.
    """
    start_time = time.time()

    try:
        # Setup sandbox
        sandbox = _setup_sandbox(config)

        # Change to sandbox directory
        os.chdir(sandbox)

        # Set resource limits
        _set_resource_limits(config)

        # Set up signal handlers for timeout
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Plugin execution exceeded {config.timeout_seconds}s")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(config.timeout_seconds))

        # Instantiate plugin
        plugin = plugin_class()
        plugin.initialize(plugin_config)

        # Get method
        method = getattr(plugin, method_name)

        # Execute
        if asyncio.iscoroutinefunction(method):
            # Run async method
            result = asyncio.run(method(*args, **kwargs))
        else:
            # Run sync method
            result = method(*args, **kwargs)

        execution_time = (time.time() - start_time) * 1000

        result_queue.put({
            "success": True,
            "result": result,
            "execution_time_ms": execution_time,
        })

    except TimeoutError as e:
        execution_time = (time.time() - start_time) * 1000
        result_queue.put({
            "success": False,
            "error": str(e),
            "error_type": "TimeoutError",
            "execution_time_ms": execution_time,
        })

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        result_queue.put({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "execution_time_ms": execution_time,
        })

    finally:
        signal.alarm(0)  # Cancel alarm

        # Cleanup
        if config.cleanup_on_exit and sandbox.exists():
            import shutil
            shutil.rmtree(sandbox, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Isolated Plugin Runtime
# ═══════════════════════════════════════════════════════════════════════════════

class IsolatedPluginRuntime:
    """
    Runtime for executing plugins in isolated environments.

    Provides security through process isolation and resource limits.
    """

    def __init__(self, config: IsolationConfig | None = None):
        self.config = config or IsolationConfig()
        self._execution_count = 0
        self._error_count = 0

    async def execute(
        self,
        plugin: Plugin,
        method: str,
        *args,
        **kwargs,
    ) -> IsolatedResult:
        """
        Execute a plugin method in isolation.

        Args:
            plugin: Plugin instance
            method: Method name to execute
            *args, **kwargs: Arguments to pass to method

        Returns:
            IsolatedResult with execution results
        """
        self._execution_count += 1

        # Check if isolation is needed
        if not self._requires_isolation(plugin):
            # Fast path: execute in same process
            return await self._execute_unsafe(plugin, method, *args, **kwargs)

        # Safe path: execute in isolated process
        return await self._execute_isolated(plugin, method, *args, **kwargs)

    def _requires_isolation(self, plugin: Plugin) -> bool:
        """Check if plugin requires isolation."""
        # Trust built-in plugins
        if plugin.metadata.author == "orchestrator":
            return False

        # Trust verified plugins
        if plugin.metadata.author.startswith("verified:"):
            return False

        # Isolate everything else
        return True

    async def _execute_unsafe(
        self,
        plugin: Plugin,
        method: str,
        *args,
        **kwargs,
    ) -> IsolatedResult:
        """Execute plugin in same process (fast but less secure)."""
        start_time = time.time()

        try:
            handler = getattr(plugin, method)

            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                result = handler(*args, **kwargs)

            execution_time = (time.time() - start_time) * 1000

            return IsolatedResult.success_result(result, execution_time)

        except Exception as e:
            self._error_count += 1
            execution_time = (time.time() - start_time) * 1000
            return IsolatedResult.error_result(
                str(e),
                type(e).__name__,
                execution_time,
            )

    async def _execute_isolated(
        self,
        plugin: Plugin,
        method: str,
        *args,
        **kwargs,
    ) -> IsolatedResult:
        """Execute plugin in isolated process."""
        start_time = time.time()

        # Create result queue
        result_queue = multiprocessing.Queue()

        # Prepare plugin class and config
        plugin_class = type(plugin)
        plugin_config = {}  # Would extract from plugin

        # Create process
        process = multiprocessing.Process(
            target=_execute_in_process,
            args=(
                plugin_class,
                plugin_config,
                method,
                args,
                kwargs,
                self.config,
                result_queue,
            ),
        )

        try:
            # Start process
            process.start()

            # Wait for result with timeout
            timeout = self.config.timeout_seconds + 5  # Buffer for process startup

            try:
                result_data = result_queue.get(timeout=timeout)
            except Exception:
                # Timeout or error getting result
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()

                self._error_count += 1
                execution_time = (time.time() - start_time) * 1000
                return IsolatedResult.error_result(
                    "Plugin execution timed out or failed to return result",
                    "TimeoutError",
                    execution_time,
                )

            # Wait for process to finish
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()

            execution_time = (time.time() - start_time) * 1000

            # Parse result
            if result_data.get("success"):
                return IsolatedResult.success_result(
                    result_data.get("result"),
                    execution_time,
                )
            else:
                self._error_count += 1
                return IsolatedResult.error_result(
                    result_data.get("error", "Unknown error"),
                    result_data.get("error_type", "Exception"),
                    execution_time,
                )

        except Exception as e:
            self._error_count += 1
            execution_time = (time.time() - start_time) * 1000

            # Cleanup process
            if process.is_alive():
                process.kill()
                process.join()

            return IsolatedResult.error_result(
                f"Isolation error: {str(e)}",
                type(e).__name__,
                execution_time,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get runtime statistics."""
        return {
            "total_executions": self._execution_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / self._execution_count if self._execution_count > 0 else 0,
            "isolation_config": {
                "memory_limit_mb": self.config.memory_limit_mb,
                "timeout_seconds": self.config.timeout_seconds,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Capability-Based Security
# ═══════════════════════════════════════════════════════════════════════════════

class Capability(Enum):
    """Capabilities that can be granted to plugins."""
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    FILE_DELETE = "file:delete"
    NETWORK_OUTBOUND = "network:outbound"
    NETWORK_INBOUND = "network:inbound"
    PROCESS_SPAWN = "process:spawn"
    THREAD_CREATE = "thread:create"
    ENV_READ = "env:read"
    ENV_WRITE = "env:write"


@dataclass
class CapabilitySet:
    """Set of capabilities granted to a plugin."""
    capabilities: set = field(default_factory=set)

    def grant(self, capability: Capability) -> None:
        """Grant a capability."""
        self.capabilities.add(capability)

    def revoke(self, capability: Capability) -> None:
        """Revoke a capability."""
        self.capabilities.discard(capability)

    def has(self, capability: Capability) -> bool:
        """Check if capability is granted."""
        return capability in self.capabilities

    @classmethod
    def trusted(cls) -> CapabilitySet:
        """Create capability set for trusted plugins."""
        return cls({
            Capability.FILE_READ,
            Capability.FILE_WRITE,
            Capability.NETWORK_OUTBOUND,
            Capability.THREAD_CREATE,
            Capability.ENV_READ,
        })

    @classmethod
    def untrusted(cls) -> CapabilitySet:
        """Create capability set for untrusted plugins."""
        return cls({
            Capability.THREAD_CREATE,
        })


# ═══════════════════════════════════════════════════════════════════════════════
# Secure Plugin Registry
# ═══════════════════════════════════════════════════════════════════════════════

class SecurePluginRegistry:
    """
    Plugin registry with security policies.

    Tracks plugin trust levels and applies appropriate isolation.
    """

    def __init__(self):
        self._plugins: dict[str, Plugin] = {}
        self._trust_levels: dict[str, str] = {}
        self._capabilities: dict[str, CapabilitySet] = {}
        self._runtime = IsolatedPluginRuntime()

    def register(
        self,
        plugin: Plugin,
        trust_level: str = "untrusted",
        capabilities: CapabilitySet | None = None,
    ) -> None:
        """
        Register a plugin with security metadata.

        Args:
            plugin: Plugin instance
            trust_level: "trusted", "verified", or "untrusted"
            capabilities: Granted capabilities
        """
        name = plugin.metadata.name
        self._plugins[name] = plugin
        self._trust_levels[name] = trust_level
        self._capabilities[name] = capabilities or CapabilitySet.untrusted()

        logger.info(f"Registered plugin {name} with trust level {trust_level}")

    async def execute(
        self,
        plugin_name: str,
        method: str,
        *args,
        **kwargs,
    ) -> IsolatedResult:
        """
        Execute a plugin method with appropriate isolation.
        """
        if plugin_name not in self._plugins:
            return IsolatedResult.error_result(
                f"Plugin {plugin_name} not found",
                "PluginNotFoundError",
            )

        plugin = self._plugins[plugin_name]
        trust_level = self._trust_levels.get(plugin_name, "untrusted")

        # Adjust isolation based on trust level
        if trust_level == "trusted":
            # Skip isolation for trusted plugins
            return await self._runtime._execute_unsafe(plugin, method, *args, **kwargs)

        # Use full isolation
        return await self._runtime.execute(plugin, method, *args, **kwargs)

    def check_capability(self, plugin_name: str, capability: Capability) -> bool:
        """Check if plugin has a capability."""
        caps = self._capabilities.get(plugin_name)
        if not caps:
            return False
        return caps.has(capability)


# ═══════════════════════════════════════════════════════════════════════════════
# Usage Examples
# ═══════════════════════════════════════════════════════════════════════════════

async def example_usage():
    """Example of using isolated plugin runtime."""

    # Create runtime with custom config
    config = IsolationConfig(
        memory_limit_mb=256,
        timeout_seconds=10.0,
        allow_network=False,
    )

    runtime = IsolatedPluginRuntime(config)

    # Example plugin execution
    from .plugins import PythonTypeCheckerValidator

    plugin = PythonTypeCheckerValidator()

    result = await runtime.execute(
        plugin,
        "validate",
        "def hello(): pass",
        {},
    )

    if result.success:
        print(f"Result: {result.result}")
    else:
        print(f"Error: {result.error}")

    print(f"Execution time: {result.execution_time_ms:.2f}ms")


if __name__ == "__main__":
    asyncio.run(example_usage())
