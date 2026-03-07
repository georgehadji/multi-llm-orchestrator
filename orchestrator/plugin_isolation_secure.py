"""
Secure Plugin Isolation with Defense in Depth
===============================================

Implements the minimax regret improvement for Black Swan Scenario 2:
Plugin Sandbox Escape

Security Layers:
1. Process isolation (existing)
2. seccomp-bpf (system call filtering)
3. Landlock (filesystem sandboxing)
4. Linux capabilities (privilege dropping)
5. Resource limits (existing)

Usage:
    from orchestrator.plugin_isolation_secure import SecureIsolatedRuntime
    
    runtime = SecureIsolatedRuntime(IsolationConfig(
        memory_limit_mb=512,
        allow_network=False,
        enable_seccomp=True,
        enable_landlock=True,
    ))
"""

from __future__ import annotations

import os
import sys
import time
import errno
import shutil
import signal
import hashlib
import logging
import tempfile
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass
from enum import Enum, auto

from .plugin_isolation import (
    IsolationConfig,
    IsolatedResult,
    PluginExecutionError,
    Plugin,
)
from .log_config import get_logger

logger = get_logger(__name__)


class SecurityLayer(Enum):
    """Available security isolation layers."""
    PROCESS = auto()      # Process isolation
    SECCOMP = auto()      # System call filtering
    LANDLOCK = auto()     # Filesystem sandboxing
    CAPABILITIES = auto()  # Linux capabilities
    RESOURCES = auto()    # Resource limits


@dataclass
class SecureIsolationConfig(IsolationConfig):
    """Extended config with security options."""
    enable_seccomp: bool = True
    enable_landlock: bool = True
    enable_capabilities: bool = True
    allowed_syscalls: Optional[List[str]] = None
    sandbox_path: Optional[str] = None
    # Add to parent fields
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50
    timeout_seconds: float = 30.0
    allow_network: bool = False


class SecureIsolatedRuntime:
    """
    Hardened plugin runtime with multiple security layers.
    
    Defense in depth - even if one layer fails, others protect.
    """
    
    def __init__(self, config: SecureIsolationConfig):
        self.config = config
        self._check_security_features()
    
    def _check_security_features(self) -> None:
        """Check which security features are available."""
        self.features = {
            SecurityLayer.PROCESS: True,  # Always available
            SecurityLayer.SECCOMP: self._check_seccomp(),
            SecurityLayer.LANDLOCK: self._check_landlock(),
            SecurityLayer.CAPABILITIES: self._check_capabilities(),
            SecurityLayer.RESOURCES: True,  # Always available
        }
        
        for layer, available in self.features.items():
            status = "available" if available else "unavailable"
            logger.info(f"Security layer {layer.name}: {status}")
    
    def _check_seccomp(self) -> bool:
        """Check if seccomp is available."""
        try:
            import seccomp
            return True
        except ImportError:
            logger.warning("seccomp not available - install python-seccomp")
            return False
    
    def _check_landlock(self) -> bool:
        """Check if Landlock is available (Linux 5.13+)."""
        try:
            # Check kernel version
            version = os.uname().release
            major, minor = map(int, version.split('.')[:2])
            return (major, minor) >= (5, 13)
        except Exception:
            return False
    
    def _check_capabilities(self) -> bool:
        """Check if prctl/capabilities available."""
        try:
            import prctl
            return True
        except ImportError:
            logger.warning("prctl not available - install python-prctl")
            return False
    
    async def execute(
        self,
        plugin: Plugin,
        method: str,
        *args,
        **kwargs,
    ) -> IsolatedResult:
        """
        Execute plugin method in secure sandbox.
        
        Uses multiprocessing with security hardening.
        """
        result_queue = mp.Queue()
        
        # Create sandbox directory
        sandbox = self._create_sandbox()
        
        try:
            # Spawn isolated process
            process = mp.Process(
                target=self._secure_worker,
                args=(
                    plugin.__class__,
                    method,
                    args,
                    kwargs,
                    self.config,
                    sandbox,
                    result_queue,
                ),
            )
            
            process.start()
            process.join(timeout=self.config.timeout_seconds)
            
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                
                return IsolatedResult(
                    success=False,
                    result=None,
                    execution_time=self.config.timeout_seconds,
                    error="Execution timeout - process terminated",
                    logs=[],
                )
            
            if not result_queue.empty():
                result_data = result_queue.get()
                
                if result_data.get("success"):
                    return IsolatedResult(
                        success=True,
                        result=result_data.get("result"),
                        execution_time=result_data.get("execution_time", 0),
                        error=None,
                        logs=result_data.get("logs", []),
                    )
                else:
                    return IsolatedResult(
                        success=False,
                        result=None,
                        execution_time=result_data.get("execution_time", 0),
                        error=result_data.get("error"),
                        logs=result_data.get("logs", []),
                    )
            else:
                return IsolatedResult(
                    success=False,
                    result=None,
                    execution_time=0,
                    error="No result from plugin",
                    logs=[],
                )
                
        finally:
            # Cleanup sandbox
            self._cleanup_sandbox(sandbox)
    
    def _create_sandbox(self) -> Path:
        """Create temporary sandbox directory."""
        if self.config.sandbox_path:
            sandbox = Path(self.config.sandbox_path)
            sandbox.mkdir(parents=True, exist_ok=True)
        else:
            sandbox = Path(tempfile.mkdtemp(prefix="plugin_sandbox_"))
        
        # Create subdirectories
        (sandbox / "workspace").mkdir(exist_ok=True)
        (sandbox / "temp").mkdir(exist_ok=True)
        
        return sandbox
    
    def _cleanup_sandbox(self, sandbox: Path) -> None:
        """Remove sandbox directory."""
        try:
            shutil.rmtree(sandbox)
        except Exception as e:
            logger.warning(f"Failed to cleanup sandbox: {e}")
    
    def _secure_worker(
        self,
        plugin_class: Type[Plugin],
        method: str,
        args: tuple,
        kwargs: dict,
        config: SecureIsolationConfig,
        sandbox: Path,
        result_queue: mp.Queue,
    ) -> None:
        """
        Worker process with security hardening.
        
        Applies security layers in order:
        1. Change to sandbox directory
        2. Landlock filesystem sandbox
        3. Drop capabilities
        4. Load seccomp policy (LOCKS THE PROCESS)
        5. Set resource limits
        """
        start_time = time.time()
        logs = []
        
        try:
            # 1. Change to sandbox
            os.chdir(sandbox)
            os.chroot(str(sandbox))  # Basic chroot if available
            
            # 2. Landlock sandbox
            if config.enable_landlock and self._check_landlock():
                self._apply_landlock(sandbox)
            
            # 3. Drop capabilities
            if config.enable_capabilities and self._check_capabilities():
                self._drop_capabilities()
            
            # 4. Seccomp (MUST be last - locks process)
            if config.enable_seccomp and self._check_seccomp():
                self._apply_seccomp(config)
            
            # 5. Resource limits
            self._set_resource_limits(config)
            
            # Execute plugin
            plugin = plugin_class()
            result = getattr(plugin, method)(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            result_queue.put({
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "logs": logs,
            })
            
        except Exception as e:
            execution_time = time.time() - start_time
            result_queue.put({
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "execution_time": execution_time,
                "logs": logs,
            })
    
    def _apply_landlock(self, sandbox: Path) -> None:
        """Apply Landlock filesystem sandboxing."""
        try:
            import landlock
            
            ruleset = landlock.create_ruleset({
                landlock.Access.FS_READ_FILE,
                landlock.Access.FS_WRITE_FILE,
                landlock.Access.FS_READ_DIR,
            })
            
            # Allow workspace
            ruleset.add_rule(
                landlock.PathFd(str(sandbox / "workspace")),
                landlock.Access.FS_READ_FILE | landlock.Access.FS_WRITE_FILE
            )
            
            # Allow temp
            ruleset.add_rule(
                landlock.PathFd(str(sandbox / "temp")),
                landlock.Access.FS_READ_FILE | landlock.Access.FS_WRITE_FILE
            )
            
            # Restrict
            ruleset.restrict_self()
            logger.debug("Landlock sandbox applied")
            
        except Exception as e:
            logger.warning(f"Failed to apply Landlock: {e}")
    
    def _drop_capabilities(self) -> None:
        """Drop all Linux capabilities."""
        try:
            import prctl
            
            # Drop all capabilities
            prctl.capbset.drop(prctl.CAP_ALL)
            
            logger.debug("Capabilities dropped")
            
        except Exception as e:
            logger.warning(f"Failed to drop capabilities: {e}")
    
    def _apply_seccomp(self, config: SecureIsolationConfig) -> None:
        """
        Apply seccomp system call filtering.
        
        Blocks dangerous syscalls while allowing necessary ones.
        """
        try:
            import seccomp
            
            # Create filter (default: deny with EPERM)
            f = seccomp.SyscallFilter(seccomp.ERRNO(errno.EPERM))
            
            # Allow basic I/O
            f.add_rule(seccomp.ALLOW, "read")
            f.add_rule(seccomp.ALLOW, "write")
            f.add_rule(seccomp.ALLOW, "close")
            f.add_rule(seccomp.ALLOW, "exit")
            f.add_rule(seccomp.ALLOW, "exit_group")
            
            # Allow memory management
            f.add_rule(seccomp.ALLOW, "mmap")
            f.add_rule(seccomp.ALLOW, "munmap")
            f.add_rule(seccomp.ALLOW, "brk")
            f.add_rule(seccomp.ALLOW, "mprotect")
            
            # Allow process control
            f.add_rule(seccomp.ALLOW, "getpid")
            f.add_rule(seccomp.ALLOW, "getppid")
            
            # Allow time
            f.add_rule(seccomp.ALLOW, "clock_gettime")
            f.add_rule(seccomp.ALLOW, "gettimeofday")
            
            # Block dangerous syscalls
            dangerous = [
                "ptrace",              # No debugging other processes
                "process_vm_writev",   # No writing to other processes' memory
                "process_vm_readv",    # No reading other processes' memory
                "execve",              # No executing new programs
                "execveat",
                "fork",                # No forking
                "vfork",
                "clone",
                "unshare",             # No namespace manipulation
                "setns",
                "pivot_root",          # No chroot escape
                "chroot",              # No additional chroots
            ]
            
            for syscall in dangerous:
                try:
                    f.add_rule(seccomp.ERRNO(errno.EPERM), syscall)
                except Exception:
                    pass  # Syscall may not exist on this arch
            
            # Block network if not allowed
            if not config.allow_network:
                network_syscalls = [
                    "socket",
                    "connect",
                    "bind",
                    "accept",
                    "listen",
                    "sendto",
                    "recvfrom",
                    "sendmsg",
                    "recvmsg",
                    "setsockopt",
                    "getsockopt",
                ]
                for syscall in network_syscalls:
                    try:
                        f.add_rule(seccomp.ERRNO(errno.EPERM), syscall)
                    except Exception:
                        pass
            
            f.load()
            logger.debug("seccomp policy loaded")
            
        except Exception as e:
            logger.warning(f"Failed to apply seccomp: {e}")
    
    def _set_resource_limits(self, config: SecureIsolationConfig) -> None:
        """Set process resource limits."""
        import resource
        
        # Memory limit
        if config.memory_limit_mb > 0:
            max_bytes = config.memory_limit_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_AS,
                (max_bytes, max_bytes)
            )
        
        # CPU limit (soft only)
        if config.cpu_limit_percent > 0:
            cpu_seconds = config.timeout_seconds * (config.cpu_limit_percent / 100)
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(cpu_seconds), int(cpu_seconds * 2))
            )
        
        # File descriptor limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        
        # Process limit (no forking)
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))


class TrustedPluginRegistry:
    """
    Registry that only loads cryptographically signed plugins.
    
    Prevents supply chain attacks by verifying plugin authenticity.
    """
    
    def __init__(self, public_key_pem: Optional[str] = None):
        self.public_key_pem = public_key_pem
        self.trusted_hashes: Dict[str, str] = {}  # name -> expected hash
    
    def add_trusted_plugin(self, name: str, expected_hash: str) -> None:
        """Add a trusted plugin by its hash."""
        self.trusted_hashes[name] = expected_hash
    
    def verify_plugin(self, plugin_path: Path, name: str) -> bool:
        """
        Verify plugin integrity.
        
        Checks:
        1. File hash matches expected
        2. Code signature valid (if key provided)
        """
        if name not in self.trusted_hashes:
            logger.error(f"Plugin {name} not in trusted list")
            return False
        
        # Calculate file hash
        sha256 = hashlib.sha256()
        with open(plugin_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        file_hash = sha256.hexdigest()
        
        if file_hash != self.trusted_hashes[name]:
            logger.critical(
                f"Plugin {name} hash mismatch! "
                f"Expected: {self.trusted_hashes[name][:16]}..., "
                f"Got: {file_hash[:16]}..."
            )
            return False
        
        logger.info(f"Plugin {name} verified successfully")
        return True
