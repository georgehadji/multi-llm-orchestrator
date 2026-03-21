"""
Sandbox — Secure code execution sandbox
=======================================
Module for securely executing untrusted code in isolated environments.

Pattern: Proxy
Async: Yes — for I/O-bound operations
Layer: L3 Agents

Usage:
    from orchestrator.sandbox import Sandbox
    sandbox = Sandbox()
    result = await sandbox.execute_code("python", "print('Hello, world!')")
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("orchestrator.sandbox")


@dataclass
class ExecutionResult:
    """Represents the result of code execution."""
    
    success: bool
    output: str
    error: str
    execution_time: float
    resources_used: Dict[str, Any]  # CPU, memory, etc.


class Sandbox:
    """Securely executes untrusted code in isolated environments."""

    def __init__(self, timeout: float = 30.0, max_memory_mb: int = 100, 
                 allowed_languages: Optional[List[str]] = None):
        """Initialize the sandbox."""
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allowed_languages = allowed_languages or ["python", "javascript", "bash"]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="orchestrator_sandbox_"))
        self.execution_counter = 0
        self.max_execution_time = timeout
        self.resource_limits = {
            "cpu_time": timeout,
            "memory_mb": max_memory_mb,
            "disk_mb": 10,
            "network": False  # By default, no network access
        }
    
    async def execute_code(self, language: str, code: str, 
                          inputs: Optional[List[str]] = None,
                          resource_limits: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute code in a secure sandbox.
        
        Args:
            language: Programming language of the code
            code: The code to execute
            inputs: List of inputs to provide to the program
            resource_limits: Custom resource limits for this execution
            
        Returns:
            ExecutionResult: Result of the code execution
        """
        if language not in self.allowed_languages:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Language {language} not allowed in sandbox",
                execution_time=0.0,
                resources_used={}
            )
        
        # Use provided limits or default
        limits = resource_limits or self.resource_limits
        
        # Create a unique execution environment
        exec_id = f"exec_{self.execution_counter}_{int(time.time())}"
        self.execution_counter += 1
        
        # Create execution directory
        exec_dir = self.temp_dir / exec_id
        exec_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write code to a temporary file
            code_file = await self._write_code_file(exec_dir, language, code)
            
            # Prepare inputs if provided
            input_file = None
            if inputs:
                input_file = exec_dir / "input.txt"
                with open(input_file, 'w') as f:
                    f.write("\n".join(inputs))
            
            # Execute the code with resource limits
            start_time = time.time()
            stdout, stderr, exit_code = await self._execute_with_limits(
                code_file, language, input_file, limits
            )
            execution_time = time.time() - start_time
            
            # Check if execution was successful
            success = exit_code == 0
            
            # Get resource usage
            resources_used = await self._get_resource_usage(exec_dir)
            
            return ExecutionResult(
                success=success,
                output=stdout,
                error=stderr,
                execution_time=execution_time,
                resources_used=resources_used
            )
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                resources_used={}
            )
        finally:
            # Cleanup execution directory
            await self._cleanup_exec_dir(exec_dir)
    
    async def _write_code_file(self, exec_dir: Path, language: str, code: str) -> Path:
        """Write code to a file with the appropriate extension."""
        ext_map = {
            "python": ".py",
            "javascript": ".js",
            "bash": ".sh",
            "go": ".go",
            "rust": ".rs",
            "java": ".java",
            "c": ".c",
            "cpp": ".cpp"
        }
        
        ext = ext_map.get(language, ".txt")
        code_file = exec_dir / f"code{ext}"
        
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return code_file
    
    async def _execute_with_limits(self, code_file: Path, language: str, 
                                   input_file: Optional[Path], 
                                   limits: Dict[str, Any]) -> Tuple[str, str, int]:
        """Execute code with resource limits."""
        # Determine the command to run based on language
        cmd = self._get_execution_command(code_file, language)
        
        # Create input stream if input file is provided
        stdin = None
        if input_file:
            stdin = open(input_file, 'r')
        
        try:
            # Run the command with timeout
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=stdin,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=limits["cpu_time"]
            )
            
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=limits["cpu_time"]
            )
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            return stdout_str, stderr_str, proc.returncode
        except asyncio.TimeoutError:
            # Terminate the process if it times out
            if proc:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    proc.kill()  # Force kill if it doesn't terminate gracefully
            
            return "", "Execution timed out", -1
        finally:
            if stdin:
                stdin.close()
    
    def _get_execution_command(self, code_file: Path, language: str) -> List[str]:
        """Get the command to execute code in the specified language."""
        if language == "python":
            return ["python", str(code_file)]
        elif language == "javascript":
            return ["node", str(code_file)]
        elif language == "bash":
            return ["bash", str(code_file)]
        elif language == "go":
            return ["go", "run", str(code_file)]
        elif language == "rust":
            # For Rust, we'd typically compile first, but for simplicity we'll use rust-script
            return ["rust-script", str(code_file)]
        elif language == "java":
            # For Java, we'd typically compile first
            class_name = code_file.stem
            compile_cmd = ["javac", str(code_file)]
            # Run compilation in a subprocess
            import subprocess
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Compilation failed: {result.stderr}")
            return ["java", "-cp", str(code_file.parent), class_name]
        elif language == "c":
            exe_file = code_file.with_suffix('.exe')
            compile_cmd = ["gcc", str(code_file), "-o", str(exe_file)]
            import subprocess
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Compilation failed: {result.stderr}")
            return [str(exe_file)]
        elif language == "cpp":
            exe_file = code_file.with_suffix('.exe')
            compile_cmd = ["g++", str(code_file), "-o", str(exe_file)]
            import subprocess
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Compilation failed: {result.stderr}")
            return [str(exe_file)]
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    async def _get_resource_usage(self, exec_dir: Path) -> Dict[str, Any]:
        """Get resource usage for the execution."""
        # For now, just return a basic resource usage report
        # In a more advanced implementation, we would measure actual resource usage
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(exec_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        return {
            "disk_usage_mb": round(total_size / (1024 * 1024), 2),
            "peak_memory_mb": 0,  # Placeholder - would need actual measurement
            "cpu_time_seconds": 0  # Placeholder - would need actual measurement
        }
    
    async def _cleanup_exec_dir(self, exec_dir: Path):
        """Clean up the execution directory."""
        import shutil
        try:
            shutil.rmtree(exec_dir)
        except Exception as e:
            logger.error(f"Failed to clean up execution directory {exec_dir}: {e}")
    
    async def validate_code(self, language: str, code: str) -> Tuple[bool, List[str]]:
        """
        Validate code for security issues before execution.
        
        Args:
            language: Programming language of the code
            code: The code to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues_found)
        """
        issues = []
        
        # Check for dangerous patterns based on language
        if language == "python":
            dangerous_patterns = [
                "import os", "import sys", "import subprocess", 
                "import shutil", "import socket", "import urllib",
                "__import__", "eval(", "exec(", "compile(",
                "open(", "file("
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code:
                    issues.append(f"Dangerous pattern found: {pattern}")
        
        elif language == "javascript":
            dangerous_patterns = [
                "require('child_process')", "require('fs')", "require('net')",
                "require('tls')", "require('dgram')", "require('dns')",
                "eval(", "new Function(", "setTimeout(", "setInterval("
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code:
                    issues.append(f"Dangerous pattern found: {pattern}")
        
        elif language == "bash":
            dangerous_patterns = [
                "rm -rf", "chmod", "chown", "wget", "curl",
                "> /dev/tcp/", "nc ", "netcat", "ssh ", "scp "
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code.lower():
                    issues.append(f"Dangerous pattern found: {pattern}")
        
        # Check for excessive length (potential DoS)
        if len(code) > 10000:  # 10KB limit
            issues.append("Code exceeds length limit (potential DoS)")
        
        return len(issues) == 0, issues
    
    async def execute_safe_code(self, language: str, code: str, 
                               inputs: Optional[List[str]] = None) -> ExecutionResult:
        """
        Execute code after validating it for security issues.
        
        Args:
            language: Programming language of the code
            code: The code to execute
            inputs: List of inputs to provide to the program
            
        Returns:
            ExecutionResult: Result of the code execution
        """
        is_valid, issues = await self.validate_code(language, code)
        
        if not is_valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Code validation failed: {'; '.join(issues)}",
                execution_time=0.0,
                resources_used={}
            )
        
        return await self.execute_code(language, code, inputs)
    
    def get_allowed_languages(self) -> List[str]:
        """Get the list of allowed languages."""
        return self.allowed_languages
    
    def add_allowed_language(self, language: str):
        """Add a language to the allowed list."""
        if language not in self.allowed_languages:
            self.allowed_languages.append(language)
            logger.info(f"Added {language} to allowed languages")
    
    def remove_allowed_language(self, language: str):
        """Remove a language from the allowed list."""
        if language in self.allowed_languages:
            self.allowed_languages.remove(language)
            logger.info(f"Removed {language} from allowed languages")
    
    async def get_sandbox_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the sandbox.
        
        Returns:
            Dict with sandbox statistics
        """
        # Calculate disk usage of temp directory
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.temp_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        return {
            "execution_count": self.execution_counter,
            "allowed_languages": self.allowed_languages,
            "temp_dir": str(self.temp_dir),
            "disk_usage_mb": round(total_size / (1024 * 1024), 2),
            "timeout_setting": self.timeout,
            "max_memory_mb": self.max_memory_mb
        }
    
    async def cleanup_temp_files(self):
        """Clean up temporary files created by the sandbox."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = Path(tempfile.mkdtemp(prefix="orchestrator_sandbox_"))
            logger.info("Cleaned up sandbox temporary files")
        except Exception as e:
            logger.error(f"Failed to clean up sandbox temp files: {e}")
    
    async def execute_multiple_codes(self, executions: List[Dict[str, str]]) -> List[ExecutionResult]:
        """
        Execute multiple code snippets in parallel.
        
        Args:
            executions: List of dicts with 'language' and 'code' keys
            
        Returns:
            List of ExecutionResults
        """
        tasks = []
        for exec_data in executions:
            task = self.execute_safe_code(
                exec_data["language"], 
                exec_data["code"], 
                exec_data.get("inputs")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred during execution
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    ExecutionResult(
                        success=False,
                        output="",
                        error=f"Execution error: {str(result)}",
                        execution_time=0.0,
                        resources_used={}
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results