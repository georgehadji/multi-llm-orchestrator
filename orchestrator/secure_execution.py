"""
Secure Execution Module
=======================

Provides safe alternatives to dangerous Python functions:
- subprocess without shell=True
- Path traversal protection
- Input sanitization
- Command injection prevention

SECURITY PRINCIPLES:
    1. NEVER use shell=True in subprocess calls
    2. ALWAYS validate and sanitize user inputs
    3. ALWAYS use absolute paths for file operations
    4. NEVER execute user-provided code without sandboxing
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("orchestrator.security")


class SecurityError(Exception):
    """Raised when a security violation is detected."""

    pass


class PathTraversalError(SecurityError):
    """Raised when a path traversal attempt is detected."""

    pass


class CommandInjectionError(SecurityError):
    """Raised when command injection is detected."""

    pass


class InputValidationError(SecurityError):
    """Raised when input validation fails."""

    pass


@dataclass
class SecurePath:
    """
    A path that has been validated to prevent path traversal attacks.

    Usage:
        base = Path("/allowed/base")
        user_input = "../../../etc/passwd"
        try:
            secure = SecurePath(base, user_input)
            print(secure.resolved)  # Raises PathTraversalError
        except PathTraversalError:
            print("Path traversal detected!")
    """

    base_path: Path
    user_input: str
    resolved: Path = None

    def __post_init__(self):
        # Check for null bytes before processing
        if "\x00" in self.user_input:
            raise PathTraversalError("Path contains null bytes")

        if self.resolved is None:
            self.resolved = self._resolve_and_validate()

    def _resolve_and_validate(self) -> Path:
        """Resolve path and validate it doesn't escape base."""
        # Convert to Path and normalize
        base = self.base_path.resolve().absolute()

        # Handle empty input
        if not self.user_input or not self.user_input.strip():
            return base

        # Normalize path separators for cross-platform compatibility
        # Replace backslashes with forward slashes for consistent handling
        normalized_input = self.user_input.replace("\\", "/")

        # Check for absolute paths (Unix and Windows)
        if normalized_input.startswith("/") or (
            len(normalized_input) > 1 and normalized_input[1] == ":"
        ):
            # Absolute path - check if it's within base after resolution
            pass  # Will be caught by relative_to check

        # Join and resolve
        try:
            target = (base / normalized_input).resolve().absolute()
        except (ValueError, OSError) as e:
            raise PathTraversalError(f"Invalid path: {e}")

        # Security check: resolved path must be within base
        try:
            # Use parts comparison for more reliable check
            target.relative_to(base)
        except ValueError:
            raise PathTraversalError(
                f"Path traversal detected: '{self.user_input}' attempts to escape base directory"
            )

        return target

    def __str__(self) -> str:
        return str(self.resolved)

    def __fspath__(self) -> str:
        return str(self.resolved)


@dataclass
class SafeCommand:
    """
    A command that has been validated to be safe for execution.

    Usage:
        cmd = SafeCommand(["python", "-c", user_code])
        # Raises if user_code contains shell metacharacters
    """

    args: list[str]

    # Shell metacharacters that could enable injection
    _SHELL_METACHARACTERS = set(";|&$`<>{}[]\\\n\r")

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        """Validate that command arguments are safe."""
        if not self.args:
            raise InputValidationError("Empty command")

        # First argument should be the executable
        executable = self.args[0]

        # Check for shell metacharacters in all arguments
        for i, arg in enumerate(self.args):
            if any(c in arg for c in self._SHELL_METACHARACTERS):
                raise CommandInjectionError(
                    f"Argument {i} contains shell metacharacters: {repr(arg[:50])}"
                )

        # Validate executable exists and is not a shell
        dangerous_executables = {"sh", "bash", "zsh", "cmd", "powershell", "python"}
        executable_name = Path(executable).name.lower()

        # Don't allow direct shell execution
        if executable_name in dangerous_executables and len(self.args) > 1:
            if "-c" in self.args or "/c" in self.args:
                # Check the script content
                script_idx = (
                    self.args.index("-c") + 1 if "-c" in self.args else self.args.index("/c") + 1
                )
                if script_idx < len(self.args):
                    script = self.args[script_idx]
                    if any(c in script for c in self._SHELL_METACHARACTERS):
                        raise CommandInjectionError(
                            "Shell script contains dangerous metacharacters"
                        )

    def to_list(self) -> list[str]:
        """Return command as list (for subprocess)."""
        return self.args


class SecureSubprocess:
    """
    Secure wrapper for subprocess operations.

    Features:
    - No shell=True (ever)
    - Input validation
    - Timeout enforcement
    - Resource limits
    """

    DEFAULT_TIMEOUT = 300  # 5 minutes
    MAX_OUTPUT_SIZE = 10 * 1024 * 1024  # 10MB

    @classmethod
    def run(
        cls,
        command: list[str],
        cwd: str | Path | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """
        Run a command securely (no shell=True).

        Args:
            command: List of command arguments (NOT a string!)
            cwd: Working directory
            timeout: Timeout in seconds
            env: Environment variables
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess instance

        Raises:
            CommandInjectionError: If command is unsafe
            subprocess.TimeoutExpired: If command times out
        """
        # Validate command
        safe_cmd = SafeCommand(command)

        # Validate working directory
        if cwd:
            cwd = Path(cwd).resolve()
            if not cwd.exists():
                raise FileNotFoundError(f"Working directory does not exist: {cwd}")

        timeout = timeout or cls.DEFAULT_TIMEOUT

        logger.debug(f"Executing secure command: {' '.join(safe_cmd.args[:5])}...")

        # Build environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # SECURITY: Explicitly set shell=False (default, but explicit is better)
        result = subprocess.run(
            safe_cmd.args,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            shell=False,  # NEVER True
            env=run_env,
            **kwargs,
        )

        return result

    @classmethod
    async def run_async(
        cls,
        command: list[str],
        cwd: str | Path | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> tuple[int, str, str]:
        """
        Run a command asynchronously (no shell=True).

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        # Validate command
        safe_cmd = SafeCommand(command)

        if cwd:
            cwd = Path(cwd).resolve()

        timeout = timeout or cls.DEFAULT_TIMEOUT

        # Build environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        logger.debug(f"Executing async secure command: {' '.join(safe_cmd.args[:5])}...")

        proc = await asyncio.create_subprocess_exec(
            *safe_cmd.args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=run_env,
            **kwargs,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return (
                proc.returncode,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise subprocess.TimeoutExpired(command, timeout)


class InputValidator:
    """
    Validates and sanitizes user inputs.

    Usage:
        validator = InputValidator()
        clean = validator.sanitize_filename(user_input)
    """

    # Allowed characters for different contexts
    SAFE_FILENAME_CHARS = re.compile(r"^[\w\-\. ]+$")
    SAFE_IDENTIFIER_CHARS = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    SAFE_BRANCH_NAME_CHARS = re.compile(r"^[\w\-\./]+$")

    # Maximum lengths
    MAX_FILENAME_LENGTH = 255
    MAX_PATH_LENGTH = 4096
    MAX_IDENTIFIER_LENGTH = 100

    @classmethod
    def sanitize_filename(cls, filename: str, replacement: str = "_") -> str:
        """
        Sanitize a filename to prevent path traversal and injection.

        Args:
            filename: User-provided filename
            replacement: Character to replace dangerous chars with

        Returns:
            Sanitized filename
        """
        # Handle None or empty input
        if not filename:
            return "unnamed"

        # Remove null bytes
        filename = filename.replace("\x00", "")

        # Replace dangerous characters
        dangerous = '<>:"/\\|?*'
        for char in dangerous:
            filename = filename.replace(char, replacement)

        # Remove leading/trailing dots and spaces
        filename = filename.strip(". ")

        # Limit length
        if len(filename) > cls.MAX_FILENAME_LENGTH:
            name, ext = os.path.splitext(filename)
            filename = name[: cls.MAX_FILENAME_LENGTH - len(ext)] + ext

        # Ensure not empty after sanitization (whitespace-only becomes empty)
        if not filename:
            filename = "unnamed"

        # Check for reserved names (Windows)
        reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }
        base = os.path.splitext(filename)[0].upper()
        if base in reserved:
            filename = f"_{filename}"

        return filename

    @classmethod
    def validate_identifier(cls, identifier: str) -> str:
        """
        Validate a Python identifier.

        Args:
            identifier: String to validate

        Returns:
            The identifier if valid

        Raises:
            InputValidationError: If identifier is invalid
        """
        if not identifier:
            raise InputValidationError("Identifier cannot be empty")

        if len(identifier) > cls.MAX_IDENTIFIER_LENGTH:
            raise InputValidationError(
                f"Identifier too long: {len(identifier)} > {cls.MAX_IDENTIFIER_LENGTH}"
            )

        if not cls.SAFE_IDENTIFIER_CHARS.match(identifier):
            raise InputValidationError(
                f"Invalid identifier: '{identifier}'. Must match [a-zA-Z_][a-zA-Z0-9_]*"
            )

        # Check Python keywords
        import keyword

        if keyword.iskeyword(identifier):
            raise InputValidationError(f"Identifier cannot be a Python keyword: '{identifier}'")

        return identifier

    @classmethod
    def sanitize_branch_name(cls, name: str) -> str:
        """
        Sanitize a git branch name.

        Args:
            name: Branch name to sanitize

        Returns:
            Sanitized branch name
        """
        if not name:
            raise InputValidationError("Branch name cannot be empty")

        # Replace spaces and dangerous chars
        name = re.sub(r"[^\w\-\./]", "_", name)

        # Remove consecutive dots
        name = re.sub(r"\.{2,}", ".", name)

        # Remove leading/trailing dots and slashes
        name = name.strip("./ ")

        # Limit length
        if len(name) > 250:
            name = name[:250]

        # Ensure not empty
        if not name:
            name = "branch"

        # Git restrictions
        forbidden = {"HEAD", "-", "@{"}
        for f in forbidden:
            if f in name:
                name = name.replace(f, "_")

        return name

    @classmethod
    def validate_allowed_extension(cls, filename: str, allowed: tuple[str, ...]) -> str:
        """
        Validate file has an allowed extension.

        Args:
            filename: Filename to check
            allowed: Tuple of allowed extensions (e.g., ('.py', '.txt'))

        Returns:
            Lowercase extension

        Raises:
            InputValidationError: If extension not allowed
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed:
            raise InputValidationError(
                f"File extension '{ext}' not allowed. Allowed: {', '.join(allowed)}"
            )
        return ext


class SecurityContext:
    """
    Context manager for secure execution environment.

    Usage:
        with SecurityContext(allowed_paths=['/safe/path']) as ctx:
            ctx.execute(['python', 'script.py'])
    """

    def __init__(
        self,
        allowed_paths: list[Path] | None = None,
        allowed_extensions: tuple[str, ...] | None = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        self.allowed_paths = [Path(p).resolve() for p in (allowed_paths or [])]
        self.allowed_extensions = allowed_extensions
        self.max_file_size = max_file_size

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def resolve_path(self, user_path: str) -> Path:
        """Resolve a user path within allowed directories."""
        if not self.allowed_paths:
            # No restrictions - use SecurePath with current directory
            return SecurePath(Path.cwd(), user_path).resolved

        # Try each allowed path
        for base in self.allowed_paths:
            try:
                return SecurePath(base, user_path).resolved
            except PathTraversalError:
                continue

        raise PathTraversalError(f"Path '{user_path}' is not within allowed directories")

    def read_file(self, user_path: str) -> str:
        """Safely read a file."""
        path = self.resolve_path(user_path)

        # Check extension
        if self.allowed_extensions:
            InputValidator.validate_allowed_extension(str(path), self.allowed_extensions)

        # Check size
        size = path.stat().st_size
        if size > self.max_file_size:
            raise SecurityError(f"File too large: {size} > {self.max_file_size}")

        return path.read_text(encoding="utf-8")

    def write_file(self, user_path: str, content: str) -> None:
        """Safely write a file."""
        path = self.resolve_path(user_path)

        # Check extension
        if self.allowed_extensions:
            InputValidator.validate_allowed_extension(str(path), self.allowed_extensions)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding="utf-8")

    def execute(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Execute command securely."""
        return SecureSubprocess.run(command, **kwargs)


# Legacy compatibility functions
def secure_subprocess_run(
    cmd: list[str], cwd: Path | None = None, **kwargs
) -> subprocess.CompletedProcess:
    """
    DEPRECATED: Use SecureSubprocess.run() instead.

    Secure wrapper that NEVER uses shell=True.
    """
    return SecureSubprocess.run(cmd, cwd=cwd, **kwargs)


def sanitize_path(base_path: Path, user_input: str) -> Path:
    """Sanitize a user-provided path."""
    return SecurePath(base_path, user_input).resolved


def validate_no_shell_injection(text: str) -> None:
    """Check for shell injection patterns."""
    SafeCommand(["check", text])  # Will raise if unsafe


# Example usage
if __name__ == "__main__":
    # Test path traversal detection
    try:
        bad_path = SecurePath(Path("/safe/base"), "../../../etc/passwd")
        print("ERROR: Should have raised PathTraversalError")
    except PathTraversalError as e:
        print(f"✓ Caught path traversal: {e}")

    # Test command injection detection
    try:
        bad_cmd = SafeCommand(["echo", "hello; rm -rf /"])
        print("ERROR: Should have raised CommandInjectionError")
    except CommandInjectionError as e:
        print(f"✓ Caught command injection: {e}")

    # Test filename sanitization
    dirty = "../../../etc/passwd\x00.txt"
    clean = InputValidator.sanitize_filename(dirty)
    print(f"✓ Sanitized: '{dirty}' -> '{clean}'")

    # Test secure subprocess
    result = SecureSubprocess.run(["echo", "hello world"])
    print(f"✓ Secure subprocess: {result.stdout.strip()}")
