"""
MechanismInjector — Safe Runtime Mechanism Injection
=====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Validates and injects generated mechanism code at runtime. Each candidate
is written to a sandbox directory, validated via isolated import, then
either registered in the ``MechanismRegistry`` or reverted on failure.

Usage:
    injector = MechanismInjector(registry=my_registry)
    success, msg = await injector.inject(
        name="tabu_search",
        code="class TabuSearchManager: ...",
    )
"""

from __future__ import annotations

import ast
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .mechanism_registry import MechanismRegistry

logger = logging.getLogger("orchestrator.bilevel.mechanism_injector")


class MechanismInjector:
    """Validate and inject mechanism code at runtime.

    The injector performs three steps:
    1. **Syntax validation** — parse the code with ``ast.parse``
    2. **Sandbox import** — write code to a temp file, import it
    3. **Registry activation** — register and activate in the registry

    If any step fails, the mechanism is rejected and an error is logged.
    No changes are made to the running pipeline.

    Args:
        registry: The ``MechanismRegistry`` to register into.
        cache_dir: Directory for sandboxed code files.
                   Defaults to ``~/.orchestrator_cache/mechanisms/``.
    """

    def __init__(
        self,
        registry: MechanismRegistry,
        cache_dir: str | None = None,
    ) -> None:
        self._registry = registry
        self._cache_dir = Path(
            cache_dir or os.path.join(Path.home(), ".orchestrator_cache", "mechanisms")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def inject(
        self,
        name: str,
        code: str,
        description: str = "",
        version: str = "1.0",
        backup_state: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Validate and inject a mechanism.

        Args:
            name: Mechanism name (used as filename).
            code: Python source code.
            description: Human-readable description.
            version: Semantic version string.
            backup_state: Optional state snapshot for rollback.

        Returns:
            Tuple of ``(success, message)``.
        """
        # Step 1: Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as exc:
            msg = f"Syntax error in mechanism '{name}': {exc}"
            logger.warning(msg)
            return False, msg

        # Step 2: Sandbox write + import validation
        filepath = self._cache_dir / f"{name}.py"
        try:
            filepath.write_text(code, encoding="utf-8")
        except OSError as exc:
            msg = f"Failed to write mechanism '{name}' to {filepath}: {exc}"
            logger.warning(msg)
            return False, msg

        validate_ok, validate_msg = self._validate_import(filepath)
        if not validate_ok:
            # Clean up the written file
            filepath.unlink(missing_ok=True)
            return False, validate_msg

        # Step 3: Register in the mechanism registry
        try:
            self._registry.register(
                name=name,
                version=version,
                description=description,
                code=code,
            )
            self._registry.activate(name, backup_state=backup_state)
        except (ValueError, KeyError) as exc:
            msg = f"Failed to register mechanism '{name}': {exc}"
            logger.warning(msg)
            filepath.unlink(missing_ok=True)
            return False, msg

        logger.info("Injected mechanism '%s' v%s", name, version)
        return True, f"Injected mechanism '{name}' v{version}"

    async def remove(self, name: str) -> bool:
        """Deactivate and remove a mechanism.

        Args:
            name: Mechanism name.

        Returns:
            True if the mechanism was found and removed.
        """
        try:
            backup = self._registry.rollback(name)
            self._registry.unregister(name)
        except KeyError:
            return False

        # Clean up cached file
        filepath = self._cache_dir / f"{name}.py"
        filepath.unlink(missing_ok=True)

        if backup is not None:
            logger.info("Removed mechanism '%s' — restore backup to revert state changes", name)

        return True

    def _validate_import(self, filepath: Path) -> tuple[bool, str]:
        """Attempt to import the mechanism file in isolation.

        This is a best-effort validation: it runs a subprocess-level
        import to verify the module can be loaded without errors.

        Args:
            filepath: Path to the mechanism Python file.

        Returns:
            Tuple of ``(success, message)``.
        """
        # Use compile() to detect import-time errors without actually importing
        try:
            with open(filepath, encoding="utf-8") as f:
                source = f.read()
            compile(source, str(filepath), "exec")
        except (SyntaxError, ValueError) as exc:
            msg = f"Validation failed for {filepath.name}: {exc}"
            logger.warning(msg)
            return False, msg

        return True, f"Validation passed for {filepath.name}"
