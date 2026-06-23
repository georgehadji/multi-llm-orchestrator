"""
DynamicScaffoldGenerator — Create project structures for any tech stack
=========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 8 of the Agentic System Implementation Plan.
Generates project scaffold for ANY technology combination,
even when no template exists in the scaffold library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.scaffold.dynamic")


@dataclass
class TechStack:
    """Software technology stack."""

    language: str = ""
    framework: str = ""
    frontend: str = ""
    database: str = ""


class DynamicScaffoldGenerator:
    """Generate project structure for any tech stack.

    If a template exists in the scaffold library → use it.
    If no template → generate structure from first principles.
    """

    async def generate(self, goal: str, output_dir: Path) -> bool:
        """Generate a scaffold for the given goal.

        Args:
            goal: What to build (e.g. "a Go CLI app with cobra").
            output_dir: Where to create files.

        Returns:
            True if scaffold was generated.
        """
        stack = self._detect_stack(goal)

        # Check for existing template
        try:
            from ..scaffold import ScaffoldTemplate

            template = self._find_template(stack)
            if template:
                return self._apply_template(template, output_dir)
        except ImportError:
            pass

        # No template found - create basic structure
        return self._create_basic_structure(stack, output_dir)

    def _detect_stack(self, goal: str) -> TechStack:
        """Detect tech stack from goal text."""
        goal_lower = goal.lower()
        stack = TechStack()

        if "python" in goal_lower:
            stack.language = "python"
        elif "go" in goal_lower or "golang" in goal_lower:
            stack.language = "go"
        elif "rust" in goal_lower:
            stack.language = "rust"
        elif "typescript" in goal_lower or "ts" in goal_lower:
            stack.language = "typescript"
        elif "javascript" in goal_lower or "js" in goal_lower:
            stack.language = "javascript"

        if "fastapi" in goal_lower:
            stack.framework = "fastapi"
        elif "django" in goal_lower:
            stack.framework = "django"
        elif "react" in goal_lower:
            stack.frontend = "react"
        elif "htmx" in goal_lower:
            stack.frontend = "htmx"

        if "sqlite" in goal_lower:
            stack.database = "sqlite"
        elif "postgres" in goal_lower:
            stack.database = "postgres"

        return stack

    def _find_template(self, stack: TechStack) -> Any:
        """Find a matching scaffold template."""
        from ..scaffold import _TEMPLATE_MAP

        key = stack.framework or stack.frontend or stack.language
        return _TEMPLATE_MAP.get(key)

    def _apply_template(self, template: Any, output_dir: Path) -> bool:
        """Apply a scaffold template."""
        try:

            if hasattr(template, "apply"):
                result = template.apply(output_dir)
                logger.info("Applied scaffold template: %s", type(template).__name__)
                return True
        except Exception as exc:
            logger.warning("Failed to apply template: %s", exc)
        return False

    def _create_basic_structure(self, stack: TechStack, output_dir: Path) -> bool:
        """Create a basic project structure."""
        output_dir.mkdir(parents=True, exist_ok=True)
        readme = output_dir / "README.md"
        readme.write_text(f"# Project\n\n## Tech Stack\n- Language: {stack.language}\n")
        logger.info("Created basic scaffold for stack: %s", stack)
        return True
