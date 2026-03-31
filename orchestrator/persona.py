"""
Persona Modes — Behavior Customization
=======================================

Implements Mnemo Cortex persona modes:
- STRICT: Business/production use, strict validation
- CREATIVE: Brainstorming, flexible output
- BALANCED: Default balanced behavior
- CUSTOM: User-defined persona

Usage:
    from orchestrator.persona import PersonaManager, Persona, PersonaMode

    manager = PersonaManager()

    # Set persona for a project
    manager.set_persona("project_001", PersonaMode.STRICT)

    # Get persona settings
    settings = manager.get_persona_settings("project_001")

    # Apply to orchestrator
    orchestrator.set_persona(PersonaMode.CREATIVE)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


class PersonaMode(Enum):
    """Persona behavior modes."""

    STRICT = "strict"  # Business/production, strict validation
    CREATIVE = "creative"  # Brainstorming, flexible output
    BALANCED = "balanced"  # Default balanced behavior
    CUSTOM = "custom"  # User-defined persona


@dataclass
class PersonaSettings:
    """Settings for a persona mode."""

    # Temperature settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40

    # Validation settings
    strict_validation: bool = True
    require_tests: bool = False
    require_documentation: bool = False
    max_iterations: int = 5

    # Output settings
    include_reasoning: bool = False
    verbose_output: bool = False
    format_code: bool = True

    # Safety settings
    enable_preflight: bool = True
    preflight_mode: str = "warn"  # pass, enrich, warn, block

    # Token settings
    max_tokens: int = 4096
    context_truncation: int = 40000

    # Custom instructions
    system_prompt_addition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "strict_validation": self.strict_validation,
            "require_tests": self.require_tests,
            "require_documentation": self.require_documentation,
            "max_iterations": self.max_iterations,
            "include_reasoning": self.include_reasoning,
            "verbose_output": self.verbose_output,
            "format_code": self.format_code,
            "enable_preflight": self.enable_preflight,
            "preflight_mode": self.preflight_mode,
            "max_tokens": self.max_tokens,
            "context_truncation": self.context_truncation,
            "system_prompt_addition": self.system_prompt_addition,
        }


class Persona:
    """
    Represents a complete persona configuration.
    """

    # Default personas
    PRESETS = {
        PersonaMode.STRICT: PersonaSettings(
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            strict_validation=True,
            require_tests=True,
            require_documentation=True,
            max_iterations=3,
            enable_preflight=True,
            preflight_mode="block",
            max_tokens=4096,
            system_prompt_addition="Follow best practices. Include tests and documentation. Prioritize correctness over speed.",
        ),
        PersonaMode.CREATIVE: PersonaSettings(
            temperature=0.9,
            top_p=0.95,
            top_k=80,
            strict_validation=False,
            require_tests=False,
            require_documentation=False,
            max_iterations=8,
            include_reasoning=True,
            verbose_output=True,
            enable_preflight=False,
            preflight_mode="pass",
            max_tokens=8192,
            context_truncation=60000,
            system_prompt_addition="Be creative and explore multiple solutions. Think outside the box. Don't be afraid to suggest innovative approaches.",
        ),
        PersonaMode.BALANCED: PersonaSettings(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            strict_validation=True,
            require_tests=False,
            require_documentation=False,
            max_iterations=5,
            include_reasoning=False,
            format_code=True,
            enable_preflight=True,
            preflight_mode="warn",
            max_tokens=4096,
            context_truncation=40000,
            system_prompt_addition="Provide balanced responses. Balance correctness with efficiency. Consider multiple approaches but choose the best one.",
        ),
    }

    def __init__(
        self,
        mode: PersonaMode,
        custom_settings: PersonaSettings | None = None,
    ):
        self.mode = mode
        self.settings = custom_settings or self.PRESETS.get(mode, PersonaSettings())

    def get_temperature(self) -> float:
        return self.settings.temperature

    def get_system_prompt(self, base_prompt: str = "") -> str:
        """Get the full system prompt with persona additions."""
        if self.settings.system_prompt_addition:
            return f"{base_prompt}\n\n{self.settings.system_prompt_addition}".strip()
        return base_prompt

    def should_require_tests(self) -> bool:
        return self.settings.require_tests

    def should_require_docs(self) -> bool:
        return self.settings.require_documentation

    def get_max_iterations(self) -> int:
        return self.settings.max_iterations

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            **self.settings.to_dict(),
        }


class PersonaManager:
    """
    Manages persona configurations for projects.

    Provides:
    - Per-project persona assignment
    - Persona presets (STRICT, CREATIVE, BALANCED)
    - Custom persona creation
    - Runtime persona switching
    """

    def __init__(self):
        self._project_personas: dict[str, PersonaMode] = {}
        self._custom_personas: dict[str, Persona] = {}
        self._default_mode = PersonaMode.BALANCED

    def set_persona(
        self,
        project_id: str,
        mode: PersonaMode,
        custom_settings: PersonaSettings | None = None,
    ) -> None:
        """
        Set persona for a project.

        Args:
            project_id: The project to set persona for
            mode: The persona mode
            custom_settings: Optional custom settings (for CUSTOM mode)
        """
        if mode == PersonaMode.CUSTOM and custom_settings:
            self._custom_personas[project_id] = Persona(mode, custom_settings)
        elif mode != PersonaMode.CUSTOM:
            self._project_personas[project_id] = mode

        logger.info(f"Set persona for project {project_id}: {mode.value}")

    def get_persona(self, project_id: str) -> Persona:
        """
        Get the persona for a project.

        Returns the Persona object with all settings.
        """
        mode = self._project_personas.get(project_id, self._default_mode)

        # Check for custom
        if project_id in self._custom_personas:
            return self._custom_personas[project_id]

        return Persona(mode)

    def get_persona_settings(self, project_id: str) -> PersonaSettings:
        """Get just the settings for a project."""
        return self.get_persona(project_id).settings

    def get_persona_mode(self, project_id: str) -> PersonaMode:
        """Get the persona mode for a project."""
        return self._project_personas.get(project_id, self._default_mode)

    def clear_persona(self, project_id: str) -> None:
        """Clear persona for a project (revert to default)."""
        self._project_personas.pop(project_id, None)
        self._custom_personas.pop(project_id, None)
        logger.info(f"Cleared persona for project {project_id}")

    def set_default_mode(self, mode: PersonaMode) -> None:
        """Set the default persona mode."""
        self._default_mode = mode
        logger.info(f"Set default persona mode: {mode.value}")

    def create_custom_persona(
        self,
        name: str,
        temperature: float = 0.7,
        strict_validation: bool = True,
        require_tests: bool = False,
        **kwargs,
    ) -> Persona:
        """
        Create a custom persona with specific settings.

        Returns a Persona that can be assigned to projects.
        """
        settings = PersonaSettings(
            temperature=temperature,
            strict_validation=strict_validation,
            require_tests=require_tests,
            **kwargs,
        )

        persona = Persona(PersonaMode.CUSTOM, settings)

        # Store by name for reference
        self._custom_personas[name] = persona

        return persona

    def apply_to_orchestrator(
        self,
        project_id: str,
        orchestrator: Any,
    ) -> None:
        """
        Apply persona settings to an orchestrator instance.

        This configures the orchestrator with persona-specific settings.
        """
        persona = self.get_persona(project_id)
        settings = persona.settings

        # Apply settings to orchestrator
        # Note: This depends on orchestrator's API
        if hasattr(orchestrator, "temperature"):
            orchestrator.temperature = settings.temperature

        if hasattr(orchestrator, "max_iterations"):
            orchestrator.max_iterations = settings.max_iterations

        if hasattr(orchestrator, "context_truncation_limit"):
            orchestrator.context_truncation_limit = settings.context_truncation

        logger.info(f"Applied persona settings to orchestrator for project {project_id}")

    def get_available_modes(self) -> list[PersonaMode]:
        """Get list of available persona modes."""
        return list(PersonaMode)

    def get_preset_settings(self, mode: PersonaMode) -> PersonaSettings:
        """Get the preset settings for a mode."""
        return Persona.PRESETS.get(mode, PersonaSettings())


# Global manager instance
_default_manager: PersonaManager | None = None


def get_persona_manager() -> PersonaManager:
    """Get the default persona manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PersonaManager()
    return _default_manager


def get_persona_settings(project_id: str) -> PersonaSettings:
    """Convenience function to get persona settings."""
    return get_persona_manager().get_persona_settings(project_id)
