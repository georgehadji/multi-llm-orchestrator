"""
PersonaModes — Persona-based behavioral modes
===========================================

Standalone orchestrator-persona package version of persona modes.
Abstracts the orchestrator's specific ModeManager and Model implementations
using Protocol interfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, TypeVar

logger = logging.getLogger("orchestrator_persona.persona_modes")

TModel = TypeVar('TModel')
TOperationMode = TypeVar('TOperationMode')


class ModeConfigProtocol(Protocol):
    temperature: float
    top_p: float
    max_tokens: int | None
    presence_penalty: float
    frequency_penalty: float
    stop_sequences: list[str]
    model_override: Any | None
    validation_level: str
    creativity_boost: float


class ModeManagerCallback(Protocol):
    def set_mode(self, mode: Any) -> None:
        ...

    def apply_mode_to_params(self, params: dict[str, Any], mode: Any) -> dict[str, Any]:
        ...


class Persona(Enum):
    """Different personas for the orchestrator."""

    STRICT = "strict"
    CREATIVE = "creative"
    BALANCED = "balanced"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    EXPERT = "expert"
    HELPFUL = "helpful"
    CRITICAL = "critical"
    PRECISION = "precision"
    PONYTAIL = "ponytail"


@dataclass
class PersonaConfig:
    """Configuration for a specific persona."""

    mode_config_dict: dict[str, Any]

    tone: str
    approach: str
    focus: list[str]
    communication_style: str
    decision_making_style: str
    risk_tolerance: float


class PersonaModeManager:
    """Manages persona-based behavioral modes, isolated from core orchestrator dependencies."""

    def __init__(
        self,
        mode_manager_callback: ModeManagerCallback | None = None,
        operation_mode_mapper: Callable[[Persona], Any] | None = None,
        default_model: Any | None = None,
    ):
        """Initialize the persona mode manager."""
        self.current_persona = Persona.BALANCED
        self.persona_configs = self._initialize_persona_configs()
        self.mode_manager_callback = mode_manager_callback
        self.operation_mode_mapper = operation_mode_mapper
        self.default_model = default_model

    def _initialize_persona_configs(self) -> dict[Persona, PersonaConfig]:
        """Initialize default configurations for each persona using raw dicts for ModeConfig params."""
        return {
            Persona.STRICT: PersonaConfig(
                mode_config_dict=dict(
                    temperature=0.1,
                    top_p=0.1,
                    max_tokens=None,
                    presence_penalty=0.5,
                    frequency_penalty=0.5,
                    stop_sequences=[],
                    model_override=None,
                    validation_level="thorough",
                    creativity_boost=0.2,
                ),
                tone="formal",
                approach="rule-following",
                focus=["accuracy", "compliance", "correctness"],
                communication_style="precise and direct",
                decision_making_style="conservative and cautious",
                risk_tolerance=0.1,
            ),
            Persona.CREATIVE: PersonaConfig(
                mode_config_dict=dict(
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=None,
                    presence_penalty=0.2,
                    frequency_penalty=0.2,
                    stop_sequences=[],
                    model_override=None,
                    validation_level="basic",
                    creativity_boost=1.5,
                ),
                tone="exploratory",
                approach="innovative",
                focus=["novelty", "ideation", "possibilities"],
                communication_style="imaginative and open-minded",
                decision_making_style="experimental and bold",
                risk_tolerance=0.9,
            ),
            Persona.BALANCED: PersonaConfig(
                mode_config_dict=dict(
                    temperature=0.5,
                    top_p=0.7,
                    max_tokens=None,
                    presence_penalty=0.3,
                    frequency_penalty=0.3,
                    stop_sequences=[],
                    model_override=None,
                    validation_level="basic",
                    creativity_boost=1.0,
                ),
                tone="neutral",
                approach="pragmatic",
                focus=["effectiveness", "efficiency", "appropriateness"],
                communication_style="clear and informative",
                decision_making_style="considered and balanced",
                risk_tolerance=0.5,
            ),
            Persona.PONYTAIL: PersonaConfig(
                mode_config_dict=dict(
                    temperature=0.1,
                    top_p=0.1,
                    max_tokens=2048,
                    presence_penalty=0.8,
                    frequency_penalty=0.8,
                    stop_sequences=[],
                    model_override="STEPFUN_STEP_3_5_FLASH", # Deferred enum string binding
                    validation_level="basic",
                    creativity_boost=0.1,
                ),
                tone="minimalist",
                approach="lazy-but-correct",
                focus=["YAGNI", "stdlib-first", "zero-dependencies", "brevity"],
                communication_style="ultra-concise (code-first, max 3 lines of prose)",
                decision_making_style="ruthless simplification",
                risk_tolerance=0.4,
            ),
            # Add analytical, conversational, expert, helpful, critical, precision...
        }

    def set_persona(self, persona: Persona):
        self.current_persona = persona
        logger.info(f"Persona set to: {persona.value}")

        if self.mode_manager_callback and self.operation_mode_mapper:
            mapped_mode = self.operation_mode_mapper(persona)
            self.mode_manager_callback.set_mode(mapped_mode)

    def get_current_persona(self) -> Persona:
        return self.current_persona

    def get_config_for_persona(self, persona: Persona) -> PersonaConfig:
        return self.persona_configs.get(persona, self.persona_configs[Persona.BALANCED])

    def apply_persona_to_params(
        self, params: dict[str, Any], persona: Persona | None = None
    ) -> dict[str, Any]:
        target_persona = persona or self.current_persona
        
        if self.mode_manager_callback and self.operation_mode_mapper:
            mapped_mode = self.operation_mode_mapper(target_persona)
            return self.mode_manager_callback.apply_mode_to_params(params, mapped_mode)
        
        # Fallback if no ModeManager is injected
        config = self.get_config_for_persona(target_persona).mode_config_dict
        params.update({
            "temperature": config.get("temperature", params.get("temperature")),
            "top_p": config.get("top_p", params.get("top_p")),
        })
        return params

    def get_model_for_persona(self, persona: Persona | None = None) -> Any:
        target_persona = persona or self.current_persona
        config = self.get_config_for_persona(target_persona).mode_config_dict
        override = config.get("model_override")
        return override if override else self.default_model

    def switch_persona_smoothly(self, new_persona: Persona, transition_message: bool = True) -> str:
        old_persona = self.current_persona
        self.set_persona(new_persona)

        if transition_message:
            old_char = self.get_config_for_persona(old_persona)
            new_char = self.get_config_for_persona(new_persona)

            return (
                f"Switching from {old_persona.value} persona to {new_persona.value} persona. "
                f"Changing approach from '{old_char.approach}' to '{new_char.approach}', "
                f"and communication style from '{old_char.communication_style}' "
                f"to '{new_char.communication_style}'."
            )
        return ""
