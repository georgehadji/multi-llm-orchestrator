"""
Modes — Per-request behavioral modes
====================================
Module for managing different operational modes (Strict, Creative, etc.) that affect
the behavior of the orchestrator for specific requests.

Pattern: Strategy
Async: No — pure logic operations
Layer: L4 Supervisor

Usage:
    from orchestrator.modes import ModeManager, OperationMode
    manager = ModeManager()
    manager.set_mode(OperationMode.STRICT)
    result = await orchestrator.run_task(task, mode=OperationMode.CREATIVE)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .models import Model

logger = logging.getLogger("orchestrator.modes")


class OperationMode(Enum):
    """Different operational modes for the orchestrator."""

    STRICT = "strict"  # Conservative, follows rules strictly
    CREATIVE = "creative"  # Innovative, exploratory approach
    BALANCED = "balanced"  # Middle ground between strict and creative
    FAST = "fast"  # Prioritizes speed over quality
    ACCURATE = "accurate"  # Prioritizes accuracy over speed
    CHEAP = "cheap"  # Prioritizes cost efficiency
    VERBOSE = "verbose"  # Provides detailed output
    CONCISE = "concise"  # Provides minimal output


@dataclass
class ModeConfig:
    """Configuration for a specific operational mode."""

    temperature: float  # Controls randomness (0.0-2.0)
    top_p: float  # Nucleus sampling parameter (0.0-1.0)
    max_tokens: int | None  # Maximum tokens to generate
    presence_penalty: float  # Penalty for repeated content
    frequency_penalty: float  # Penalty for frequent tokens
    stop_sequences: list[str]  # Sequences to stop generation
    model_override: Model | None  # Override default model selection
    validation_level: str  # Level of validation (none, basic, thorough)
    creativity_boost: float  # Factor to boost creative elements (0.0-2.0)


class ModeManager:
    """Manages different operational modes for the orchestrator."""

    def __init__(self):
        """Initialize the mode manager with default configurations."""
        self.current_mode = OperationMode.BALANCED
        self.mode_configs = self._initialize_mode_configs()

    def _initialize_mode_configs(self) -> dict[OperationMode, ModeConfig]:
        """Initialize default configurations for each mode."""
        return {
            OperationMode.STRICT: ModeConfig(
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
            OperationMode.CREATIVE: ModeConfig(
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
            OperationMode.BALANCED: ModeConfig(
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
            OperationMode.FAST: ModeConfig(
                temperature=0.3,
                top_p=0.5,
                max_tokens=512,
                presence_penalty=0.1,
                frequency_penalty=0.1,
                stop_sequences=[],
                model_override=Model.DEEPSEEK_CHAT,  # Use faster model
                validation_level="none",
                creativity_boost=0.7,
            ),
            OperationMode.ACCURATE: ModeConfig(
                temperature=0.1,
                top_p=0.1,
                max_tokens=None,
                presence_penalty=0.8,
                frequency_penalty=0.8,
                stop_sequences=[],
                model_override=Model.DEEPSEEK_REASONER,  # Use reasoning model
                validation_level="thorough",
                creativity_boost=0.3,
            ),
            OperationMode.CHEAP: ModeConfig(
                temperature=0.4,
                top_p=0.6,
                max_tokens=256,
                presence_penalty=0.2,
                frequency_penalty=0.2,
                stop_sequences=[],
                model_override=Model.DEEPSEEK_CHAT,  # Use cheaper model
                validation_level="none",
                creativity_boost=0.6,
            ),
            OperationMode.VERBOSE: ModeConfig(
                temperature=0.6,
                top_p=0.8,
                max_tokens=None,
                presence_penalty=0.2,
                frequency_penalty=0.2,
                stop_sequences=[],
                model_override=None,
                validation_level="basic",
                creativity_boost=1.0,
            ),
            OperationMode.CONCISE: ModeConfig(
                temperature=0.2,
                top_p=0.3,
                max_tokens=256,
                presence_penalty=0.5,
                frequency_penalty=0.5,
                stop_sequences=["\n\n"],
                model_override=None,
                validation_level="none",
                creativity_boost=0.5,
            ),
        }

    def set_mode(self, mode: OperationMode):
        """Set the current operational mode."""
        self.current_mode = mode
        logger.info(f"Operational mode set to: {mode.value}")

    def get_current_mode(self) -> OperationMode:
        """Get the current operational mode."""
        return self.current_mode

    def get_config_for_mode(self, mode: OperationMode) -> ModeConfig:
        """Get the configuration for a specific mode."""
        if mode in self.mode_configs:
            return self.mode_configs[mode]
        else:
            # Return balanced mode config as fallback
            return self.mode_configs[OperationMode.BALANCED]

    def get_current_config(self) -> ModeConfig:
        """Get the configuration for the current mode."""
        return self.get_config_for_mode(self.current_mode)

    def apply_mode_to_params(
        self, params: dict[str, Any], mode: OperationMode | None = None
    ) -> dict[str, Any]:
        """
        Apply mode-specific parameters to a parameter dictionary.

        Args:
            params: Original parameters
            mode: Mode to apply (uses current mode if not specified)

        Returns:
            Dict[str, Any]: Updated parameters with mode-specific values
        """
        target_mode = mode or self.current_mode
        config = self.get_config_for_mode(target_mode)

        # Update parameters with mode-specific values
        updated_params = params.copy()

        # Apply temperature if not already set
        if "temperature" not in updated_params:
            updated_params["temperature"] = config.temperature

        # Apply top_p if not already set
        if "top_p" not in updated_params:
            updated_params["top_p"] = config.top_p

        # Apply max_tokens if not already set and config has a value
        if "max_tokens" not in updated_params and config.max_tokens is not None:
            updated_params["max_tokens"] = config.max_tokens

        # Apply penalties
        if "presence_penalty" not in updated_params:
            updated_params["presence_penalty"] = config.presence_penalty

        if "frequency_penalty" not in updated_params:
            updated_params["frequency_penalty"] = config.frequency_penalty

        # Apply stop sequences
        if "stop" not in updated_params:
            updated_params["stop"] = config.stop_sequences
        else:
            # Combine with existing stop sequences
            updated_params["stop"] = list(set(updated_params["stop"] + config.stop_sequences))

        return updated_params

    def get_model_for_mode(self, mode: OperationMode | None = None) -> Model:
        """
        Get the appropriate model for a specific mode.

        Args:
            mode: Mode to get model for (uses current mode if not specified)

        Returns:
            Model: The model appropriate for the mode
        """
        target_mode = mode or self.current_mode
        config = self.get_config_for_mode(target_mode)

        # Return overridden model if specified, otherwise default
        return config.model_override or Model.DEEPSEEK_CHAT

    def adjust_validation_level(self, mode: OperationMode | None = None) -> str:
        """
        Get the validation level for a specific mode.

        Args:
            mode: Mode to get validation level for (uses current mode if not specified)

        Returns:
            str: The validation level ('none', 'basic', 'thorough')
        """
        target_mode = mode or self.current_mode
        config = self.get_config_for_mode(target_mode)

        return config.validation_level

    def get_creativity_boost(self, mode: OperationMode | None = None) -> float:
        """
        Get the creativity boost factor for a specific mode.

        Args:
            mode: Mode to get creativity boost for (uses current mode if not specified)

        Returns:
            float: The creativity boost factor (0.0-2.0)
        """
        target_mode = mode or self.current_mode
        config = self.get_config_for_mode(target_mode)

        return config.creativity_boost

    def create_mode_context(self, mode: OperationMode) -> dict[str, Any]:
        """
        Create a context dictionary for a specific mode.

        Args:
            mode: Mode to create context for

        Returns:
            Dict[str, Any]: Context dictionary with mode-specific settings
        """
        config = self.get_config_for_mode(mode)

        return {
            "mode": mode.value,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "stop_sequences": config.stop_sequences,
            "model_override": config.model_override.value if config.model_override else None,
            "validation_level": config.validation_level,
            "creativity_boost": config.creativity_boost,
        }


# Global mode manager instance for convenience
_global_mode_manager = ModeManager()


def get_global_mode_manager() -> ModeManager:
    """Get the global mode manager instance."""
    return _global_mode_manager


def set_global_mode(mode: OperationMode):
    """Set the global operational mode."""
    _global_mode_manager.set_mode(mode)


def get_global_mode() -> OperationMode:
    """Get the global operational mode."""
    return _global_mode_manager.get_current_mode()
