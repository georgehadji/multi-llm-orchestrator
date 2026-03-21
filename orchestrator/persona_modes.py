"""
PersonaModes — Persona-based behavioral modes
===========================================
Module for implementing persona-based behavioral modes (Strict for production, 
Creative for ideation) without requiring configuration changes.

Pattern: Strategy
Async: No — pure logic operations
Layer: L4 Supervisor

Usage:
    from orchestrator.persona_modes import PersonaModeManager
    persona_manager = PersonaModeManager()
    persona_manager.set_persona("strict")
    result = await orchestrator.run_task(task, persona="creative")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .models import Model
from .modes import ModeManager, OperationMode, ModeConfig

logger = logging.getLogger("orchestrator.persona_modes")


class Persona(Enum):
    """Different personas for the orchestrator."""
    
    STRICT = "strict"  # Conservative, follows rules strictly, focuses on correctness
    CREATIVE = "creative"  # Innovative, exploratory approach, generates novel ideas
    BALANCED = "balanced"  # Middle ground between strict and creative
    ANALYTICAL = "analytical"  # Focuses on analysis, data, and logical reasoning
    CONVERSATIONAL = "conversational"  # Friendly, engaging, natural language
    EXPERT = "expert"  # Authoritative, detailed, technical responses
    HELPFUL = "helpful"  # Assistive, clear explanations, guidance-focused
    CRITICAL = "critical"  # Evaluative, identifies flaws, challenges assumptions
    PRECISION = "precision"  # Highly accurate, detailed, meticulous


@dataclass
class PersonaConfig:
    """Configuration for a specific persona."""
    
    # Core mode configuration
    mode_config: ModeConfig
    
    # Persona-specific settings
    tone: str  # The tone the persona should use
    approach: str  # How the persona approaches tasks
    focus: List[str]  # Primary focus areas of the persona
    communication_style: str  # How the persona communicates
    decision_making_style: str  # How the persona makes decisions
    risk_tolerance: float  # 0.0 (very conservative) to 1.0 (risk-taking)


class PersonaModeManager:
    """Manages persona-based behavioral modes for the orchestrator."""

    def __init__(self):
        """Initialize the persona mode manager."""
        self.current_persona = Persona.BALANCED
        self.persona_configs = self._initialize_persona_configs()
        self.mode_manager = ModeManager()  # Use the existing mode manager
    
    def _initialize_persona_configs(self) -> Dict[Persona, PersonaConfig]:
        """Initialize default configurations for each persona."""
        return {
            Persona.STRICT: PersonaConfig(
                mode_config=ModeConfig(
                    temperature=0.1,
                    top_p=0.1,
                    max_tokens=None,
                    presence_penalty=0.5,
                    frequency_penalty=0.5,
                    stop_sequences=[],
                    model_override=None,
                    validation_level="thorough",
                    creativity_boost=0.2
                ),
                tone="formal",
                approach="rule-following",
                focus=["accuracy", "compliance", "correctness"],
                communication_style="precise and direct",
                decision_making_style="conservative and cautious",
                risk_tolerance=0.1
            ),
            Persona.CREATIVE: PersonaConfig(
                mode_config=ModeConfig(
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=None,
                    presence_penalty=0.2,
                    frequency_penalty=0.2,
                    stop_sequences=[],
                    model_override=None,
                    validation_level="basic",
                    creativity_boost=1.5
                ),
                tone="exploratory",
                approach="innovative",
                focus=["novelty", "ideation", "possibilities"],
                communication_style="imaginative and open-minded",
                decision_making_style="experimental and bold",
                risk_tolerance=0.9
            ),
            Persona.BALANCED: PersonaConfig(
                mode_config=ModeConfig(
                    temperature=0.5,
                    top_p=0.7,
                    max_tokens=None,
                    presence_penalty=0.3,
                    frequency_penalty=0.3,
                    stop_sequences=[],
                    model_override=None,
                    validation_level="basic",
                    creativity_boost=1.0
                ),
                tone="neutral",
                approach="pragmatic",
                focus=["effectiveness", "efficiency", "appropriateness"],
                communication_style="clear and informative",
                decision_making_style="considered and balanced",
                risk_tolerance=0.5
            ),
            Persona.ANALYTICAL: PersonaConfig(
                mode_config=ModeConfig(
                    temperature=0.3,
                    top_p=0.5,
                    max_tokens=None,
                    presence_penalty=0.6,
                    frequency_penalty=0.6,
                    stop_sequences=[],
                    model_override=Model.DEEPSEEK_REASONER,  # Use reasoning model
                    validation_level="thorough",
                    creativity_boost=0.4
                ),
                tone="objective",
                approach="data-driven",
                focus=["analysis", "patterns", "evidence"],
                communication_style="logical and systematic",
                decision_making_style="evidence-based",
                risk_tolerance=0.3
            ),
            Persona.CONVERSATIONAL: PersonaConfig(
                mode_config=ModeConfig(
                    temperature=0.7,
                    top_p=0.8,
                    max_tokens=None,
                    presence_penalty=0.2,
                    frequency_penalty=0.2,
                    stop_sequences=[],
                    model_override=None,
                    validation_level="none",
                    creativity_boost=0.8
                ),
                tone="friendly",
                approach="engaging",
                focus=["rapport", "clarity", "approachability"],
                communication_style="natural and conversational",
                decision_making_style="collaborative",
                risk_tolerance=0.6
            ),
            Persona.EXPERT: PersonaConfig(
                mode_config=ModeConfig(
                    temperature=0.2,
                    top_p=0.3,
                    max_tokens=None,
                    presence_penalty=0.7,
                    frequency_penalty=0.7,
                    stop_sequences=[],
                    model_override=Model.DEEPSEEK_REASONER,  # Use reasoning model
                    validation_level="thorough",
                    creativity_boost=0.3
                ),
                tone="authoritative",
                approach="comprehensive",
                focus=["depth", "detail", "technical accuracy"],
                communication_style="informed and detailed",
                decision_making_style="knowledge-based",
                risk_tolerance=0.2
            ),
            Persona.HELPFUL: PersonaConfig(
                mode_config=ModeConfig(
                    temperature=0.6,
                    top_p=0.7,
                    max_tokens=None,
                    presence_penalty=0.3,
                    frequency_penalty=0.3,
                    stop_sequences=[],
                    model_override=None,
                    validation_level="basic",
                    creativity_boost=0.7
                ),
                tone="supportive",
                approach="assistive",
                focus=["guidance", "explanation", "problem-solving"],
                communication_style="clear and encouraging",
                decision_making_style="solution-oriented",
                risk_tolerance=0.5
            ),
            Persona.CRITICAL: PersonaConfig(
                mode_config=ModeConfig(
                    temperature=0.4,
                    top_p=0.6,
                    max_tokens=None,
                    presence_penalty=0.8,
                    frequency_penalty=0.8,
                    stop_sequences=[],
                    model_override=Model.DEEPSEEK_REASONER,  # Use reasoning model
                    validation_level="thorough",
                    creativity_boost=0.3
                ),
                tone="evaluative",
                approach="skeptical",
                focus=["flaws", "assumptions", "improvements"],
                communication_style="challenging and probing",
                decision_making_style="skeptical and questioning",
                risk_tolerance=0.2
            ),
            Persona.PRECISION: PersonaConfig(
                mode_config=ModeConfig(
                    temperature=0.1,
                    top_p=0.1,
                    max_tokens=None,
                    presence_penalty=0.9,
                    frequency_penalty=0.9,
                    stop_sequences=[],
                    model_override=Model.DEEPSEEK_REASONER,  # Use reasoning model
                    validation_level="thorough",
                    creativity_boost=0.1
                ),
                tone="meticulous",
                approach="detailed",
                focus=["accuracy", "completeness", "attention to detail"],
                communication_style="precise and thorough",
                decision_making_style="careful and methodical",
                risk_tolerance=0.1
            )
        }
    
    def set_persona(self, persona: Persona):
        """Set the current persona."""
        self.current_persona = persona
        logger.info(f"Persona set to: {persona.value}")
        
        # Update the underlying mode manager to match persona characteristics
        persona_config = self.persona_configs[persona]
        self.mode_manager.set_mode(self._map_persona_to_operation_mode(persona))
    
    def get_current_persona(self) -> Persona:
        """Get the current persona."""
        return self.current_persona
    
    def _map_persona_to_operation_mode(self, persona: Persona) -> OperationMode:
        """Map a persona to the closest OperationMode."""
        mapping = {
            Persona.STRICT: OperationMode.STRICT,
            Persona.CREATIVE: OperationMode.CREATIVE,
            Persona.BALANCED: OperationMode.BALANCED,
            Persona.ANALYTICAL: OperationMode.ACCURATE,  # Closest match
            Persona.CONVERSATIONAL: OperationMode.VERBOSE,  # Closest match
            Persona.EXPERT: OperationMode.ACCURATE,  # Closest match
            Persona.HELPFUL: OperationMode.VERBOSE,  # Closest match
            Persona.CRITICAL: OperationMode.ACCURATE,  # Closest match
            Persona.PRECISION: OperationMode.ACCURATE  # Closest match
        }
        return mapping.get(persona, OperationMode.BALANCED)
    
    def get_config_for_persona(self, persona: Persona) -> PersonaConfig:
        """Get the configuration for a specific persona."""
        if persona in self.persona_configs:
            return self.persona_configs[persona]
        else:
            # Return balanced persona config as fallback
            return self.persona_configs[Persona.BALANCED]
    
    def get_current_config(self) -> PersonaConfig:
        """Get the configuration for the current persona."""
        return self.get_config_for_persona(self.current_persona)
    
    def apply_persona_to_params(self, params: Dict[str, Any], 
                               persona: Optional[Persona] = None) -> Dict[str, Any]:
        """
        Apply persona-specific parameters to a parameter dictionary.
        
        Args:
            params: Original parameters
            persona: Persona to apply (uses current persona if not specified)
            
        Returns:
            Dict[str, Any]: Updated parameters with persona-specific values
        """
        target_persona = persona or self.current_persona
        config = self.get_config_for_persona(target_persona).mode_config
        
        # Use the underlying mode manager's method to apply mode-specific params
        return self.mode_manager.apply_mode_to_params(params, 
                                                     self._map_persona_to_operation_mode(target_persona))
    
    def get_model_for_persona(self, persona: Optional[Persona] = None) -> Model:
        """
        Get the appropriate model for a specific persona.
        
        Args:
            persona: Persona to get model for (uses current persona if not specified)
            
        Returns:
            Model: The model appropriate for the persona
        """
        target_persona = persona or self.current_persona
        config = self.get_config_for_persona(target_persona).mode_config
        
        # Return overridden model if specified, otherwise default
        return config.model_override or Model.DEEPSEEK_CHAT
    
    def adjust_validation_level(self, persona: Optional[Persona] = None) -> str:
        """
        Get the validation level for a specific persona.
        
        Args:
            persona: Persona to get validation level for (uses current persona if not specified)
            
        Returns:
            str: The validation level ('none', 'basic', 'thorough')
        """
        target_persona = persona or self.current_persona
        config = self.get_config_for_persona(target_persona).mode_config
        
        return config.validation_level
    
    def get_creativity_boost(self, persona: Optional[Persona] = None) -> float:
        """
        Get the creativity boost factor for a specific persona.
        
        Args:
            persona: Persona to get creativity boost for (uses current persona if not specified)
            
        Returns:
            float: The creativity boost factor (0.0-2.0)
        """
        target_persona = persona or self.current_persona
        config = self.get_config_for_persona(target_persona).mode_config
        
        return config.creativity_boost
    
    def get_persona_characteristics(self, persona: Optional[Persona] = None) -> Dict[str, Any]:
        """
        Get the characteristics of a specific persona.
        
        Args:
            persona: Persona to get characteristics for (uses current persona if not specified)
            
        Returns:
            Dict[str, Any]: Characteristics of the persona
        """
        target_persona = persona or self.current_persona
        config = self.get_config_for_persona(target_persona)
        
        return {
            "tone": config.tone,
            "approach": config.approach,
            "focus": config.focus,
            "communication_style": config.communication_style,
            "decision_making_style": config.decision_making_style,
            "risk_tolerance": config.risk_tolerance
        }
    
    def create_persona_context(self, persona: Persona) -> Dict[str, Any]:
        """
        Create a context dictionary for a specific persona.
        
        Args:
            persona: Persona to create context for
            
        Returns:
            Dict[str, Any]: Context dictionary with persona-specific settings
        """
        config = self.get_config_for_persona(persona)
        
        return {
            "persona": persona.value,
            "mode_config": config.mode_config,
            "tone": config.tone,
            "approach": config.approach,
            "focus": config.focus,
            "communication_style": config.communication_style,
            "decision_making_style": config.decision_making_style,
            "risk_tolerance": config.risk_tolerance
        }
    
    def switch_persona_smoothly(self, new_persona: Persona, 
                              transition_message: bool = True) -> str:
        """
        Switch to a new persona with a smooth transition.
        
        Args:
            new_persona: The new persona to switch to
            transition_message: Whether to return a transition message
            
        Returns:
            str: Transition message if requested
        """
        old_persona = self.current_persona
        self.set_persona(new_persona)
        
        if transition_message:
            old_char = self.get_persona_characteristics(old_persona)
            new_char = self.get_persona_characteristics(new_persona)
            
            return (f"Switching from {old_persona.value} persona to {new_persona.value} persona. "
                    f"Changing approach from '{old_char['approach']}' to '{new_char['approach']}', "
                    f"and communication style from '{old_char['communication_style']}' "
                    f"to '{new_char['communication_style']}'.")
        
        return ""


# Global persona manager instance for convenience
_global_persona_manager = PersonaModeManager()


def get_global_persona_manager() -> PersonaModeManager:
    """Get the global persona manager instance."""
    return _global_persona_manager


def set_global_persona(persona: Persona):
    """Set the global persona."""
    _global_persona_manager.set_persona(persona)


def get_global_persona() -> Persona:
    """Get the global persona."""
    return _global_persona_manager.get_current_persona()