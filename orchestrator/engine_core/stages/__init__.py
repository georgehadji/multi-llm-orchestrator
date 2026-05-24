"""Pipeline stages for task execution.

Each stage is a class implementing PipelineStage Protocol.
Stages are composed by TaskPipeline in engine.py __init__.
"""

from __future__ import annotations

from .generate import GenerateStage
from .critique import CritiqueStage
from .evaluate import EvaluateStage
from .validate import ValidateStage
from .preflight import PreflightStage
from .self_consistency import EnhancedSelfConsistencyStage as SelfConsistencyStage
from .persuasion_defense import PersuasionDefenseStage

__all__ = [
    "GenerateStage",
    "CritiqueStage",
    "EvaluateStage",
    "ValidateStage",
    "PreflightStage",
    "SelfConsistencyStage",
    "PersuasionDefenseStage",
]
