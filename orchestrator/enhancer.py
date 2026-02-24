"""
Enhancement Module for Project Enhancer Feature
================================================

Provides the Enhancement dataclass and pure utility functions for handling
LLM-generated enhancement suggestions to improve project descriptions and
success criteria before core decomposition runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from orchestrator.models import Model


@dataclass(frozen=True)
class Enhancement:
    """A single enhancement suggestion for a project description or criteria.
    
    An Enhancement represents a specific suggestion to improve a project's
    description or success criteria. The enhancement is immutable (frozen).
    
    Attributes
    ----------
    type : str
        Category of enhancement: "completeness", "criteria", or "risk"
    title : str
        Short title of the enhancement (≤8 words)
    description : str
        1–2 sentence description of what is being suggested
    patch_description : str
        Clause to append to the project description
    patch_criteria : str
        Clause to append to the success criteria
    """
    type: str           # "completeness" | "criteria" | "risk"
    title: str          # ≤8 words
    description: str    # 1–2 sentences
    patch_description: str  # Clause to append to project description
    patch_criteria: str     # Clause to append to success criteria


def _select_enhance_model(description: str) -> Model:
    """Select the LLM model for enhancement suggestions based on description length.
    
    Uses DEEPSEEK_REASONER (o1-class) for longer descriptions (>50 words)
    that may require deeper reasoning, and DEEPSEEK_CHAT (V3) for shorter
    descriptions that are simpler to enhance.
    
    Parameters
    ----------
    description : str
        The project description
    
    Returns
    -------
    Model
        DEEPSEEK_REASONER if description has >50 words, DEEPSEEK_CHAT otherwise
    """
    word_count = len(description.split())
    if word_count > 50:
        return Model.DEEPSEEK_REASONER
    return Model.DEEPSEEK_CHAT


def _parse_enhancements(llm_output: str) -> list[Enhancement]:
    """Parse LLM JSON response into Enhancement objects.
    
    Gracefully handles all errors (invalid JSON, missing fields, type validation)
    by returning an empty list. Never raises exceptions.
    
    Expected JSON format:
    [
        {
            "type": "completeness" | "criteria" | "risk",
            "title": "...",
            "description": "...",
            "patch_description": "...",
            "patch_criteria": "..."
        },
        ...
    ]
    
    Parameters
    ----------
    llm_output : str
        JSON string from LLM containing enhancement suggestions
    
    Returns
    -------
    list[Enhancement]
        List of parsed Enhancement objects, or [] on any error
    """
    try:
        data = json.loads(llm_output)
        if not isinstance(data, list):
            return []
        
        enhancements = []
        valid_types = {"completeness", "criteria", "risk"}
        
        for item in data:
            try:
                if not isinstance(item, dict):
                    return []
                
                # Check for required fields
                required_fields = {
                    "type", "title", "description",
                    "patch_description", "patch_criteria"
                }
                if not required_fields.issubset(item.keys()):
                    return []
                
                # Validate type value
                if item["type"] not in valid_types:
                    return []
                
                # Create Enhancement object
                enhancement = Enhancement(
                    type=item["type"],
                    title=item["title"],
                    description=item["description"],
                    patch_description=item["patch_description"],
                    patch_criteria=item["patch_criteria"],
                )
                enhancements.append(enhancement)
            except (KeyError, TypeError, ValueError):
                return []
        
        return enhancements
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _apply_enhancements(
    description: str,
    criteria: str,
    accepted: list[Enhancement]
) -> tuple[str, str]:
    """Apply all accepted enhancements to description and criteria.
    
    Appends each enhancement's patches in order:
    - patch_description appended to description: f"{description} {patch}"
    - patch_criteria appended to criteria: f"{criteria}; {patch}"
    
    Parameters
    ----------
    description : str
        Project description (may be empty)
    criteria : str
        Success criteria (may be empty)
    accepted : list[Enhancement]
        List of enhancements to apply
    
    Returns
    -------
    tuple[str, str]
        (new_description, new_criteria) with all patches applied in order
    """
    new_description = description
    new_criteria = criteria
    
    for enhancement in accepted:
        new_description = f"{new_description} {enhancement.patch_description}"
        new_criteria = f"{new_criteria}; {enhancement.patch_criteria}"
    
    return (new_description, new_criteria)
