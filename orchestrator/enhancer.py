"""
Enhancement Module for Project Enhancer Feature
================================================

Provides the Enhancement dataclass and pure utility functions for handling
LLM-generated enhancement suggestions to improve project descriptions and
success criteria before core decomposition runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from orchestrator.models import Model
from orchestrator.api_clients import UnifiedClient

logger = logging.getLogger("orchestrator.enhancer")


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


class ProjectEnhancer:
    """LLM-powered enhancement generator for project specifications.

    Generates enhancement suggestions to improve project descriptions and success
    criteria before core decomposition runs. Uses an LLM to analyze the project
    and suggest concrete improvements.

    Attributes
    ----------
    client : UnifiedClient
        Async LLM client for making enhancement generation requests
    """

    def __init__(self, client: UnifiedClient | None = None):
        """Initialize ProjectEnhancer with optional UnifiedClient.

        Parameters
        ----------
        client : UnifiedClient | None
            Optional pre-configured UnifiedClient. If None, creates a new instance.
        """
        self.client = client or UnifiedClient()

    async def analyze(self, description: str, criteria: str) -> list[Enhancement]:
        """Generate enhancement suggestions for a project description and criteria.

        Combines the description and criteria to select an appropriate model,
        then prompts the LLM to suggest 3-7 concrete enhancements. The LLM
        response is parsed into Enhancement objects.

        Parameters
        ----------
        description : str
            The project description to enhance
        criteria : str
            The success criteria to enhance

        Returns
        -------
        list[Enhancement]
            List of suggested enhancements, or [] if LLM call fails

        Notes
        ------
        - Combines description and criteria for model selection
        - Uses appropriate model based on combined length
        - Budget cap: $0.10 per call
        - Graceful error handling: returns [] on any exception
        """
        try:
            # Select model based on combined length
            combined = f"{description} {criteria}"
            model = _select_enhance_model(combined)

            # Create LLM prompt
            prompt = (
                "You are a project specification expert. Analyze the following project description "
                "and success criteria, then suggest 3-7 concrete improvements to make them more complete, realistic, and measurable.\n\n"
                f"Project Description: {description}\n"
                f"Success Criteria: {criteria}\n\n"
                "For each enhancement, provide the following information:\n"
                "1. type: \"completeness\" | \"criteria\" | \"risk\"\n"
                "   - \"completeness\" for missing details about what the project should include\n"
                "   - \"criteria\" for missing or unclear success metrics\n"
                "   - \"risk\" for unaddressed risks or edge cases\n"
                "2. title: short title (≤8 words)\n"
                "3. description: 1-2 sentence explanation of why this improvement is needed\n"
                "4. patch_description: a clause to append to the project description (e.g., \"with JWT authentication\")\n"
                "5. patch_criteria: a clause to append to the success criteria (e.g., \"supports JWT authentication\")\n\n"
                "Return your response as a valid JSON array. Example format:\n"
                "[\n"
                "  {\n"
                "    \"type\": \"completeness\",\n"
                "    \"title\": \"Add Performance Metrics\",\n"
                "    \"description\": \"The project lacks specific performance requirements.\",\n"
                "    \"patch_description\": \"with response time < 100ms for all endpoints\",\n"
                "    \"patch_criteria\": \"achieve response time < 100ms for all endpoints\"\n"
                "  }\n"
                "]\n\n"
                "Important: Return ONLY the JSON array, with no additional text or markdown formatting."
            )

            # Call LLM via UnifiedClient
            llm_response = await self.client.call(
                model=model,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7,
            )

            # Extract text from response
            llm_text = llm_response.text

            # Parse JSON into Enhancement objects
            enhancements = _parse_enhancements(llm_text)

            return enhancements

        except Exception as e:
            # Graceful error handling: never raise
            logger.warning(f"ProjectEnhancer.analyze failed: {e}", exc_info=True)
            return []


def _present_enhancements(enhancements: list[Enhancement]) -> list[Enhancement]:
    """Present each enhancement interactively; return accepted ones.

    Y/y/Enter → accept. n/N → reject. Ctrl-C → reject all remaining.

    Parameters
    ----------
    enhancements : list[Enhancement]
        List of enhancement suggestions to present

    Returns
    -------
    list[Enhancement]
        List of enhancements the user accepted
    """
    if not enhancements:
        print("  ✓ Spec looks complete — no suggestions.\n")
        return []

    total = len(enhancements)
    print(f"\n  📋 {total} improvement{'s' if total != 1 else ''} found:\n")

    accepted: list[Enhancement] = []
    try:
        for i, e in enumerate(enhancements, start=1):
            print(f"  [{i}/{total}] {e.type} — {e.title}")
            print(f"        {e.description}")
            if e.patch_description:
                print(f"        Adds: \"{e.patch_description}\"")
            if e.patch_criteria:
                print(f"        Adds criteria: \"{e.patch_criteria}\"")
            try:
                answer = input("        Apply? [Y/n]: ").strip().lower()
            except KeyboardInterrupt:
                print("\n  (interrupted — skipping remaining)")
                break
            if answer in ("", "y", "yes"):
                accepted.append(e)
            print()
    except Exception as exc:
        logger.warning("Unexpected error during enhancement prompts: %s", exc)

    applied = len(accepted)
    print(f"✓ Applied {applied}/{total} enhancements. Running enhanced project...")
    print("─" * 54)
    return accepted
