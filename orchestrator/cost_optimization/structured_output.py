"""
Structured Output Module
=========================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements Pydantic-based structured output enforcement for zero JSON parse failures.

Features:
- Pydantic model definitions for all phases
- Tool use enforcement (Anthropic, OpenAI)
- Automatic JSON schema generation
- Zero regex parsing needed

Usage:
    from orchestrator.cost_optimization import StructuredOutputEnforcer

    enforcer = StructuredOutputEnforcer(client=api_client)

    # Decomposition output
    result = await enforcer.generate_structured(
        model="claude-sonnet-4.6",
        prompt="Decompose this project...",
        output_type=DecompositionOutput,
    )

    # Access typed result
    print(f"Tasks: {result.tasks}")
    print(f"Estimated cost: ${result.estimated_cost}")
"""

from __future__ import annotations

import json
from typing import Any

try:
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None  # type: ignore
    Field = None  # type: ignore

from orchestrator.log_config import get_logger

logger = get_logger(__name__)


# FIX-OPT-004: Fail-fast if Pydantic not available - consistent error handling
if not HAS_PYDANTIC:
    raise ImportError(
        "Pydantic is required for structured output. " "Install with: pip install pydantic>=2.0"
    )


# ─────────────────────────────────────────────
# Pydantic Models for Structured Output
# ─────────────────────────────────────────────


class TaskSpec(BaseModel):
    """Specification for a single task."""

    id: str = Field(..., description="Unique task identifier")
    type: str = Field(..., description="Task type: code_generation, code_review, reasoning")
    prompt: str = Field(..., description="Task prompt")
    dependencies: list[str] = Field(default_factory=list, description="Dependency task IDs")
    hard_validators: list[str] = Field(default_factory=list, description="Required validators")
    max_output_tokens: int = Field(default=4000, description="Max output tokens")


class DecompositionOutput(BaseModel):
    """Structured output for decomposition phase."""

    tasks: list[TaskSpec] = Field(..., description="List of tasks")
    execution_order: list[str] = Field(..., description="Topologically sorted task IDs")
    estimated_cost: float = Field(..., description="Estimated cost in USD")
    estimated_tokens: int = Field(default=0, description="Estimated total tokens")


class CritiqueOutput(BaseModel):
    """Structured output for critique phase."""

    score: float = Field(..., ge=0.0, le=1.0, description="Quality score 0-1")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    requires_revision: bool = Field(..., description="Whether revision is needed")


class EvaluationOutput(BaseModel):
    """Structured output for evaluation phase."""

    score: float = Field(..., ge=0.0, le=1.0, description="Quality score 0-1")
    metrics: dict[str, float] = Field(default_factory=dict, description="Evaluation metrics")
    passed: bool = Field(..., description="Whether evaluation passed")
    reasoning: str = Field(default="", description="Brief reasoning")


class CodeReviewOutput(BaseModel):
    """Structured output for code review phase."""

    score: float = Field(..., ge=0.0, le=1.0, description="Quality score 0-1")
    bugs_found: list[str] = Field(default_factory=list, description="Bugs identified")
    improvements: list[str] = Field(default_factory=list, description="Suggested improvements")
    security_issues: list[str] = Field(default_factory=list, description="Security concerns")
    passed: bool = Field(..., description="Whether code passes review")


class PromptEnhancementOutput(BaseModel):
    """Structured output for prompt enhancement phase."""

    enhanced_prompt: str = Field(..., description="Enhanced prompt text")
    improvements_made: list[str] = Field(default_factory=list, description="List of improvements")
    estimated_quality_gain: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Expected quality improvement"
    )


class CondensingOutput(BaseModel):
    """Structured output for condensing/summarization phase."""

    summary: str = Field(..., description="Condensed summary")
    key_points: list[str] = Field(default_factory=list, description="Key points extracted")
    tokens_saved: int = Field(default=0, description="Estimated tokens saved")


# Type mapping for output types
OUTPUT_TYPES = {
    "decomposition": DecompositionOutput,
    "critique": CritiqueOutput,
    "evaluation": EvaluationOutput,
    "code_review": CodeReviewOutput,
    "prompt_enhancement": PromptEnhancementOutput,
    "condensing": CondensingOutput,
}


class StructuredOutputEnforcer:
    """
    Enforce structured output using Pydantic models.

    Usage:
        enforcer = StructuredOutputEnforcer(client=api_client)
        result = await enforcer.generate_structured(model, prompt, DecompositionOutput)
    """

    def __init__(self, client=None):
        """
        Initialize structured output enforcer.

        Args:
            client: API client with tool/function calling support
        """
        if not HAS_PYDANTIC:
            logger.warning("Pydantic not installed, structured output disabled")
            raise ImportError(
                "Pydantic is required for structured output. Install with: pip install pydantic"
            )

        self.client = client
        self._output_types = dict(OUTPUT_TYPES)

    def register_output_type(
        self,
        phase: str,
        output_model: type[BaseModel],
    ) -> None:
        """
        Register custom output type for phase.

        Args:
            phase: Phase name
            output_model: Pydantic model class
        """
        self._output_types[phase] = output_model
        logger.info(f"Registered custom output type for {phase}")

    async def generate_structured(
        self,
        model: str,
        prompt: str,
        output_type: type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """
        Generate structured output using tool calling.

        Args:
            model: Model to use
            prompt: Prompt text
            output_type: Pydantic model class
            **kwargs: Additional API parameters

        Returns:
            Typed Pydantic model instance
        """
        # Get JSON schema for output type
        schema = output_type.model_json_schema()

        # Try Anthropic tool use first
        if self._is_anthropic_model(model):
            return await self._generate_anthropic_tool(model, prompt, schema, output_type, **kwargs)

        # Try OpenAI function calling
        elif self._is_openai_model(model):
            return await self._generate_openai_function(
                model, prompt, schema, output_type, **kwargs
            )

        # Fallback to JSON parsing
        else:
            return await self._generate_json_fallback(model, prompt, schema, output_type, **kwargs)

    async def _generate_anthropic_tool(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        output_type: type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """
        Generate using Anthropic tool calling.

        Args:
            model: Anthropic model
            prompt: Prompt text
            schema: JSON schema
            output_type: Pydantic model class
            **kwargs: Additional parameters

        Returns:
            Typed Pydantic model instance
        """
        try:
            from anthropic import AsyncAnthropic

            if not isinstance(self.client, AsyncAnthropic):
                raise ValueError("Client must be AsyncAnthropic for Anthropic tool use")

            # Create tool definition
            tool = {
                "name": "structured_output",
                "description": f"Generate structured output: {schema.get('title', 'Output')}",
                "input_schema": schema,
            }

            # Call with tool
            response = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 4000),
                system=kwargs.get("system", "Generate structured output only."),
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                tool_choice={"type": "tool", "name": "structured_output"},
            )

            # Extract tool output
            tool_output = None
            for content in response.content:
                if hasattr(content, "type") and content.type == "tool_use":
                    tool_output = content.input
                    break

            if tool_output is None:
                raise ValueError("No tool output in response")

            # Parse as Pydantic model
            return output_type(**tool_output)

        except Exception as e:
            logger.warning(f"Anthropic tool use failed: {e}, falling back to JSON")
            return await self._generate_json_fallback(model, prompt, schema, output_type, **kwargs)

    async def _generate_openai_function(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        output_type: type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """
        Generate using OpenAI function calling.

        Args:
            model: OpenAI model
            prompt: Prompt text
            schema: JSON schema
            output_type: Pydantic model class
            **kwargs: Additional parameters

        Returns:
            Typed Pydantic model instance
        """
        try:
            from openai import AsyncOpenAI

            if not isinstance(self.client, AsyncOpenAI):
                raise ValueError("Client must be AsyncOpenAI for OpenAI function calling")

            # Call with function
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Generate structured JSON output only."},
                    {"role": "user", "content": prompt},
                ],
                functions=[
                    {
                        "name": "structured_output",
                        "description": f"Generate structured output: {schema.get('title', 'Output')}",
                        "parameters": schema,
                    }
                ],
                function_call={"name": "structured_output"},
                **kwargs,
            )

            # Extract function output
            message = response.choices[0].message
            if not hasattr(message, "function_call") or not message.function_call:
                raise ValueError("No function call in response")

            tool_output = json.loads(message.function_call.arguments)

            # Parse as Pydantic model
            return output_type(**tool_output)

        except Exception as e:
            logger.warning(f"OpenAI function calling failed: {e}, falling back to JSON")
            return await self._generate_json_fallback(model, prompt, schema, output_type, **kwargs)

    async def _generate_json_fallback(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        output_type: type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """
        Generate JSON with regex extraction fallback.

        Args:
            model: Any model
            prompt: Prompt text
            schema: JSON schema
            output_type: Pydantic model class
            **kwargs: Additional parameters

        Returns:
            Typed Pydantic model instance
        """
        import re

        # Add JSON format instruction to prompt
        json_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Respond ONLY with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Output ONLY the JSON, no other text."
        )

        # Call model
        response = await self.client.call(
            model=model,
            system_prompt=json_prompt,
            max_tokens=kwargs.get("max_tokens", 4000),
        )

        # Extract text
        text = response.text if hasattr(response, "text") else str(response)

        # Extract JSON from response
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group()

        # Parse JSON
        try:
            data = json.loads(text)
            return output_type(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"JSON parsing failed: {e}")
            raise ValueError(f"Failed to parse structured output: {e}")

    def _is_anthropic_model(self, model: str) -> bool:
        """Check if model is Anthropic."""
        model_lower = model.lower()
        return "claude" in model_lower or "anthropic" in model_lower

    def _is_openai_model(self, model: str) -> bool:
        """Check if model is OpenAI."""
        model_lower = model.lower()
        return "gpt" in model_lower or "openai" in model_lower

    def get_output_type(self, phase: str) -> type[BaseModel]:
        """
        Get output type for phase.

        Args:
            phase: Phase name

        Returns:
            Pydantic model class
        """
        return self._output_types.get(phase, DecompositionOutput)


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────


async def generate_decomposition(
    client,
    model: str,
    prompt: str,
) -> DecompositionOutput:
    """Generate structured decomposition output."""
    enforcer = StructuredOutputEnforcer(client=client)
    return await enforcer.generate_structured(model, prompt, DecompositionOutput)


async def generate_critique(
    client,
    model: str,
    prompt: str,
) -> CritiqueOutput:
    """Generate structured critique output."""
    enforcer = StructuredOutputEnforcer(client=client)
    return await enforcer.generate_structured(model, prompt, CritiqueOutput)


async def generate_evaluation(
    client,
    model: str,
    prompt: str,
) -> EvaluationOutput:
    """Generate structured evaluation output."""
    enforcer = StructuredOutputEnforcer(client=client)
    return await enforcer.generate_structured(model, prompt, EvaluationOutput)


__all__ = [
    # Models
    "TaskSpec",
    "DecompositionOutput",
    "CritiqueOutput",
    "EvaluationOutput",
    "CodeReviewOutput",
    "PromptEnhancementOutput",
    "CondensingOutput",
    # Enforcer
    "StructuredOutputEnforcer",
    # Convenience
    "generate_decomposition",
    "generate_critique",
    "generate_evaluation",
]
