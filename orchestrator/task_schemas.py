"""
Task Output Schemas
===================
Pydantic models defining structured output formats for each TaskType.
Used with OpenRouter's response_format: json_schema feature.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from typing import Any, Literal
from pydantic import BaseModel, Field

from .models import TaskType


class CodeGenerationOutput(BaseModel):
    """Schema for CODE_GEN task outputs."""

    code: str = Field(
        ...,
        description="The generated source code",
        min_length=1,
    )

    language: str = Field(
        ...,
        description="Programming language of the generated code (e.g., 'python', 'javascript')",
        examples=["python", "javascript", "typescript", "rust", "go"],
    )

    explanation: str | None = Field(
        None,
        description="Optional explanation of the code's functionality",
    )

    imports: list[str] = Field(
        default_factory=list,
        description="List of required imports/libraries",
    )

    tests: list[str] = Field(
        default_factory=list,
        description="Optional test cases demonstrating usage",
    )


class CodeReviewOutput(BaseModel):
    """Schema for CODE_REVIEW task outputs."""

    overall_score: int = Field(
        ...,
        description="Overall code quality score (1-10)",
        ge=1,
        le=10,
    )

    summary: str = Field(
        ...,
        description="Brief summary of the review findings",
    )

    issues: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of identified issues",
    )

    suggestions: list[str] = Field(
        default_factory=list,
        description="Improvement suggestions",
    )

    strengths: list[str] = Field(
        default_factory=list,
        description="Notable strengths of the code",
    )


class ReasoningOutput(BaseModel):
    """Schema for REASONING task outputs."""

    conclusion: str = Field(
        ...,
        description="The final conclusion or answer",
    )

    reasoning_steps: list[str] = Field(
        ...,
        description="Step-by-step reasoning process",
        min_length=1,
    )

    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence level in the conclusion",
    )

    caveats: list[str] = Field(
        default_factory=list,
        description="Potential caveats or limitations",
    )


class WritingOutput(BaseModel):
    """Schema for WRITING task outputs."""

    content: str = Field(
        ...,
        description="The written content",
        min_length=1,
    )

    title: str | None = Field(
        None,
        description="Title or heading for the content",
    )

    tone: Literal["formal", "informal", "technical", "creative", "persuasive"] = Field(
        ...,
        description="Tone of the writing",
    )

    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms or concepts covered",
    )


class DataExtractionOutput(BaseModel):
    """Schema for DATA_EXTRACT task outputs."""

    extracted_data: dict[str, Any] = Field(
        ...,
        description="The extracted structured data",
    )

    data_type: str = Field(
        ...,
        description="Type of data extracted (e.g., 'contact_info', 'product_specs')",
    )

    confidence_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence score (0-1) for each extracted field",
    )

    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields that could not be extracted",
    )


class SummarizationOutput(BaseModel):
    """Schema for SUMMARIZE task outputs."""

    summary: str = Field(
        ...,
        description="The condensed summary",
        min_length=1,
    )

    key_points: list[str] = Field(
        ...,
        description="Bullet points of main ideas",
        min_length=1,
    )

    word_count: int = Field(
        ...,
        description="Word count of the summary",
        ge=1,
    )

    original_word_count: int | None = Field(
        None,
        description="Word count of the original text (if known)",
    )


class EvaluationOutput(BaseModel):
    """Schema for EVALUATE task outputs."""

    score: float = Field(
        ...,
        description="Quality score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    passed: bool = Field(
        ...,
        description="Whether the evaluated item meets quality threshold",
    )

    feedback: str = Field(
        ...,
        description="Detailed evaluation feedback",
    )

    criteria_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores for individual evaluation criteria",
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="Specific recommendations for improvement",
    )


# Mapping of TaskType to output schema
TASK_OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "CODE_GEN": CodeGenerationOutput,
    "CODE_REVIEW": CodeReviewOutput,
    "REASONING": ReasoningOutput,
    "WRITING": WritingOutput,
    "DATA_EXTRACT": DataExtractionOutput,
    "SUMMARIZE": SummarizationOutput,
    "EVALUATE": EvaluationOutput,
}


def _resolve_schema_key(task_type: "str | TaskType") -> str | None:
    """Resolve a task-type identifier to the registry key (the enum *name*).

    ``TASK_OUTPUT_SCHEMAS`` is keyed by enum name (e.g. ``"DATA_EXTRACT"``), but
    callers across layers may pass the enum itself, its name, or its lowercase
    value (e.g. ``"data_extraction"``). Accept all three so a value-keyed call
    site can never silently miss the schema.

    Returns the canonical registry key, or ``None`` if it does not map to a
    schema-backed task type.
    """
    if isinstance(task_type, TaskType):
        return task_type.name if task_type.name in TASK_OUTPUT_SCHEMAS else None
    if not isinstance(task_type, str):
        return None
    if task_type in TASK_OUTPUT_SCHEMAS:  # already an enum name
        return task_type
    if task_type.upper() in TASK_OUTPUT_SCHEMAS:  # case-insensitive name
        return task_type.upper()
    for member in TaskType:  # enum value, e.g. "data_extraction"
        if task_type == member.value and member.name in TASK_OUTPUT_SCHEMAS:
            return member.name
    return None


def get_schema_for_task_type(task_type: "str | TaskType") -> type[BaseModel] | None:
    """Get the appropriate output schema for a task type.

    Args:
        task_type: A ``TaskType`` enum, its name, or its value.

    Returns:
        The Pydantic model class for the task output, or None if not found
    """
    key = _resolve_schema_key(task_type)
    return TASK_OUTPUT_SCHEMAS.get(key) if key else None


def generate_openrouter_schema(task_type: "str | TaskType") -> dict[str, Any] | None:
    """Generate OpenRouter-compatible JSON schema for a task type.

    Args:
        task_type: A ``TaskType`` enum, its name, or its value.

    Returns:
        OpenRouter response_format schema dict, or None if task type unknown

    Note:
        ``strict`` is ``False`` on purpose. These output schemas contain
        free-form ``dict`` fields (e.g. ``extracted_data: dict[str, Any]``),
        which OpenAI/Azure reject under strict json_schema mode. Non-strict
        mode supplies the schema as guidance while tolerating minor JSON
        deviations — exactly the case the OpenRouter ``response-healing``
        plugin repairs server-side.
    """
    key = _resolve_schema_key(task_type)
    schema_class = TASK_OUTPUT_SCHEMAS.get(key) if key else None
    if not schema_class:
        return None

    json_schema = schema_class.model_json_schema()

    # OpenRouter expects this specific format
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{key.lower()}_output",
            "schema": json_schema,
            "strict": False,
        },
    }
