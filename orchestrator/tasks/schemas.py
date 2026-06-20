"""
Task Output Schemas (compatibility shim)
========================================
The canonical definitions live in :mod:`orchestrator.task_schemas`. This module
previously held a byte-for-byte duplicate, which let bugs diverge between the two
copies (e.g. the value-vs-name schema lookup and the strict-mode flag). It now
re-exports the canonical module so there is a single source of truth.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from ..task_schemas import (  # noqa: F401
    TASK_OUTPUT_SCHEMAS,
    CodeGenerationOutput,
    CodeReviewOutput,
    DataExtractionOutput,
    EvaluationOutput,
    ReasoningOutput,
    SummarizationOutput,
    WritingOutput,
    generate_openrouter_schema,
    get_schema_for_task_type,
)

__all__ = [
    "TASK_OUTPUT_SCHEMAS",
    "CodeGenerationOutput",
    "CodeReviewOutput",
    "DataExtractionOutput",
    "EvaluationOutput",
    "ReasoningOutput",
    "SummarizationOutput",
    "WritingOutput",
    "generate_openrouter_schema",
    "get_schema_for_task_type",
]
