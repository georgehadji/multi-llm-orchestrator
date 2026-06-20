"""
Task Handler Protocol — Backward-compatibility shim
===================================================
The canonical handlers live in :mod:`orchestrator.task_handlers`. This module
was previously a byte-for-byte duplicate whose verbatim ``.models`` /
``.api_clients`` imports resolved to nonexistent ``orchestrator.tasks.*`` paths,
leaving the whole ``orchestrator.tasks`` package unimportable. It now re-exports
the canonical module so there is a single source of truth.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from ..task_handlers import (  # noqa: F401
    CodeGenerationHandler,
    CodeReviewHandler,
    EvaluationHandler,
    ReasoningHandler,
    TaskHandler,
    get_handler,
    get_handler_or_none,
    register,
    registered_types,
)

__all__ = [
    "CodeGenerationHandler",
    "CodeReviewHandler",
    "EvaluationHandler",
    "ReasoningHandler",
    "TaskHandler",
    "get_handler",
    "get_handler_or_none",
    "register",
    "registered_types",
]
