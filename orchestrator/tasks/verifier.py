"""
Task Completion Verification — Backward-compatibility shim
==========================================================
The canonical implementation lives in :mod:`orchestrator.task_verifier`. This
module was previously a byte-for-byte duplicate whose verbatim ``.log_config``
import resolved to a nonexistent ``orchestrator.tasks.log_config`` path, leaving
the whole ``orchestrator.tasks`` package unimportable. It now re-exports the
canonical module so there is a single source of truth.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from ..task_verifier import (  # noqa: F401
    Discrepancy,
    DiscrepancyType,
    ExpectedOutcome,
    TaskVerifier,
    VerificationResult,
    VerificationSeverity,
)

__all__ = [
    "Discrepancy",
    "DiscrepancyType",
    "ExpectedOutcome",
    "TaskVerifier",
    "VerificationResult",
    "VerificationSeverity",
]
