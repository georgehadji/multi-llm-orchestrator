"""
Objective verifiers for routing-quality gates.

Provides a ``Verifier`` protocol and concrete implementations:
json-schema validation, Python AST compile-check, regex assertion,
and a ``CompositeVerifier`` that weights multiple sub-verifiers.

Usage:
    from orchestrator.verification import Verifier, Verdict
    from orchestrator.verification.python_ast import PythonASTVerifier

    verifier = PythonASTVerifier()
    verdict = await verifier.verify(prompt="...", response="...", task_type=...)
"""

from orchestrator.verification.port import Verifier
from orchestrator.models import Verdict

__all__ = [
    "Verifier",
    "Verdict",
]
