"""
AuditLog — Backward-compatibility shim
========================================
Canonical location: orchestrator/infrastructure/audit.py
Import from there directly for new code; this shim exists for existing callers.
"""
from .infrastructure.audit import AuditLog, AuditRecord  # noqa: F401

__all__ = ["AuditLog", "AuditRecord"]
