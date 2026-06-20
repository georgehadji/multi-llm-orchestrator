"""
Context Compressor — Backward-compatibility shim.
===================================================
The canonical implementation is in orchestrator/context_compressor.py.
Import from there directly for new code; this module exists only for callers
that reference orchestrator.application.context_compressor.
"""

from orchestrator.context_compressor import ContextCompressor

__all__ = ["ContextCompressor"]
