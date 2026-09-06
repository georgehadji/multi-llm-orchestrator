"""GeneratorService — re-export shim from application/decomposer.py (renamed
to DecomposerService there; same backward-compat aliasing services/__init__.py
already does, applied here too so this submodule stops shadowing it with a
stale, independently-defined 251-line copy of a now-713-line class)."""

from ..application.decomposer import DecomposerMetrics as GeneratorMetrics
from ..application.decomposer import DecomposerResult as GeneratorResult
from ..application.decomposer import DecomposerService as GeneratorService

__all__ = ["GeneratorMetrics", "GeneratorResult", "GeneratorService"]
