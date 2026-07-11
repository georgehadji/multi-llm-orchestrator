"""
Ingest Adapters — External Artifact Intake
============================================
Author: Orchestrator core

Provides adapters that consume structured specification artifacts from
external tools (notably Spec-Kit) and convert them into orchestrator's
native ``Task`` and ``ProjectConstitution`` data models.

Package architecture
--------------------
- ``speckit_adapter.py`` — Spec-Kit Mode A ingestion (tasks.md, spec.md,
  plan.md, constitution.md → orchestrator domain types).
- ``domain/ports.py`` — ``FileReaderPort`` used by all ingest adapters.

Every adapter in this package:
  - Lives at the application / engine-core boundary (pure data transforms).
  - Exposes a ``.load(dir) -> (dict[str, Task], ProjectConstitution, hints)``
    interface so the orchestrator entry-point can branch cleanly.
  - Never opens files directly — delegates to ``FileReaderPort``.
  - Fails fast on malformed input with file+line diagnostics.
"""

from __future__ import annotations

from .speckit_adapter import SpecArtifacts, SpecKitAdapter

__all__ = [
    "SpecArtifacts",
    "SpecKitAdapter",
]
