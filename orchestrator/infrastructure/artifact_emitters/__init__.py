"""Artifact emitters (Phase 7, P-4+): write production artifacts into a
workspace via ``ArtifactEmitterPort``. Implementations wrap existing
generators rather than reimplementing them — see each module's docstring.
"""

from __future__ import annotations

from .docker_emitter import DockerEmitter

ALL_EMITTERS = (DockerEmitter(),)

__all__ = ["ALL_EMITTERS", "DockerEmitter"]
