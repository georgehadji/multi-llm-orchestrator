"""Artifact emitters (Phase 7, P-4+): write production artifacts into a
workspace via ``ArtifactEmitterPort``. Implementations wrap existing
generators rather than reimplementing them — see each module's docstring.
"""

from __future__ import annotations

from .cicd_emitter import CicdEmitter
from .docker_emitter import DockerEmitter
from .observability_emitter import ObservabilityEmitter

ALL_EMITTERS = (DockerEmitter(), CicdEmitter(), ObservabilityEmitter())

__all__ = ["ALL_EMITTERS", "CicdEmitter", "DockerEmitter", "ObservabilityEmitter"]
