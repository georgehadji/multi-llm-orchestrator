"""Archetype adapter over the existing AppDetector (Phase 7, P-1).

Reuses ``appbuilder/detector.py`` — no second detector, no second source of
truth about what kind of app this is (plan §3.5.1). This module only maps
the detector's ``AppProfile.app_type`` string onto the readiness domain's
``AppArchetype`` enum.
"""

from __future__ import annotations

from ...domain.readiness import AppArchetype

_APP_TYPE_TO_ARCHETYPE: dict[str, AppArchetype] = {
    "fastapi": AppArchetype.PYTHON_SERVICE,
    "flask": AppArchetype.PYTHON_SERVICE,
    "react-fastapi": AppArchetype.FULLSTACK,
    "nextjs": AppArchetype.FULLSTACK,
    "cli": AppArchetype.PYTHON_CLI,
    "script": AppArchetype.PYTHON_CLI,
    "generic": AppArchetype.PYTHON_CLI,
    "library": AppArchetype.LIBRARY,
}


def archetype_for(app_type: str) -> AppArchetype:
    """Map an ``AppProfile.app_type`` string to its readiness archetype.

    Unknown types fall back to PYTHON_CLI — the least demanding rubric
    (plan §3.5.4 default thresholds), so an unrecognized app type is never
    silently held to a bar it may not be able to meet. ``WEB_STATIC`` is
    unreachable from the current detector vocabulary; web-only output is
    archetyped elsewhere once P-9's web pipeline lands.
    """
    return _APP_TYPE_TO_ARCHETYPE.get(app_type, AppArchetype.PYTHON_CLI)


__all__ = ["archetype_for"]
