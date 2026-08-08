"""CycloneDX SBOM emission for the Docker emitter (Phase 7, P-4).

Matches ``SbomPresentProbe`` (P-2): filename ``sbom.cdx.json``, a JSON
object declaring ``bomFormat``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

_SBOM_FILENAME = "sbom.cdx.json"


def build_cyclonedx_sbom(
    project_name: str, project_version: str, components: Sequence[tuple[str, str]]
) -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {"type": "application", "name": project_name, "version": project_version}
        },
        "components": [
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{name}@{version}",
            }
            for name, version in components
        ],
    }


def write_sbom(root: Path, sbom: dict) -> Path:
    path = root / _SBOM_FILENAME
    path.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["build_cyclonedx_sbom", "write_sbom"]
