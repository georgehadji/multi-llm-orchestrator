"""
ScaffoldEngine — creates folder structure and boilerplate for a new app.
Author: Georgios-Chrysovalantis Chatzivantsidis

Usage:
    engine = ScaffoldEngine()
    scaffold_dict = engine.scaffold(profile, output_dir)
    # scaffold_dict: dict[str, str] — relative path -> file content
"""
from __future__ import annotations

import logging
from pathlib import Path

from orchestrator.app_detector import AppProfile

from .templates import cli, fastapi, generic, library

logger = logging.getLogger(__name__)

# Map app_type -> template FILES dict
_TEMPLATE_MAP: dict[str, dict[str, str]] = {
    "fastapi": fastapi.FILES,
    "flask": generic.FILES,   # use generic as fallback for flask
    "cli": cli.FILES,
    "library": library.FILES,
    "script": generic.FILES,
    "react-fastapi": fastapi.FILES,  # partial — full-stack not fully supported in Phase 1
    "nextjs": generic.FILES,
    "generic": generic.FILES,
}


class ScaffoldEngine:
    """
    Creates the initial file structure for an app from an AppProfile.

    scaffold() returns dict[str, str] (relative_path -> content) and
    writes files to output_dir, skipping files that already exist.
    """

    def scaffold(self, profile: AppProfile, output_dir: Path) -> dict[str, str]:
        """
        Create the scaffold for the given profile in output_dir.

        Returns only files that were actually written (skips pre-existing files).
        Files that already exist on disk are NOT overwritten.
        """
        template = _TEMPLATE_MAP.get(profile.app_type)
        if template is None:
            logger.warning(
                "No template for app_type '%s'; using generic fallback", profile.app_type
            )
            template = generic.FILES

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written: dict[str, str] = {}
        for rel_path, content in template.items():
            dest = output_dir / rel_path
            if dest.exists():
                logger.debug("Skipping existing file: %s", rel_path)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            logger.debug("Scaffolded: %s", rel_path)
            written[rel_path] = content

        return written
