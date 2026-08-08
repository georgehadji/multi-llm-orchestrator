"""Shared parse helpers for static readiness probes (Phase 7, P-2).

Every helper returns ``(value, error)`` instead of raising — a probe that
cannot parse its input must degrade to ``INDETERMINATE`` evidence, never
crash and never be mistaken for ``VIOLATED`` (a malformed workflow is not
the same claim as a missing permissions block).
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml


def read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, str(exc)


def parse_toml(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    text, err = read_text(path)
    if text is None:
        return None, err
    try:
        return tomllib.loads(text), None
    except tomllib.TOMLDecodeError as exc:
        return None, str(exc)


def parse_json(path: Path) -> tuple[Any | None, str | None]:
    text, err = read_text(path)
    if text is None:
        return None, err
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def parse_yaml(path: Path) -> tuple[Any | None, str | None]:
    text, err = read_text(path)
    if text is None:
        return None, err
    try:
        return yaml.safe_load(text), None
    except yaml.YAMLError as exc:
        return None, str(exc)


_FROM_RE = re.compile(r"^\s*FROM\s+(\S+)", re.IGNORECASE)


def dockerfile_from_lines(path: Path) -> tuple[list[tuple[int, str]] | None, str | None]:
    """Return ``[(line_no, image_ref), ...]`` for every ``FROM`` directive."""
    text, err = read_text(path)
    if text is None:
        return None, err
    refs = []
    for i, line in enumerate(text.splitlines(), start=1):
        m = _FROM_RE.match(line)
        if m:
            refs.append((i, m.group(1)))
    return refs, None


def find_workflow_files(root: Path) -> list[Path]:
    workflows_dir = root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        return []
    return sorted(p for p in workflows_dir.iterdir() if p.suffix in (".yml", ".yaml"))
