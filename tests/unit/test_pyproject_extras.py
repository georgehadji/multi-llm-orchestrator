"""The `ide` extra and the CI installs that depend on it.

`ide_backend` imports fastapi at package import and uvicorn when it serves. CI
installs `.[dev,ide]` instead of repeating a loose pin, so the extra must stay
complete, must not drift from the `dashboard` pin, and the Test job's fallback
install must still bring fastapi in.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from orchestrator.crosscutting.toml_compat import tomllib

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _requirement(extra: str, name: str) -> str | None:
    for req in _extras()[extra]:
        if re.split(r"[<>=!~\[; ]", req, maxsplit=1)[0].lower() == name:
            return req
    return None


def test_ide_extra_can_serve_the_ide_backend() -> None:
    why = "ide_backend imports uvicorn to serve; `pip install .[ide]` must provide it"
    assert _requirement("ide", "fastapi") is not None
    assert _requirement("ide", "uvicorn") is not None, why


def test_fastapi_pin_is_the_same_in_ide_and_dashboard() -> None:
    ide, dashboard = _requirement("ide", "fastapi"), _requirement("dashboard", "fastapi")
    assert ide == dashboard, "bump the fastapi pin in both the `ide` and `dashboard` extras"


def test_test_job_fallback_install_still_brings_in_fastapi() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lines = [ln for ln in ci.splitlines() if "[dev,ide]" in ln and "||" in ln]
    assert lines, "expected the Test job install line with a fallback"
    for line in lines:
        fallback = line.split("||", 1)[1]
        assert ".[ide]" in fallback, f"fallback install drops fastapi: {line.strip()}"
