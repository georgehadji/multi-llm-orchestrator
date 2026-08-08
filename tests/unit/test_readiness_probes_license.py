"""Tests for licence compliance probes (Phase 7, P-2)."""

from __future__ import annotations

import pytest

from orchestrator.domain.ports import ProbeContext
from orchestrator.domain.readiness import AppArchetype, ProbeResult
from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.readiness_probes.license_probes import (
    LicenseFilePresentProbe,
    LicenseMatchesDeclarationProbe,
)

CTX = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)

MIT_TEXT = "MIT License\n\nPermission is hereby granted, free of charge, ...\n"


def _ws(root):
    return Workspace(root=root, framework="python")


@pytest.mark.unit
class TestLicenseFilePresentProbe:
    async def test_violated_when_absent(self, tmp_path):
        outcome = await LicenseFilePresentProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_violated_when_empty(self, tmp_path):
        (tmp_path / "LICENSE").write_text("")
        outcome = await LicenseFilePresentProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_satisfied_when_present(self, tmp_path):
        (tmp_path / "LICENSE").write_text(MIT_TEXT)
        outcome = await LicenseFilePresentProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED


@pytest.mark.unit
class TestLicenseMatchesDeclarationProbe:
    async def test_not_applicable_without_pyproject(self, tmp_path):
        outcome = await LicenseMatchesDeclarationProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_not_applicable_when_no_declared_license(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        outcome = await LicenseMatchesDeclarationProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_satisfied_when_matches(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\nlicense = {text = "MIT"}\n')
        (tmp_path / "LICENSE").write_text(MIT_TEXT)
        outcome = await LicenseMatchesDeclarationProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_mismatched(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\nlicense = {text = "MIT"}\n')
        (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0\n")
        outcome = await LicenseMatchesDeclarationProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_indeterminate_when_pyproject_malformed(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("not [ valid")
        outcome = await LicenseMatchesDeclarationProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.INDETERMINATE
