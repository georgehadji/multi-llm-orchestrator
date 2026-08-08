"""Licence compliance probes (Phase 7, P-2 detection half of G-14).

Detection only: does a LICENSE file exist, and does it agree with what
pyproject declares. Emitting a correct LICENSE file is P-10's job.
"""

from __future__ import annotations

from ...domain.readiness import Evidence, ProbeResult, RequirementOutcome
from ._base import StaticProbe
from ._parsing import parse_toml

_LICENSE_FILENAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt")

_KEYWORDS = {
    "MIT": ("mit license", "permission is hereby granted"),
    "Apache-2.0": ("apache license", "version 2.0"),
    "BSD-3-Clause": ("bsd 3-clause", "redistribution and use"),
    "BSD-2-Clause": ("bsd 2-clause", "redistribution and use"),
    "GPL-3.0": ("gnu general public license", "version 3"),
    "AGPL-3.0": ("gnu affero general public license",),
    "ISC": ("isc license",),
}


def _find_license_file(root):
    for name in _LICENSE_FILENAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _declared_license(pyproject_data):
    project = pyproject_data.get("project", {}) if isinstance(pyproject_data, dict) else {}
    license_field = project.get("license")
    if isinstance(license_field, str):
        return license_field
    if isinstance(license_field, dict):
        return license_field.get("text")
    return None


class LicenseFilePresentProbe(StaticProbe):
    requirement_id = "compliance.license-file-present"

    def check(self, workspace):
        found = _find_license_file(workspace.root)
        if found is None:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.VIOLATED,
                evidence=(Evidence(probe=self.requirement_id, detail="no LICENSE file found"),),
            )
        if not found.read_text(encoding="utf-8").strip():
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.VIOLATED,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail="LICENSE file is empty",
                        location=str(found),
                    ),
                ),
            )
        return RequirementOutcome(
            requirement_id=self.requirement_id,
            result=ProbeResult.SATISFIED,
            evidence=(Evidence(probe=self.requirement_id, detail="present", location=str(found)),),
        )


class LicenseMatchesDeclarationProbe(StaticProbe):
    requirement_id = "compliance.license-matches-declaration"

    def check(self, workspace):
        pyproject = workspace.root / "pyproject.toml"
        if not pyproject.is_file():
            return RequirementOutcome(
                requirement_id=self.requirement_id, result=ProbeResult.NOT_APPLICABLE
            )
        data, err = parse_toml(pyproject)
        if err is not None:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.INDETERMINATE,
                evidence=(
                    Evidence(probe=self.requirement_id, detail=err, location=str(pyproject)),
                ),
            )
        declared = _declared_license(data)
        if not declared:
            return RequirementOutcome(
                requirement_id=self.requirement_id, result=ProbeResult.NOT_APPLICABLE
            )
        found = _find_license_file(workspace.root)
        if found is None:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.VIOLATED,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail=f"pyproject declares {declared!r} but no LICENSE file exists",
                    ),
                ),
            )
        keywords = _KEYWORDS.get(declared)
        if keywords is None:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.INDETERMINATE,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail=f"unrecognized declared licence {declared!r}",
                    ),
                ),
            )
        text = found.read_text(encoding="utf-8").lower()
        if all(kw in text for kw in keywords):
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.SATISFIED,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail=f"LICENSE matches declared {declared!r}",
                        location=str(found),
                    ),
                ),
            )
        return RequirementOutcome(
            requirement_id=self.requirement_id,
            result=ProbeResult.VIOLATED,
            evidence=(
                Evidence(
                    probe=self.requirement_id,
                    detail=f"LICENSE content doesn't match declared {declared!r}",
                    location=str(found),
                ),
            ),
        )


LICENSE_PROBES = (
    LicenseFilePresentProbe(),
    LicenseMatchesDeclarationProbe(),
)

__all__ = ["LicenseFilePresentProbe", "LicenseMatchesDeclarationProbe", "LICENSE_PROBES"]
