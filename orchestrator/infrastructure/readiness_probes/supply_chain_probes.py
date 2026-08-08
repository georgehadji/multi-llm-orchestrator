"""Supply-chain hardening probes (Phase 7, P-2).

Named checks from the design doc: a lock file present, a base image pinned
by digest, every GitHub Actions step pinned to a 40-hex SHA, and a
parseable SBOM.
"""

from __future__ import annotations

import re

from ...domain.readiness import Evidence, ProbeResult, RequirementOutcome
from ._base import StaticProbe
from ._parsing import dockerfile_from_lines, find_workflow_files, parse_json, parse_yaml

_SHA_PINNED_RE = re.compile(r"@[0-9a-f]{40}$")
_SBOM_CANDIDATES = ("sbom.json", "sbom.cdx.json", "cyclonedx.json")


class LockFilePresentProbe(StaticProbe):
    requirement_id = "supply-chain.lock-file-present"
    _LOCK_NAMES = ("uv.lock", "poetry.lock")

    def check(self, workspace):
        pyproject = workspace.root / "pyproject.toml"
        if not pyproject.is_file():
            return RequirementOutcome(
                requirement_id=self.requirement_id, result=ProbeResult.NOT_APPLICABLE
            )
        for name in self._LOCK_NAMES:
            if (workspace.root / name).is_file():
                return RequirementOutcome(
                    requirement_id=self.requirement_id,
                    result=ProbeResult.SATISFIED,
                    evidence=(Evidence(probe=self.requirement_id, detail=f"{name} present"),),
                )
        requirements_txt = workspace.root / "requirements.txt"
        if requirements_txt.is_file() and "--hash=" in requirements_txt.read_text(encoding="utf-8"):
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.SATISFIED,
                evidence=(
                    Evidence(probe=self.requirement_id, detail="requirements.txt is hash-pinned"),
                ),
            )
        return RequirementOutcome(
            requirement_id=self.requirement_id,
            result=ProbeResult.VIOLATED,
            evidence=(
                Evidence(
                    probe=self.requirement_id,
                    detail="no uv.lock, poetry.lock, or hash-pinned requirements.txt found",
                ),
            ),
        )


class BaseImageDigestPinnedProbe(StaticProbe):
    requirement_id = "supply-chain.base-image-digest-pinned"

    def check(self, workspace):
        path = workspace.root / "Dockerfile"
        if not path.is_file():
            return RequirementOutcome(
                requirement_id=self.requirement_id, result=ProbeResult.NOT_APPLICABLE
            )
        refs, err = dockerfile_from_lines(path)
        if err is not None:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.INDETERMINATE,
                evidence=(Evidence(probe=self.requirement_id, detail=err, location=str(path)),),
            )
        if not refs:
            return RequirementOutcome(
                requirement_id=self.requirement_id, result=ProbeResult.NOT_APPLICABLE
            )
        evidence = [
            Evidence(
                probe=self.requirement_id,
                detail=f"not digest-pinned: {image_ref}",
                location=f"{path}:{lineno}",
            )
            for lineno, image_ref in refs
            if "@sha256:" not in image_ref
        ]
        result = ProbeResult.VIOLATED if evidence else ProbeResult.SATISFIED
        return RequirementOutcome(
            requirement_id=self.requirement_id, result=result, evidence=tuple(evidence)
        )


class ActionsShaPinnedProbe(StaticProbe):
    requirement_id = "supply-chain.actions-sha-pinned"

    def _find_uses(self, node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "uses" and isinstance(value, str):
                    yield path, value
                else:
                    yield from self._find_uses(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                yield from self._find_uses(item, f"{path}[{i}]")

    def check(self, workspace):
        workflows = find_workflow_files(workspace.root)
        if not workflows:
            return RequirementOutcome(
                requirement_id=self.requirement_id, result=ProbeResult.NOT_APPLICABLE
            )
        evidence = []
        indeterminate = False
        for wf in workflows:
            data, err = parse_yaml(wf)
            if err is not None:
                indeterminate = True
                evidence.append(Evidence(probe=self.requirement_id, detail=err, location=str(wf)))
                continue
            for _, uses in self._find_uses(data):
                if uses.startswith("./"):
                    continue  # local action, not a supply-chain surface
                if not _SHA_PINNED_RE.search(uses):
                    evidence.append(
                        Evidence(
                            probe=self.requirement_id,
                            detail=f"not SHA-pinned: {uses}",
                            location=str(wf),
                        )
                    )
        if any(e.detail.startswith("not SHA-pinned") for e in evidence):
            result = ProbeResult.VIOLATED
        elif indeterminate:
            result = ProbeResult.INDETERMINATE
        else:
            result = ProbeResult.SATISFIED
        return RequirementOutcome(
            requirement_id=self.requirement_id, result=result, evidence=tuple(evidence)
        )


class SbomPresentProbe(StaticProbe):
    requirement_id = "supply-chain.sbom-present"

    def check(self, workspace):
        found = next(
            (
                workspace.root / name
                for name in _SBOM_CANDIDATES
                if (workspace.root / name).is_file()
            ),
            None,
        )
        if found is None:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.VIOLATED,
                evidence=(Evidence(probe=self.requirement_id, detail="no SBOM file found"),),
            )
        data, err = parse_json(found)
        if err is not None or not isinstance(data, dict):
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.INDETERMINATE,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail=err or "not a JSON object",
                        location=str(found),
                    ),
                ),
            )
        if "bomFormat" not in data and "spdxVersion" not in data:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.VIOLATED,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail="parses but declares neither CycloneDX bomFormat nor SPDX spdxVersion",
                        location=str(found),
                    ),
                ),
            )
        return RequirementOutcome(
            requirement_id=self.requirement_id,
            result=ProbeResult.SATISFIED,
            evidence=(
                Evidence(probe=self.requirement_id, detail="valid SBOM", location=str(found)),
            ),
        )


SUPPLY_CHAIN_PROBES = (
    LockFilePresentProbe(),
    BaseImageDigestPinnedProbe(),
    ActionsShaPinnedProbe(),
    SbomPresentProbe(),
)

__all__ = [
    "LockFilePresentProbe",
    "BaseImageDigestPinnedProbe",
    "ActionsShaPinnedProbe",
    "SbomPresentProbe",
    "SUPPLY_CHAIN_PROBES",
]
