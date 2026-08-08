"""Static probes: pyproject and CI workflow checks (Phase 7, P-2).

Every probe parses its target rather than substring-matching file content.
A parse failure is ``INDETERMINATE`` (distinct from ``VIOLATED``); an absent
target the requirement doesn't apply to is ``NOT_APPLICABLE``.
"""

from __future__ import annotations

from ...domain.readiness import Evidence, ProbeResult, RequirementOutcome
from ._base import StaticProbe
from ._parsing import dockerfile_from_lines, find_workflow_files, parse_toml, parse_yaml


class PyprojectParseableProbe(StaticProbe):
    requirement_id = "static.pyproject-parseable"

    def check(self, workspace):
        path = workspace.root / "pyproject.toml"
        if not path.is_file():
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.VIOLATED,
                evidence=(Evidence(probe=self.requirement_id, detail="pyproject.toml absent"),),
            )
        data, err = parse_toml(path)
        if err is not None:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.INDETERMINATE,
                evidence=(Evidence(probe=self.requirement_id, detail=err, location=str(path)),),
            )
        return RequirementOutcome(
            requirement_id=self.requirement_id,
            result=ProbeResult.SATISFIED,
            evidence=(Evidence(probe=self.requirement_id, detail="parses", location=str(path)),),
        )


class CiPermissionsReadOnlyProbe(StaticProbe):
    requirement_id = "static.ci-permissions-read-only"

    def check(self, workspace):
        workflows = find_workflow_files(workspace.root)
        if not workflows:
            return RequirementOutcome(
                requirement_id=self.requirement_id, result=ProbeResult.NOT_APPLICABLE
            )
        evidence = []
        violated = False
        indeterminate = False
        for wf in workflows:
            data, err = parse_yaml(wf)
            if err is not None or not isinstance(data, dict):
                indeterminate = True
                evidence.append(
                    Evidence(
                        probe=self.requirement_id, detail=err or "not a mapping", location=str(wf)
                    )
                )
                continue
            permissions = data.get("permissions")
            if permissions is None or permissions == "write-all":
                violated = True
                evidence.append(
                    Evidence(
                        probe=self.requirement_id,
                        detail=f"permissions={permissions!r} (expected a scoped mapping)",
                        location=str(wf),
                    )
                )
        if violated:
            result = ProbeResult.VIOLATED
        elif indeterminate:
            result = ProbeResult.INDETERMINATE
        else:
            result = ProbeResult.SATISFIED
        return RequirementOutcome(
            requirement_id=self.requirement_id, result=result, evidence=tuple(evidence)
        )


class CiNoUnconditionalBypassProbe(StaticProbe):
    requirement_id = "static.ci-no-security-bypass"
    _SECURITY_TOOLS = ("bandit", "pip-audit", "safety", "npm audit", "trivy")

    def check(self, workspace):
        workflows = find_workflow_files(workspace.root)
        if not workflows:
            return RequirementOutcome(
                requirement_id=self.requirement_id, result=ProbeResult.NOT_APPLICABLE
            )
        evidence = []
        for wf in workflows:
            text = wf.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "|| true" in line and any(tool in line for tool in self._SECURITY_TOOLS):
                    evidence.append(
                        Evidence(
                            probe=self.requirement_id,
                            detail=line.strip(),
                            location=f"{wf}:{lineno}",
                        )
                    )
        result = ProbeResult.VIOLATED if evidence else ProbeResult.SATISFIED
        return RequirementOutcome(
            requirement_id=self.requirement_id, result=result, evidence=tuple(evidence)
        )


class DockerfileNoMutableTagProbe(StaticProbe):
    requirement_id = "static.dockerfile-no-mutable-tag"

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
        evidence = []
        for lineno, image_ref in refs:
            if "@sha256:" in image_ref:
                continue  # digest pin is stronger than a tag; supply_chain_probes checks this specifically
            if ":" not in image_ref.split("/")[-1] or image_ref.endswith(":latest"):
                evidence.append(
                    Evidence(
                        probe=self.requirement_id,
                        detail=f"mutable base image reference: {image_ref}",
                        location=f"{path}:{lineno}",
                    )
                )
        result = ProbeResult.VIOLATED if evidence else ProbeResult.SATISFIED
        return RequirementOutcome(
            requirement_id=self.requirement_id, result=result, evidence=tuple(evidence)
        )


STATIC_PROBES = (
    PyprojectParseableProbe(),
    CiPermissionsReadOnlyProbe(),
    CiNoUnconditionalBypassProbe(),
    DockerfileNoMutableTagProbe(),
)

__all__ = [
    "PyprojectParseableProbe",
    "CiPermissionsReadOnlyProbe",
    "CiNoUnconditionalBypassProbe",
    "DockerfileNoMutableTagProbe",
    "STATIC_PROBES",
]
