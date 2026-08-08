"""Production-readiness domain model (Phase 7, P-1).

Frozen value objects plus one pure roll-up function (§3.5.5). Stdlib only
(Contract 1) — probes, rubrics-as-registries, and archetype detection are
imperative-shell concerns that live in application/ and infrastructure/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class AppArchetype(str, Enum):
    PYTHON_SERVICE = "python_service"
    PYTHON_CLI = "python_cli"
    LIBRARY = "library"
    WEB_STATIC = "web_static"
    FULLSTACK = "fullstack"


class ReadinessLevel(IntEnum):
    PROTOTYPE = 0
    DEPLOYABLE = 1
    OPERABLE = 2
    HARDENED = 3


class ProbeResult(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    NOT_APPLICABLE = "not_applicable"
    INDETERMINATE = "indeterminate"  # probe could not run — never counts as satisfied


_ROLLUP_SATISFYING = (ProbeResult.SATISFIED, ProbeResult.NOT_APPLICABLE)


@dataclass(frozen=True)
class Evidence:
    """What a probe observed. Reports cite evidence; they never assert bare verdicts."""

    probe: str
    detail: str
    location: str | None = None  # file:line, URL, or container id
    raw: str = ""  # truncated observation


@dataclass(frozen=True)
class RequirementOutcome:
    requirement_id: str
    result: ProbeResult
    evidence: tuple[Evidence, ...] = ()
    remediable: bool = False


@dataclass(frozen=True)
class Requirement:
    """Specification object: declares its own probe and optional deterministic fix.

    The probe/remediation callables themselves are wired in the application
    layer's registry (Rubric) — this dataclass stays plain data so it can be
    hashed, diffed, and enumerated without importing anything that does I/O.
    """

    id: str
    title: str
    derives_from: str  # standard or practice cited
    level: ReadinessLevel
    archetypes: frozenset[AppArchetype]
    blocking: bool = True  # False => advisory at this level


@dataclass(frozen=True)
class ReadinessReport:
    archetype: AppArchetype
    achieved_level: ReadinessLevel
    required_level: ReadinessLevel
    outcomes: tuple[RequirementOutcome, ...] = ()
    remediations_applied: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.achieved_level >= self.required_level

    @property
    def blocking_violations(self) -> tuple[RequirementOutcome, ...]:
        return tuple(o for o in self.outcomes if o.result is ProbeResult.VIOLATED)


def achieved_level(
    outcomes: tuple[RequirementOutcome, ...],
    requirements: tuple[Requirement, ...],
    archetype: AppArchetype,
) -> ReadinessLevel:
    """Roll up outcomes into the highest level achieved.

    A level is achieved only when every *blocking* requirement scoped to
    *archetype* at that level — and at every level below it — resolved to
    SATISFIED or NOT_APPLICABLE. The first level with an unsatisfied
    blocking requirement stops the roll-up; levels above it are never
    consulted, even if individually satisfied (the classic roll-up error
    this function exists to prevent).

    The roll-up never climbs past the highest level that actually has a
    requirement registered for *archetype* — an unregistered level is
    "not characterized", not "trivially satisfied". Without this cap, an
    archetype with only L0/L1 requirements would vacuously "achieve" L3
    the moment L1 passes, which is exactly the checkbox-theatre failure
    mode (R-17) the rubric exists to prevent.
    """
    by_id = {o.requirement_id: o for o in outcomes}
    applicable = [r for r in requirements if archetype in r.archetypes]
    ceiling = max((r.level for r in applicable), default=ReadinessLevel.PROTOTYPE)

    level = ReadinessLevel.PROTOTYPE
    for candidate in sorted(ReadinessLevel, key=int):
        if candidate > ceiling:
            break
        blocking_here = [r for r in applicable if r.blocking and r.level == candidate]
        for req in blocking_here:
            outcome = by_id.get(req.id)
            if outcome is None or outcome.result not in _ROLLUP_SATISFYING:
                return level
        level = candidate
    return level


__all__ = [
    "AppArchetype",
    "Evidence",
    "ProbeResult",
    "ReadinessLevel",
    "ReadinessReport",
    "Requirement",
    "RequirementOutcome",
    "achieved_level",
]
