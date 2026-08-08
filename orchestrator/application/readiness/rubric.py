"""Rubric — the Composite registry of readiness Requirements (Phase 7, P-1).

Requirements are data (domain/readiness.py); Rubric is where they get
collected, filtered by archetype, and rolled up into a level. Registration
mirrors the RefinementService.register_operator idiom already used in
Phase 6 — an explicit call in the composition root, never a module-level
singleton (plan §3.5.3 non-goals).
"""

from __future__ import annotations

from ...domain.readiness import (
    AppArchetype,
    ReadinessLevel,
    Requirement,
    RequirementOutcome,
    achieved_level,
)


class Rubric:
    """Holds the full requirement set and answers archetype-scoped questions."""

    def __init__(self) -> None:
        self._requirements: list[Requirement] = []

    def register(self, requirement: Requirement) -> None:
        self._requirements.append(requirement)

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        return tuple(self._requirements)

    def applicable(self, archetype: AppArchetype) -> tuple[Requirement, ...]:
        return tuple(r for r in self._requirements if archetype in r.archetypes)

    def achieved_level(
        self, outcomes: tuple[RequirementOutcome, ...], archetype: AppArchetype
    ) -> ReadinessLevel:
        return achieved_level(outcomes, self.requirements, archetype)

    def explain(self) -> str:
        """Render the full rubric as an indented tree, grouped by level.

        Supports the "readiness --explain" acceptance criterion (P-1):
        the requirement set must be readable as one tree, not traced
        across seven generators.
        """
        lines: list[str] = []
        for level in sorted(ReadinessLevel, key=int):
            at_level = [r for r in self._requirements if r.level == level]
            if not at_level:
                continue
            lines.append(f"{level.name} (L{int(level)})")
            for req in sorted(at_level, key=lambda r: r.id):
                kind = "blocking" if req.blocking else "advisory"
                archetypes = ", ".join(sorted(a.value for a in req.archetypes))
                lines.append(f"  - [{kind}] {req.id}: {req.title} ({archetypes})")
        return "\n".join(lines)


__all__ = ["Rubric"]
