"""
ProjectContext — Accumulated cross-phase knowledge for generative context
=========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Accumulates architectural decisions and learnings across pipeline phases.
Populated by the decomposer and architecture advisor, consumed by
the generator and evaluator. Persisted across task boundaries.

Design:
  - Immutable-like dataclass — all fields are plain values, no shared state
  - to_system_prompt() renders the accumulated context as system prompt text
  - Must stay within token limits when truncated
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectContext:
    """Accumulates architectural decisions and learnings across pipeline phases.

    Populated by the decomposer and architecture advisor, consumed by
    the generator and evaluator. Persisted across task boundaries.

    Attributes:
        project_type: Detected project type (e.g. "backend", "frontend", "cli")
        tech_stack: Technologies chosen for this project
        architecture_style: Structural pattern (layered, hexagonal, mvc, etc.)
        architecture_decisions: List of ArchitectureDecision-like dicts
        key_constraints: Important constraints to respect across all tasks
        phase_learnings: Map of phase_name → insight string
    """

    project_type: str = ""
    tech_stack: list[str] = field(default_factory=list)
    architecture_style: str = ""
    architecture_decisions: list[dict] = field(default_factory=list)
    key_constraints: list[str] = field(default_factory=list)
    phase_learnings: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return True if no context has been accumulated yet."""
        return not self.project_type and not self.tech_stack

    def to_system_prompt(self) -> str:
        """Render accumulated context as a system prompt block for the generator.

        Returns:
            Formatted string suitable for injection into a system prompt,
            or empty string if no context has been accumulated.
        """
        if self.is_empty():
            return ""

        lines: list[str] = [
            "## PROJECT CONTEXT (accumulated across phases)",
            "",
        ]

        if self.project_type:
            lines.append(f"Project type: {self.project_type}")
        if self.tech_stack:
            lines.append(f"Tech stack: {', '.join(self.tech_stack)}")
        if self.architecture_style:
            lines.append(f"Architecture: {self.architecture_style}")

        if self.architecture_decisions:
            lines.append("")
            lines.append("Architecture decisions:")
            for i, dec in enumerate(self.architecture_decisions[:5], 1):
                app_type = dec.get("app_type", "")
                rationale = dec.get("rationale", "")
                if app_type:
                    lines.append(f"  {i}. App type: {app_type}")
                if rationale:
                    lines.append(f"     Rationale: {rationale[:200]}")

        if self.key_constraints:
            lines.append("")
            lines.append("Key constraints:")
            for c in self.key_constraints:
                lines.append(f"  - {c}")

        if self.phase_learnings:
            lines.append("")
            lines.append("Phase learnings:")
            for phase, insight in self.phase_learnings.items():
                lines.append(f"  - [{phase}] {insight[:150]}")

        return "\n".join(lines)
