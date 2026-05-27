"""
SkillsExporter - SKILL.md files for Claude, Cursor, Copilot.
==============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 9, Phase B7 (Base44-inspired).
Exports orchestrator skills as SKILL.md files compatible with
external AI agents (Claude, Cursor, Copilot).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

SKILL_TEMPLATE = """# {name} — {description}

{content}

## Triggers
{triggers}

## Usage
```
{usage}
```
"""


@dataclass
class SkillExport:
    name: str
    description: str = ""
    content: str = ""
    triggers: list = field(default_factory=list)
    usage: str = ""
    target_agents: list = field(default_factory=list)  # ["claude", "cursor", "copilot"]

    def to_skill_md(self):
        return SKILL_TEMPLATE.format(
            name=self.name,
            description=self.description or "Orchestrator skill",
            content=self.content,
            triggers="\n".join(f"- {t}" for t in self.triggers or ["auto"]),
            usage=self.usage or "/skill-name or automatic trigger",
        )


class SkillsExporter:
    """Exports orchestrator skills for external AI agents."""

    def __init__(self, output_dir=".agents/skills"):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def export(self, skill, target_agents=None):
        """Export a skill as SKILL.md for specific agent targets.

        Writes to: .agents/skills/{name}/SKILL.md
        Compatible with the Agent Skills open standard.
        """
        se = SkillExport(
            name=skill.name if hasattr(skill, "name") else str(skill),
            description=getattr(skill, "description", ""),
            content=getattr(skill, "content", ""),
            triggers=getattr(skill, "triggers", []),
            target_agents=target_agents or ["claude", "cursor", "copilot"],
        )
        out = self._dir / se.name / "SKILL.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(se.to_skill_md(), encoding="utf-8")
        return str(out)

    def export_all(self, skills):
        """Export multiple skills at once."""
        return [self.export(s) for s in skills]

    def generate_default_skills(self):
        """Generate default orchestrator skills for external agents."""
        defaults = [
            (
                "Code Generation",
                "Generate production-quality code with the AI Orchestrator",
                "Use type hints, follow project conventions, write tests alongside code.",
                ["code", "implement", "build", "generate", "create"],
            ),
            (
                "Code Review",
                "Review generated or existing code for quality and security",
                "Check for bugs, security vulnerabilities, style violations, and suggest improvements.",
                ["review", "audit", "check", "analyze"],
            ),
            (
                "Test Generation",
                "Generate comprehensive test suites with pytest",
                "Write unit tests, integration tests, and edge case coverage. Target 80%+ coverage.",
                ["test", "pytest", "coverage", "unit test"],
            ),
            (
                "Decomposition",
                "Break down large tasks into atomic, executable subtasks",
                "Analyze requirements, identify dependencies, create targeted subtasks with clear success criteria.",
                ["plan", "design", "architecture", "decompose", "break down"],
            ),
            (
                "Documentation",
                "Generate README, API docs, and architecture documentation",
                "Create comprehensive Markdown documentation with examples and diagrams.",
                ["docs", "documentation", "readme", "explain"],
            ),
        ]
        exported = []
        for name, desc, content, triggers in defaults:
            skill = type(
                "Skill",
                (),
                {"name": name, "description": desc, "content": content, "triggers": triggers},
            )()
            path = self.export(skill)
            exported.append(path)
        return exported
