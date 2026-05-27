"""
ContextSystem — Workspace + project knowledge injection.
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 3, Phase 7 (Lovable-inspired): Workspace and project-level
.md files are always injected into the LLM context. Supports:
- Workspace knowledge (shared across projects)
- Project knowledge (per-project overrides)
- Skill playbooks (triggered by /skill-name invocation)

Phase 8 (Skills System): Named playbooks with trigger descriptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default workspace knowledge directory
DEFAULT_WORKSPACE_KNOWLEDGE = Path.home() / ".orchestrator" / "knowledge"
DEFAULT_PROJECT_KNOWLEDGE = Path(".orchestrator") / "knowledge"


@dataclass
class KnowledgeFile:
    """A single knowledge file with metadata."""

    path: str
    content: str
    category: str = ""  # e.g., "architecture", "conventions", "security"
    priority: int = 0  # Higher = injected first

    @property
    def name(self) -> str:
        return Path(self.path).stem

    def to_context(self) -> str:
        """Format for injection into LLM context."""
        header = f"## Knowledge: {self.name}"
        if self.category:
            header += f" [{self.category}]"
        return f"{header}\n\n{self.content}\n"


@dataclass
class Skill:
    """A named playbook with trigger description.

    Skills are Markdown files in a skills directory. The filename
    (without extension) is the skill name. The first heading is the
    trigger description shown to agents.
    """

    name: str
    description: str
    content: str
    path: str = ""
    triggers: list[str] = field(default_factory=list)  # Keywords that activate

    @classmethod
    def from_file(cls, path: Path) -> Skill | None:
        """Load a skill from a Markdown file.

        File format:
            # Skill Name — Brief description
            ## Triggers
            - keyword1
            - keyword2
            ## Instructions
            ... skill content ...
        """
        if not path.exists() or path.suffix != ".md":
            return None

        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # First heading is name + description
        name = path.stem.replace("_", " ").replace("-", " ").title()
        description = ""
        triggers: list[str] = []

        if lines and lines[0].startswith("# "):
            heading = lines[0][2:]
            if "—" in heading or " - " in heading:
                parts = heading.split("—" if "—" in heading else " - ", 1)
                name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""

        # Look for triggers section
        in_triggers = False
        for line in lines:
            if line.startswith("## Triggers") or line.startswith("## triggers"):
                in_triggers = True
                continue
            if in_triggers and line.startswith("- "):
                triggers.append(line[2:].strip())
            elif in_triggers and line.startswith("##"):
                in_triggers = False

        return cls(
            name=name,
            description=description,
            content=content,
            path=str(path),
            triggers=triggers,
        )

    def to_context(self) -> str:
        """Format for injection into LLM context."""
        desc = f" — {self.description}" if self.description else ""
        return f"""## Skill: {self.name}{desc}

{self.content}
"""


class WorkspaceKnowledge:
    """Manages workspace-level knowledge files.

    Workspace knowledge is shared across all projects. Stored in
    ~/.orchestrator/knowledge/ by default.
    """

    def __init__(self, workspace_dir: str | None = None):
        self._dir = Path(workspace_dir or DEFAULT_WORKSPACE_KNOWLEDGE)
        self._dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, content: str, category: str = "") -> KnowledgeFile:
        """Add or update a knowledge file."""
        safe_name = name.lower().replace(" ", "_") + ".md"
        path = self._dir / safe_name
        path.write_text(content, encoding="utf-8")
        return KnowledgeFile(path=str(path), content=content, category=category)

    def get(self, name: str) -> KnowledgeFile | None:
        """Get a knowledge file by name."""
        safe_name = name.lower().replace(" ", "_") + ".md"
        path = self._dir / safe_name
        if path.exists():
            return KnowledgeFile(
                path=str(path),
                content=path.read_text(encoding="utf-8"),
            )
        return None

    def list_all(self) -> list[KnowledgeFile]:
        """List all knowledge files."""
        files = []
        for path in sorted(self._dir.glob("*.md")):
            files.append(
                KnowledgeFile(
                    path=str(path),
                    content=path.read_text(encoding="utf-8"),
                    category=path.stem,
                )
            )
        return files

    def build_context(self) -> str:
        """Build a combined context string from all knowledge files.

        This is the string to inject into every LLM prompt.
        """
        parts = ["## Workspace Knowledge", ""]
        for kf in self.list_all():
            parts.append(kf.to_context())
        if len(parts) == 2:
            return ""
        return "\n".join(parts)


class ProjectKnowledge:
    """Manages project-level knowledge files.

    Project knowledge is stored in .orchestrator/knowledge/ within the
    project directory and overrides workspace-level knowledge.
    """

    def __init__(self, project_dir: str = "."):
        self._dir = Path(project_dir) / DEFAULT_PROJECT_KNOWLEDGE
        self._dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, content: str, category: str = "") -> KnowledgeFile:
        safe_name = name.lower().replace(" ", "_") + ".md"
        path = self._dir / safe_name
        path.write_text(content, encoding="utf-8")
        return KnowledgeFile(path=str(path), content=content, category=category)

    def list_all(self) -> list[KnowledgeFile]:
        files = []
        for path in sorted(self._dir.glob("*.md")):
            files.append(
                KnowledgeFile(
                    path=str(path),
                    content=path.read_text(encoding="utf-8"),
                    category=path.stem,
                )
            )
        return files

    def build_context(self) -> str:
        parts = ["## Project Knowledge", ""]
        for kf in self.list_all():
            parts.append(kf.to_context())
        if len(parts) == 2:
            return ""
        return "\n".join(parts)


class SkillRegistry:
    """Registry of named skills with trigger-based invocation.

    Skills are loaded from ~/.orchestrator/skills/ (workspace) and
    .orchestrator/skills/ (project). They are invoked via /skill-name
    commands or auto-triggered by keyword matching.
    """

    def __init__(self, workspace_dir: str | None = None, project_dir: str = "."):
        self._workspace_skills = Path(workspace_dir or Path.home() / ".orchestrator" / "skills")
        self._project_skills = Path(project_dir) / ".orchestrator" / "skills"
        self._workspace_skills.mkdir(parents=True, exist_ok=True)
        self._project_skills.mkdir(parents=True, exist_ok=True)

    def discover(self) -> list[Skill]:
        """Discover all available skills from workspace and project dirs."""
        skills: dict[str, Skill] = {}

        # Load workspace skills first
        for path in sorted(self._workspace_skills.glob("**/*.md")):
            skill = Skill.from_file(path)
            if skill:
                skills[skill.name.lower()] = skill

        # Project skills override workspace
        for path in sorted(self._project_skills.glob("**/*.md")):
            skill = Skill.from_file(path)
            if skill:
                skills[skill.name.lower()] = skill

        return list(skills.values())

    def find(self, name: str) -> Skill | None:
        """Find a skill by name."""
        all_skills = self.discover()
        for s in all_skills:
            if s.name.lower() == name.lower():
                return s
        return None

    def match_triggers(self, text: str) -> list[Skill]:
        """Find skills whose triggers match the given text."""
        all_skills = self.discover()
        text_lower = text.lower()
        matched = []
        for s in all_skills:
            for trigger in s.triggers:
                if trigger.lower() in text_lower:
                    matched.append(s)
                    break
        return matched

    def build_context(self, skill_names: list[str] | None = None) -> str:
        """Build a combined context from specified skills (or all).

        Args:
            skill_names: Specific skill names to include, or None for all

        Returns:
            Combined skill context string
        """
        skills = self.discover()
        if skill_names:
            skills = [s for s in skills if s.name.lower() in {n.lower() for n in skill_names}]

        if not skills:
            return ""

        parts = ["## Available Skills", ""]
        for s in skills:
            parts.append(f"- `/{s.name}`: {s.description}")
        parts.append("")
        parts.append("Invoke a skill with `/skill-name` to load its full instructions.")
        return "\n".join(parts)


class ContextBuilder:
    """Builds the complete LLM context from workspace knowledge, project
    knowledge, and skills.

    Usage:
        builder = ContextBuilder()
        context = builder.build(inject_skills=["python-testing"])
        # Use context as system message prefix
    """

    def __init__(self, project_dir: str = "."):
        self.workspace = WorkspaceKnowledge()
        self.project = ProjectKnowledge(project_dir)
        self.skills = SkillRegistry(project_dir=project_dir)

    def build(
        self,
        inject_skills: list[str] | None = None,
        include_workspace: bool = True,
        include_project: bool = True,
        include_skills: bool = True,
    ) -> str:
        """Build the combined context for injection into LLM prompts."""
        parts = []

        if include_workspace:
            ctx = self.workspace.build_context()
            if ctx:
                parts.append(ctx)

        if include_project:
            ctx = self.project.build_context()
            if ctx:
                parts.append(ctx)

        if include_skills:
            if inject_skills:
                # Load full skill content for requested skills
                for name in inject_skills:
                    skill = self.skills.find(name)
                    if skill:
                        parts.append(skill.to_context())
            else:
                # Just list available skills
                ctx = self.skills.build_context()
                if ctx:
                    parts.append(ctx)

        return "\n\n".join(parts)
