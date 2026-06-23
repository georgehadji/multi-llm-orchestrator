"""
KnowledgeSidebar - Context injection data provider for sidebar.
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 7, Phase U5 (UI): Knowledge/Skills sidebar backend.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SidebarSection:
    id: str
    title: str
    content: str = ""
    items: list = field(default_factory=list)
    expanded: bool = True

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "items": self.items,
            "expanded": self.expanded,
        }


class KnowledgeSidebarData:
    """Provides data for the knowledge/skills sidebar panel."""

    def __init__(self, project_dir="."):
        self._dir = Path(project_dir)
        self._workspace = Path.home() / ".orchestrator"

    def get_sections(self) -> list[dict]:
        """Get all sidebar sections for the knowledge panel."""
        sections = [
            self._get_knowledge_section(),
            self._get_skills_section(),
            self._get_references_section(),
            self._get_commands_section(),
            self._get_context_section(),
        ]
        return [s.to_dict() for s in sections]

    def _get_knowledge_section(self) -> SidebarSection:
        items = []
        kdir = self._workspace / "knowledge"
        if kdir.exists():
            for f in sorted(kdir.glob("*.md")):
                items.append(
                    {
                        "type": "knowledge",
                        "id": f.stem,
                        "label": f.stem.replace("_", " ").title(),
                        "path": str(f),
                    }
                )
        pdir = self._dir / ".orchestrator" / "knowledge"
        if pdir.exists():
            for f in sorted(pdir.glob("*.md")):
                items.append(
                    {
                        "type": "knowledge",
                        "id": f"project/{f.stem}",
                        "label": f"[Project] {f.stem.replace('_', ' ').title()}",
                        "path": str(f),
                    }
                )
        return SidebarSection(id="knowledge", title="Knowledge", items=items)

    def _get_skills_section(self) -> SidebarSection:
        items = []
        sdir = self._workspace / "skills"
        if sdir.exists():
            for f in sorted(sdir.glob("**/*.md")):
                items.append({"type": "skill", "id": f.stem, "label": f"/{f.stem}", "path": str(f)})
        return SidebarSection(id="skills", title="Skills", items=items)

    def _get_references_section(self) -> SidebarSection:
        items = [
            {
                "type": "reference",
                "id": "current_project",
                "label": "@Current Project",
                "description": "Reference files from this project",
            },
            {
                "type": "reference",
                "id": "outputs",
                "label": "@Outputs",
                "description": "Reference previous outputs",
            },
        ]
        return SidebarSection(id="references", title="@References", items=items)

    def _get_commands_section(self) -> SidebarSection:
        items = [
            {
                "type": "command",
                "id": "slash_help",
                "label": "/help",
                "description": "List all commands",
            },
            {
                "type": "command",
                "id": "slash_skills",
                "label": "/skills",
                "description": "List available skills",
            },
            {
                "type": "command",
                "id": "slash_knowledge",
                "label": "/knowledge",
                "description": "Show knowledge base",
            },
            {
                "type": "command",
                "id": "slash_status",
                "label": "/status",
                "description": "Project status",
            },
            {
                "type": "command",
                "id": "slash_diagnostics",
                "label": "/diagnostics",
                "description": "Run system diagnostics",
            },
        ]
        return SidebarSection(id="commands", title="/Commands", items=items)

    def _get_context_section(self) -> SidebarSection:
        items = [
            {"type": "context", "id": "autonomy", "label": "Autonomy", "value": "standard"},
            {"type": "context", "id": "model", "label": "Model", "value": "auto"},
            {"type": "context", "id": "budget", "label": "Budget", "value": "$10.00"},
        ]
        return SidebarSection(id="context", title="Session Context", items=items)

    def load_content(self, section_id, item_id):
        """Load the full content of a knowledge/skill item."""
        for section in self.get_sections():
            if section["id"] == section_id:
                for item in section.get("items", []):
                    if item.get("id") == item_id:
                        path = Path(item.get("path", ""))
                        if path.exists():
                            return path.read_text(encoding="utf-8")
        return ""
