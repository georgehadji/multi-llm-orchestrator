"""
TeamTemplates - Shareable project configs for teams.
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 9, Phase W5 (Bolt.new-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json, logging, time

logger = logging.getLogger(__name__)


@dataclass
class TeamTemplate:
    name: str
    description: str = ""
    version: str = "1.0.0"
    entities: list = field(default_factory=list)
    modules: list = field(default_factory=list)
    integrations: list = field(default_factory=list)
    auth_config: dict = field(default_factory=dict)
    design_system: dict = field(default_factory=dict)
    created_by: str = ""
    created_at: float = 0.0

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "entities": self.entities,
            "modules": self.modules,
            "integrations": self.integrations,
            "auth": self.auth_config,
            "design": self.design_system,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            version=d.get("version", "1.0.0"),
            entities=d.get("entities", []),
            modules=d.get("modules", []),
            integrations=d.get("integrations", []),
            auth_config=d.get("auth", {}),
            design_system=d.get("design", {}),
            created_by=d.get("created_by", ""),
            created_at=d.get("created_at", 0.0),
        )


class TeamTemplateManager:
    def __init__(self, templates_dir=None):
        self._dir = Path(templates_dir or Path.home() / ".orchestrator" / "templates")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._templates: dict[str, TeamTemplate] = {}
        self._load()

    def _load(self):
        for f in self._dir.glob("*.json"):
            try:
                t = TeamTemplate.from_dict(json.loads(f.read_text(encoding="utf-8")))
                self._templates[t.name] = t
            except Exception:
                pass

    def save(self, template):
        self._templates[template.name] = template
        (self._dir / f"{template.name}.json").write_text(
            json.dumps(template.to_dict(), indent=2), encoding="utf-8"
        )

    def create(
        self,
        name,
        description="",
        entities=None,
        modules=None,
        integrations=None,
        auth_config=None,
        design_system=None,
    ):
        t = TeamTemplate(
            name=name,
            description=description,
            created_at=time.time(),
            entities=entities or [],
            modules=modules or [],
            integrations=integrations or [],
            auth_config=auth_config or {},
            design_system=design_system or {},
        )
        self.save(t)
        return t

    def export(self, name, output_dir):
        t = self._templates.get(name)
        if not t:
            return None
        out = Path(output_dir) / f"{name}.template.json"
        out.write_text(json.dumps(t.to_dict(), indent=2), encoding="utf-8")
        return str(out)

    def list_templates(self):
        return [
            {
                "name": t.name,
                "description": t.description,
                "version": t.version,
                "entities": len(t.entities),
                "modules": len(t.modules),
            }
            for t in self._templates.values()
        ]

    def generate_project_config(self, template_name, output_dir):
        t = self._templates.get(template_name)
        if not t:
            return None
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        config = {
            "app_name": t.name,
            "description": t.description,
            "entities": t.entities,
            "auth": t.auth_config,
        }
        (out / "orchestrator_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        if t.design_system:
            (out / ".design-system.json").write_text(
                json.dumps(t.design_system, indent=2), encoding="utf-8"
            )
        return str(out)
