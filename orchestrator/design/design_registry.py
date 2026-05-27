"""
DesignSystemRegistry - shadcn/ui format, registry.json, CSS tokens.
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 4, Phase V1 (v0-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegistryItem:
    name: str
    type: str  # "component", "style", "token", "template"
    description: str = ""
    files: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    version: str = "0.1.0"
    author: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "files": self.files,
            "dependencies": self.dependencies,
            "version": self.version,
        }


class DesignRegistry:
    """Manages a shadcn/ui-compatible component registry."""

    def __init__(self, registry_dir=".registry"):
        self._dir = Path(registry_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, RegistryItem] = {}
        self._load()

    def _load(self):
        fp = self._dir / "registry.json"
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                self._items = {i["name"]: RegistryItem(**i) for i in data.get("items", [])}
            except Exception:
                pass

    def _save(self):
        (self._dir / "registry.json").write_text(
            json.dumps(
                {
                    "$schema": "https://ui.shadcn.com/schema/registry.json",
                    "items": [i.to_dict() for i in self._items.values()],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def register(self, item):
        self._items[item.name] = item
        self._save()

    def seed_defaults(self):
        """Seed registry with shadcn/ui-compatible base components."""
        defaults = [
            RegistryItem(
                "button",
                "component",
                "Base button with variants",
                files=["button.tsx"],
                dependencies=["utils"],
            ),
            RegistryItem(
                "card", "component", "Card with header/content/footer", files=["card.tsx"]
            ),
            RegistryItem(
                "input",
                "component",
                "Form input with validation",
                files=["input.tsx"],
                dependencies=["utils"],
            ),
            RegistryItem(
                "dialog",
                "component",
                "Modal dialog with overlay",
                files=["dialog.tsx"],
                dependencies=["button"],
            ),
            RegistryItem(
                "dropdown-menu",
                "component",
                "Dropdown menu with items",
                files=["dropdown-menu.tsx"],
            ),
            RegistryItem(
                "toast",
                "component",
                "Toast notification system",
                files=["toast.tsx", "use-toast.ts"],
            ),
            RegistryItem("tabs", "component", "Tabbed content container", files=["tabs.tsx"]),
            RegistryItem("avatar", "component", "User avatar with fallback", files=["avatar.tsx"]),
            RegistryItem("badge", "component", "Status badge with variants", files=["badge.tsx"]),
            RegistryItem("separator", "component", "Visual separator", files=["separator.tsx"]),
            RegistryItem("utils", "utility", "cn() class merge utility", files=["utils.ts"]),
            RegistryItem(
                "colors", "token", "Design system color tokens", files=["colors.ts", "globals.css"]
            ),
            RegistryItem(
                "typography", "token", "Typography scale and fonts", files=["typography.ts"]
            ),
        ]
        for item in defaults:
            self.register(item)
        return len(defaults)

    def add_component(self, name, description="", files=None, dependencies=None):
        item = RegistryItem(
            name=name,
            type="component",
            description=description,
            files=files or [f"{name}.tsx"],
            dependencies=dependencies or [],
        )
        self.register(item)
        return item

    def get(self, name):
        return self._items.get(name)

    def resolve_dependencies(self, name, resolved=None):
        if resolved is None:
            resolved = []
        item = self._items.get(name)
        if not item:
            return resolved
        for dep in item.dependencies:
            if dep not in resolved:
                self.resolve_dependencies(dep, resolved)
        if name not in resolved:
            resolved.append(name)
        return resolved

    def list_all(self):
        return [
            {
                "name": i.name,
                "type": i.type,
                "description": i.description,
                "files": i.files,
                "dependencies": i.dependencies,
            }
            for i in self._items.values()
        ]

    def export_component(self, name, output_dir):
        """Export a component and its dependencies to a directory."""
        deps = self.resolve_dependencies(name)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for dep_name in deps:
            item = self._items.get(dep_name)
            if item:
                for f in item.files:
                    (out / f).write_text(
                        f"// Component: {dep_name}\n// Generated from design registry\n",
                        encoding="utf-8",
                    )
        return deps
