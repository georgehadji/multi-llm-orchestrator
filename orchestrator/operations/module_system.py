"""
ModuleSystem - Reusable component+query packages.
===================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 5, Phase R1 (Retool-inspired): Reusable packages with
declared inputs/outputs, versioning, and dependency resolution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import json
import logging
import hashlib

logger = logging.getLogger(__name__)


class ModuleKind(str, Enum):
    COMPONENT = "component"  # UI component
    QUERY = "query"  # Data query (SQL/API)
    WORKFLOW = "workflow"  # Multi-step process
    UTILITY = "utility"  # Helper function
    INTEGRATION = "integration"  # Third-party connector


@dataclass
class ModuleInput:
    name: str
    type: str = "string"
    required: bool = True
    default: Any = None
    description: str = ""

    def to_dict(self):
        d = {"name": self.name, "type": self.type}
        if self.required:
            d["required"] = True
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class ModuleOutput:
    name: str
    type: str = "any"
    description: str = ""

    def to_dict(self):
        return {"name": self.name, "type": self.type}


@dataclass
class ModuleDefinition:
    """A reusable module package with declared inputs, outputs, and dependencies."""

    name: str
    kind: ModuleKind = ModuleKind.UTILITY
    version: str = "0.1.0"
    description: str = ""
    inputs: list[ModuleInput] = field(default_factory=list)
    outputs: list[ModuleOutput] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # Module names this depends on
    code: str = ""  # Source code or query text
    tags: list[str] = field(default_factory=list)
    author: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind.value,
            "version": self.version,
            "description": self.description,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "dependencies": self.dependencies,
            "code": self.code,
            "tags": self.tags,
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            kind=ModuleKind(d.get("kind", "utility")),
            version=d.get("version", "0.1.0"),
            description=d.get("description", ""),
            inputs=[ModuleInput(**i) for i in d.get("inputs", [])],
            outputs=[ModuleOutput(**o) for o in d.get("outputs", [])],
            dependencies=d.get("dependencies", []),
            code=d.get("code", ""),
            tags=d.get("tags", []),
            author=d.get("author", ""),
        )

    def validate_inputs(self, provided):
        """Validate that all required inputs are provided."""
        missing = []
        for inp in self.inputs:
            if inp.required and inp.name not in provided:
                missing.append(inp.name)
        return missing


class ModuleRegistry:
    """Central registry for reusable modules."""

    def __init__(self, storage_dir=None):
        self._dir = Path(storage_dir or Path.home() / ".orchestrator" / "modules")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._modules: dict[str, ModuleDefinition] = {}
        self._load()

    def _load(self):
        for f in self._dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                module = ModuleDefinition.from_dict(data)
                self._modules[module.name] = module
            except Exception:
                pass

    def register(self, module):
        self._modules[module.name] = module
        fp = self._dir / f"{module.name}.json"
        fp.write_text(json.dumps(module.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Module '{module.name}' registered (v{module.version})")

    def get(self, name):
        return self._modules.get(name)

    def find(self, kind=None, tag=None):
        results = []
        for m in self._modules.values():
            if kind and m.kind != kind:
                continue
            if tag and tag not in m.tags:
                continue
            results.append(m)
        return results

    def resolve_dependencies(self, module_name, resolved=None, visited=None):
        """Resolve module dependency tree with cycle detection."""
        if resolved is None:
            resolved = []
        if visited is None:
            visited = set()

        if module_name in visited:
            raise ValueError(f"Circular dependency detected: {module_name}")

        visited.add(module_name)
        module = self.get(module_name)
        if not module:
            raise ValueError(f"Module not found: {module_name}")

        for dep in module.dependencies:
            if dep not in resolved:
                self.resolve_dependencies(dep, resolved, visited)

        if module_name not in resolved:
            resolved.append(module_name)
        return resolved

    def list_all(self):
        return list(self._modules.values())
