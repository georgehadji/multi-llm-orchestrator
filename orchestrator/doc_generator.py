"""
DocGenerator - README, user guide, API reference generation.
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 5, Phase R7 (Retool-inspired).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .infrastructure.llm_client import UnifiedClient

logger = logging.getLogger(__name__)


@dataclass
class DocSet:
    """Complete documentation set for a project."""

    readme: str = ""
    user_guide: str = ""
    api_reference: str = ""
    architecture: str = ""
    changelog: str = ""

    def save_all(self, output_dir: str) -> dict:
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, content in [
            ("README.md", self.readme),
            ("docs/USER_GUIDE.md", self.user_guide),
            ("docs/API_REFERENCE.md", self.api_reference),
            ("docs/ARCHITECTURE.md", self.architecture),
            ("CHANGELOG.md", self.changelog),
        ]:
            if content:
                fp = out / name
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
                files[name] = str(fp)
        return files


class DocGenerator:
    """Generates project documentation from task results."""

    def __init__(self, client=None):
        self._client = client

    def generate_readme_fast(self, project_name, description, tasks, tech_stack="Python"):
        task_lines = "\n".join(f"- {t}" for t in tasks[:20])
        return f"""# {project_name}

{description}

## Tech Stack
{tech_stack}

## Tasks Completed
{task_lines}

## Getting Started
```bash
pip install -e .
python -m {project_name.lower().replace('-', '_')}
```

## Development
```bash
pip install -e ".[dev]"
pytest
ruff check .
```
"""

    async def generate_readme_llm(self, project_name, description, code_summary):
        if not self._client:
            return self.generate_readme_fast(project_name, description, [], "Python")
        prompt = f"Write a comprehensive README.md for: {project_name}\n{description[:500]}\nCode summary:\n{code_summary[:2000]}\nInclude title, description, features, tech stack, getting started, and usage."
        try:
            response = await self._client.call(
                model=None,
                prompt=prompt,
                system="You are a technical writer.",
                max_tokens=1500,
                temperature=0.3,
                timeout=60,
            )
            return response.text.strip()
        except Exception:
            return self.generate_readme_fast(project_name, description, [], "Python")

    def generate_changelog(self, versions):
        lines = ["# Changelog", ""]
        for version, desc in versions:
            lines.append(f"## {version}\n- {desc}\n")
        return "\n".join(lines)

    def generate_architecture(self, entities, endpoints):
        lines = ["# Architecture", "", "## Entities", ""]
        for e in entities:
            lines.append(f"- **{e}**")
        lines.extend(["", "## API Endpoints", ""])
        for ep in endpoints:
            lines.append(f"- `{ep}`")
        return "\n".join(lines)
