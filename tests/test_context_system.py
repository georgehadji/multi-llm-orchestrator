"""
Tests for context_system.py — Workspace knowledge, project knowledge, skills.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


from orchestrator.context_system import (
    WorkspaceKnowledge,
    ProjectKnowledge,
    SkillRegistry,
    Skill,
    ContextBuilder,
)


class TestWorkspaceKnowledge:
    """Tests for workspace-level knowledge management."""

    def test_add_and_get(self):
        """Knowledge must survive add/get cycle."""
        wk = WorkspaceKnowledge(str(Path(tempfile.mkdtemp())))
        kf = wk.add("conventions", "# Coding Style\nUse type hints.", "architecture")
        assert kf.name == "conventions"
        retrieved = wk.get("conventions")
        assert retrieved is not None
        assert "type hints" in retrieved.content

    def test_build_context(self):
        """build_context must include knowledge file content."""
        wk = WorkspaceKnowledge(str(Path(tempfile.mkdtemp())))
        wk.add("security", "# Security\nNo hardcoded keys.")
        ctx = wk.build_context()
        assert "Security" in ctx
        assert "hardcoded" in ctx

    def test_empty_workspace(self):
        """Empty workspace must return empty context."""
        wk = WorkspaceKnowledge(str(Path(tempfile.mkdtemp())))
        assert wk.build_context() == ""

    def test_list_all(self):
        """list_all must return all knowledge files."""
        wk = WorkspaceKnowledge(str(Path(tempfile.mkdtemp())))
        wk.add("a", "content A")
        wk.add("b", "content B")
        files = wk.list_all()
        assert len(files) == 2


class TestProjectKnowledge:
    """Tests for project-level knowledge."""

    def test_add(self):
        """Project knowledge must persist."""
        pk = ProjectKnowledge(str(Path(tempfile.mkdtemp())))
        pk.add("architecture", "# Architecture\nLayered design.")
        files = pk.list_all()
        assert len(files) == 1
        assert files[0].name == "architecture"

    def test_build_context(self):
        """Project context must include file content."""
        pk = ProjectKnowledge(str(Path(tempfile.mkdtemp())))
        pk.add("api", "# API Design\nRESTful.")
        ctx = pk.build_context()
        assert "API Design" in ctx


class TestSkillRegistry:
    """Tests for skill management."""

    def test_skill_from_file(self):
        """Skill must parse from valid Markdown file."""
        content = (
            "# Python Testing - Write unit tests\n"
            "## Triggers\n- pytest\n- test\n"
            "## Instructions\nUse pytest fixtures."
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            fpath = f.name
        try:
            skill = Skill.from_file(Path(fpath))
            assert skill is not None
            assert "Python" in skill.name or "Testing" in skill.name
            assert "pytest" in skill.triggers
            assert "fixtures" in skill.content
        finally:
            Path(fpath).unlink()

    def test_match_triggers(self):
        """Skill must activate on trigger keyword match."""
        parent = Path(tempfile.mkdtemp())
        skills_dir = parent / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "python.md").write_text(
            "# Python Testing\n## Triggers\n- pytest\n- unittest\n## Instructions\nTest code.",
            encoding="utf-8",
        )
        registry = SkillRegistry(workspace_dir=str(parent))
        matched = registry.match_triggers("I need pytest for this")
        assert len(matched) > 0
        assert "Python" in matched[0].name

    def test_no_match(self):
        """Skill must not activate without trigger match."""
        registry = SkillRegistry()
        matched = registry.match_triggers("nothing relevant")
        assert matched == []

    def test_find_skill(self):
        """find must locate skill by name."""
        parent = Path(tempfile.mkdtemp())
        skills_dir = parent / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "security.md").write_text(
            "# Security Review\n## Triggers\n- security\n## Instructions\nAudit code.",
            encoding="utf-8",
        )
        registry = SkillRegistry(workspace_dir=str(parent))
        found = registry.find("Security")
        assert found is not None
        assert found is not None
        assert "Security" in found.name


class TestContextBuilder:
    """Tests for combined context builder."""

    def test_build_combined(self):
        """ContextBuilder must combine workspace + project + skills."""
        d = str(Path(tempfile.mkdtemp()))
        builder = ContextBuilder(project_dir=d)
        result = builder.build()
        assert isinstance(result, str)

    def test_build_with_skills_disabled(self):
        """ContextBuilder must respect include flags."""
        d = str(Path(tempfile.mkdtemp()))
        builder = ContextBuilder(project_dir=d)
        result = builder.build(include_skills=False)
        assert isinstance(result, str)
