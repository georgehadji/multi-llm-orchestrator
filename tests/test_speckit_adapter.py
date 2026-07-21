"""
Tests for Spec-Kit Mode A Ingestion Adapter
==============================================
Author: Orchestrator core

RED → GREEN → refactor discipline.

Golden fixtures are derived from ``spec-kit-main/templates/`` to
guard against format drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from orchestrator.domain.constitution import ProjectConstitution
from orchestrator.domain.ports import FileReaderPort
from orchestrator.ingest.speckit_adapter import (
    SpecArtifacts,
    SpecKitAdapter,
    parse_constitution_md,
    parse_plan_md,
    parse_spec_md,
    parse_tasks_md,
)
from orchestrator.models import Task, TaskType

# ─────────────────────────────────────────────────────────────────────────────
# Adapters for testing — satisfy FileReaderPort in-memory
# ─────────────────────────────────────────────────────────────────────────────


class _InMemoryReader(FileReaderPort):
    """Returns pre-loaded content keyed by path; FileNotFoundError for unknowns."""

    def __init__(self, files: dict[str, str]) -> None:
        # Normalize keys to match Path-based lookups in SpecKitAdapter
        self._files = {str(Path(k)): v for k, v in files.items()}

    async def read_text(self, path: str) -> str:
        normalized = str(Path(path))
        if normalized in self._files:
            return self._files[normalized]
        raise FileNotFoundError(normalized)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_tasks_md() -> str:
    """Realistic tasks.md content based on Spec-Kit templates."""
    return """\
# Tasks: User Authentication Feature

**Input**: Design documents from `/specs/001-auth/`

## Phase 1: Setup

- [ ] T001 Create project structure per implementation plan
- [ ] T002 [P] Initialize Python project with FastAPI dependencies
- [ ] T003 [P] Configure linting and formatting tools

## Phase 3: User Story 1 - Authentication (Priority: P1)

- [ ] T010 [P] [US1] Contract test for login endpoint in tests/contract/test_auth.py
- [ ] T011 [P] [US1] Integration test for login flow in tests/integration/test_auth.py
- [ ] T012 [P] [US1] Create User model in src/models/user.py
- [ ] T013 [US1] Implement AuthService in src/services/auth.py

## Dependencies

- T013 depends on T012
"""


@pytest.fixture
def sample_constitution_md() -> str:
    """Realistic constitution.md content based on Spec-Kit principles."""
    return """\
# Auth Service Constitution

## Core Principles

### I. Code Quality & Architectural Discipline
- Protect the service layer from direct DB access
- Do not modify tests/ directory structure
- Use repository pattern for data access

### II. Test-First (NON-NEGOTIABLE)
- TDD mandatory: Tests written before implementation
- Every behavioral change must include tests

### III. Minimal Dependencies
- No forbidden imports from `requests` or `subprocess`
- Avoid dependency on external packages

## Security & Cross-Platform Constraints
- ruff and mypy must pass on all changes
- Validation required before any deployment

**Version**: 1.0.0
"""


@pytest.fixture
def sample_spec_md() -> str:
    """Realistic spec.md content with acceptance criteria."""
    return """\
# User Authentication Feature

## Overview
Implement user authentication with JWT tokens.

## Acceptance Criteria
- Users can register with email and password
- Users can login and receive JWT token
- Tokens expire after 24 hours
- Invalid tokens return 401

## Definition of Done
- All acceptance criteria met
- Tests pass at 80% coverage
- API documentation generated
"""


@pytest.fixture
def sample_plan_md() -> str:
    """Realistic plan.md with tech context."""
    return """\
# Implementation Plan: Auth Feature

## Technology Stack
- Language: Python
- Framework: FastAPI
- Database: PostgreSQL via SQLAlchemy

## Architecture
- Service layer with dependency injection
- JWT token management
"""


# ─────────────────────────────────────────────────────────────────────────────
# Tests: parse_tasks_md
# ─────────────────────────────────────────────────────────────────────────────


class TestParseTasksMd:
    def test_parse_basic_tasks(self, sample_tasks_md: str) -> None:
        """RED: Should parse checklist format into dict[str, Task]."""
        tasks = parse_tasks_md(sample_tasks_md)

        assert len(tasks) == 7  # T001, T002, T003, T010, T011, T012, T013
        assert all(isinstance(t, Task) for t in tasks.values())

    def test_task_ids_are_keys(self, sample_tasks_md: str) -> None:
        """Task IDs should be the keys in the returned dict."""
        tasks = parse_tasks_md(sample_tasks_md)
        expected_ids = {"T001", "T002", "T003", "T010", "T011", "T012", "T013"}
        assert set(tasks.keys()) == expected_ids

    def test_parallel_marker(self, sample_tasks_md: str) -> None:
        """Tasks marked [P] should be parsed correctly (no parallel field on Task currently)."""
        tasks = parse_tasks_md(sample_tasks_md)
        # T002 is marked [P] — verify it's present
        assert "T002" in tasks
        assert "Create project structure" not in tasks["T002"].prompt  # non-[P] marker item

    def test_story_group(self, sample_tasks_md: str) -> None:
        """Story markers like [US1] should be captured in tech_context."""
        tasks = parse_tasks_md(sample_tasks_md)
        # T010 has [US1] → should set tech_context = "US1"
        assert tasks["T010"].tech_context == "US1"

    def test_file_paths_extracted(self, sample_tasks_md: str) -> None:
        """File paths in descriptions should be extracted into target_path."""
        tasks = parse_tasks_md(sample_tasks_md)
        assert tasks["T012"].target_path.endswith("user.py")
        assert tasks["T013"].target_path.endswith("auth.py")

    def test_default_task_type(self, sample_tasks_md: str) -> None:
        """All parsed tasks should default to CODE_GEN."""
        tasks = parse_tasks_md(sample_tasks_md)
        for t in tasks.values():
            assert t.type == TaskType.CODE_GEN

    def test_empty_text_raises(self) -> None:
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="No parseable tasks"):
            parse_tasks_md("")

    def test_no_tasks_raises(self) -> None:
        """Text with no matching task lines should raise ValueError."""
        with pytest.raises(ValueError, match="No parseable tasks"):
            parse_tasks_md("# Just a heading\n\nSome text\n")

    def test_single_task(self) -> None:
        """Single task line should parse."""
        tasks = parse_tasks_md("- [ ] T001 Implement the thing\n")
        assert len(tasks) == 1
        assert tasks["T001"].prompt == "Implement the thing"

    def test_task_with_full_path(self) -> None:
        """File path extraction should handle full paths."""
        tasks = parse_tasks_md("- [ ] T001 [US1] Create model in app/models/entity.py\n")
        assert "app/models/entity.py" in tasks["T001"].target_path


# ─────────────────────────────────────────────────────────────────────────────
# Tests: parse_constitution_md
# ─────────────────────────────────────────────────────────────────────────────


class TestParseConstitutionMd:
    def test_parse_constitution(self, sample_constitution_md: str) -> None:
        """RED: Should parse constitution.md into ProjectConstitution."""
        constitution = parse_constitution_md(sample_constitution_md)

        assert isinstance(constitution, ProjectConstitution)

    def test_protect_paths_detected(self, sample_constitution_md: str) -> None:
        """Principles mentioning 'protect' or 'do not modify' should populate protect_paths."""
        constitution = parse_constitution_md(sample_constitution_md)
        assert len(constitution.protect_paths) > 0

    def test_forbidden_imports_detected(self, sample_constitution_md: str) -> None:
        """Principles mentioning forbidden imports should populate forbidden_imports."""
        constitution = parse_constitution_md(sample_constitution_md)
        assert "requests" in constitution.forbidden_imports
        assert "subprocess" in constitution.forbidden_imports

    def test_require_tests_detected(self, sample_constitution_md: str) -> None:
        """NON-NEGOTIABLE Test-First principle should set require_tests."""
        constitution = parse_constitution_md(sample_constitution_md)
        assert constitution.require_tests is True

    def test_required_validators_detected(self, sample_constitution_md: str) -> None:
        """Mentioned validators (ruff, mypy) should populate required_validators."""
        constitution = parse_constitution_md(sample_constitution_md)
        assert "ruff" in constitution.required_validators
        assert "mypy" in constitution.required_validators

    def test_empty_text_returns_empty(self) -> None:
        """Empty constitution text should return empty ProjectConstitution."""
        constitution = parse_constitution_md("")
        assert isinstance(constitution, ProjectConstitution)
        assert constitution.protect_paths == []
        assert constitution.forbidden_imports == []
        assert constitution.require_tests is False

    def test_garbage_text_does_not_crash(self) -> None:
        """Nonsense markdown should not crash, returns empty defaults."""
        constitution = parse_constitution_md("# Random\n\nNo structure here\n")
        assert isinstance(constitution, ProjectConstitution)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: parse_spec_md
# ─────────────────────────────────────────────────────────────────────────────


class TestParseSpecMd:
    def test_extract_criteria(self, sample_spec_md: str) -> None:
        """RED: Should extract bullet points under Acceptance Criteria."""
        criteria = parse_spec_md(sample_spec_md)
        assert len(criteria) >= 4
        assert any("register" in c for c in criteria)

    def test_definition_of_done(self, sample_spec_md: str) -> None:
        """Should also extract under 'Definition of Done'."""
        criteria = parse_spec_md(sample_spec_md)
        assert any("coverage" in c for c in criteria)

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty spec.md returns empty list."""
        assert parse_spec_md("") == []

    def test_no_criteria_returns_empty(self) -> None:
        """Text without criteria sections returns empty list."""
        text = "# Just an overview\nNo criteria here.\n"
        assert parse_spec_md(text) == []


# ─────────────────────────────────────────────────────────────────────────────
# Tests: parse_plan_md
# ─────────────────────────────────────────────────────────────────────────────


class TestParsePlanMd:
    def test_detect_language(self, sample_plan_md: str) -> None:
        """RED: Should detect Python from the tech stack."""
        hints = parse_plan_md(sample_plan_md)
        assert hints.get("language") == "python"

    def test_detect_stack(self, sample_plan_md: str) -> None:
        """Should detect FastAPI from stack keywords."""
        hints = parse_plan_md(sample_plan_md)
        assert "fastapi" in hints.get("stack", [])

    def test_empty_text_returns_empty_dict(self) -> None:
        """Empty plan.md returns empty dict."""
        assert parse_plan_md("") == {}


# ─────────────────────────────────────────────────────────────────────────────
# Tests: SpecKitAdapter.load
# ─────────────────────────────────────────────────────────────────────────────


class TestSpecKitAdapter:
    async def test_load_full_directory(self, sample_tasks_md: str, sample_spec_md: str) -> None:
        """RED: Should load a full Spec-Kit directory and return SpecArtifacts."""
        reader = _InMemoryReader(
            {
                "/specs/test/tasks.md": sample_tasks_md,
                "/specs/test/spec.md": sample_spec_md,
                "/specs/test/plan.md": "# Plan\nLanguage: Python\n",
            }
        )
        adapter = SpecKitAdapter(file_reader=reader)
        artifacts = await adapter.load("/specs/test")

        assert isinstance(artifacts, SpecArtifacts)
        assert len(artifacts.tasks) == 7
        assert isinstance(artifacts.constitution, ProjectConstitution)

    async def test_missing_tasks_md_raises(self) -> None:
        """Missing tasks.md should raise FileNotFoundError."""
        reader = _InMemoryReader({})
        adapter = SpecKitAdapter(file_reader=reader)

        with pytest.raises(FileNotFoundError, match="tasks.md"):
            await adapter.load("/nonexistent")

    async def test_invalid_tasks_md_raises(self) -> None:
        """tasks.md with no parseable tasks should raise ValueError."""
        reader = _InMemoryReader(
            {
                "/specs/bad/tasks.md": "# No tasks here\n",
            }
        )
        adapter = SpecKitAdapter(file_reader=reader)

        with pytest.raises(ValueError, match="No parseable tasks"):
            await adapter.load("/specs/bad")

    async def test_missing_constitution_is_silent(self) -> None:
        """Missing constitution.md should not crash — returns empty defaults."""
        reader = _InMemoryReader(
            {
                "/specs/test/tasks.md": "- [ ] T001 Task\n",
                "/specs/test/spec.md": "# Spec\n",
                "/specs/test/plan.md": "# Plan\n",
            }
        )
        adapter = SpecKitAdapter(file_reader=reader)
        artifacts = await adapter.load("/specs/test")

        assert isinstance(artifacts.constitution, ProjectConstitution)
        assert artifacts.constitution.protect_paths == []

    async def test_routing_hints_populated(self) -> None:
        """plan.md routing hints should be reflected in SpecArtifacts."""
        reader = _InMemoryReader(
            {
                "/specs/test/tasks.md": "- [ ] T001 Task\n",
                "/specs/test/spec.md": "# Spec\n",
                "/specs/test/plan.md": "Language: Rust\nFramework: Actix\n",
            }
        )
        adapter = SpecKitAdapter(file_reader=reader)
        artifacts = await adapter.load("/specs/test")

        assert artifacts.routing_hints.get("language") == "rust"
