"""
Spec-Kit Mode A Ingestion Adapter
===================================
Author: Orchestrator core

Consumes Spec-Kit output artifacts (``spec.md``, ``plan.md``, ``tasks.md``,
and optionally ``.specify/memory/constitution.md``) and converts them into
orchestrator-native data structures.

Design decisions
----------------
- **Parser is tolerant, never crashes on constitution/plan parse misses.**
  Only ``tasks.md`` is a hard-fail path because the orchestrator cannot run
  without a task list.
- **File reading is abstracted through ``FileReaderPort``** so the adapter
  stays pure application-layer code.
- **Output matches ``decompose_project()`` contract** (``dict[str, Task]``)
  so the pipeline requires zero changes downstream.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.constitution import ProjectConstitution
from ..domain.ports import FileReaderPort
from ..models import Task, TaskType

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SpecArtifacts:
    """Parsed contents of a Spec-Kit output directory.

    Attributes:
        tasks: Map of task ID → Task (same shape as ``decompose_project()``).
        constitution: Parsed project constitution (empty defaults if missing).
        routing_hints: Optional keyword hints extracted from ``plan.md``
            that can bias model selection. Best-effort — never crashes.
        raw_spec_criteria: List of success criteria extracted from ``spec.md``.
    """

    tasks: dict[str, Task] = field(default_factory=dict)
    constitution: ProjectConstitution = field(default_factory=ProjectConstitution)
    routing_hints: dict[str, Any] = field(default_factory=dict)
    raw_spec_criteria: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Parser: tasks.md
# ─────────────────────────────────────────────────────────────────────────────

# Regex for a task checklist line:
#   "- [ ] T001 [...] Description"
# Group 1 = task ID (e.g. "T001")
# Group 2 = optional [P] marker
# Group 3 = optional [Story] marker (e.g. [US1])
# Group 4 = description text
_TASK_LINE_RE = re.compile(
    r"^\s*-\s*\[\s*\]\s+"
    r"(?P<id>T\d{3,})"  # e.g. T001, T010
    r"(?:\s+\[(?P<parallel>P)\])?"  # optional [P]
    r"(?:\s+\[(?P<story>[^\]]+)\])?"  # optional [US1] / [StoryName]
    r"(?:\s+(?P<desc>.+))?$",
    re.IGNORECASE,
)


def parse_tasks_md(text: str) -> dict[str, Task]:
    """Parse Spec-Kit ``tasks.md`` into a ``dict[str, Task]``.

    The tasks.md checklist format (from the Spec-Kit template):

        - [ ] T001 [P] [US1] Create entity model in src/models/entity.py
        - [ ] T002 [US1] Implement service in src/services/service.py

    File paths embedded in the description are extracted as ``target_path``.

    Raises
    ------
    ValueError
        If the file contains no parseable tasks, with line-level diagnostics.
    """
    tasks: dict[str, Task] = {}
    errors: list[str] = []

    lines = text.split("\n")
    for lineno, line in enumerate(lines, start=1):
        m = _TASK_LINE_RE.match(line)
        if not m:
            continue

        tid = m.group("id")
        if not tid:
            continue

        desc = (m.group("desc") or "").strip()

        # Detect file paths in the description — heuristic: anything ending
        # in a code extension or a path with / or \
        file_path = ""
        path_pattern = re.compile(
            r"(?:src/|tests/|backend/|frontend/|api/|app/|lib/|docs/|scripts/)\S+" r"(?:\.\w+)"
        )
        path_match = path_pattern.search(desc)
        if path_match:
            file_path = path_match.group(0)

        # Determine story group for dependency grouping
        story = m.group("story") or ""

        # Build the task
        task = Task(
            id=tid,
            type=TaskType.CODE_GEN,  # default; refined later
            prompt=desc,
            context="",
            target_path=file_path,
            tech_context=story,
            dependencies=[],
            # Defaults from Task dataclass
        )
        tasks[tid] = task

    if not tasks:
        errors.append("No parseable tasks found in tasks.md (expected T001, T002, ...)")

    if errors:
        raise ValueError("; ".join(errors))

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Parser: constitution.md (Spec-Kit markdown → ProjectConstitution)
# ─────────────────────────────────────────────────────────────────────────────

# Keywords for mapping constitution principles to protect_paths
_PROTECTED_PATH_KEYWORDS = [
    "protect",
    "do not modify",
    "never modify",
    "read-only",
    "generated",
    "do not edit",
    "auto-generated",
]

# Keywords for mapping to forbidden_imports
_FORBIDDEN_IMPORT_KEYWORDS = [
    "forbidden import",
    "banned import",
    "do not import",
    "avoid dependency",
    "no dependency",
    "must not depend on",
    "must not import",
    "prohibited",
    "forbidden package",
]


def parse_constitution_md(text: str) -> ProjectConstitution:
    """Parse Spec-Kit ``constitution.md`` into a ``ProjectConstitution``.

    Mapping (best-effort, never crashes):

    - Principles mentioning "protect", "do not modify" etc. → ``protect_paths``
    - Principles mentioning "forbidden import", "no dependency" → ``forbidden_imports``
    - Principles mentioning "test" / "tdd" / "test-first" → ``require_tests: True``
    - Principles mentioning "validator" / "lint" / "ruff" → ``required_validators``
    - Everything else yields a warning, never a crash.

    Returns empty ``ProjectConstitution`` if nothing can be parsed.
    """
    protections: list[str] = []
    forbidden: list[str] = []
    validators: list[str] = []
    require_tests: bool = False

    lower = text.lower()
    lines = text.split("\n")

    # Scan for protect_paths
    for line in lines:
        stripped = line.strip()
        for kw in _PROTECTED_PATH_KEYWORDS:
            if kw in stripped.lower():
                # Try to extract a path pattern from the line. Use \S* (not \S+)
                # so a bare directory token like "tests/" is captured even when
                # followed by prose (e.g. "Do not modify tests/ directory ...").
                path_match = re.search(r"(?:src/|tests/|docs/|config/)\S*", stripped)
                if path_match:
                    p = path_match.group(0).rstrip(".,;")
                    if p not in protections:
                        protections.append(p)
                break

    # Scan for forbidden imports
    in_principles = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_principles = "principle" in stripped.lower() or "core" in stripped.lower()

        if not in_principles:
            continue

        for kw in _FORBIDDEN_IMPORT_KEYWORDS:
            if kw in stripped.lower():
                # Try to extract a package name
                words = stripped.split()
                for w in words:
                    w_clean = w.strip("`'\".,;:!")
                    if w_clean and w_clean[0].isascii() and w_clean.islower():
                        if w_clean not in forbidden and "." not in w_clean:
                            forbidden.append(w_clean)
                break

    # Scan for require_tests indicators
    test_indicators = [
        "test-first",
        "tdd",
        "test backed",
        "test-first (non-negotiable)",
        "tests mandatory",
        "write tests first",
        "test must",
        "tests gate",
        "every behavioral change",
    ]
    for line in lines:
        stripped = line.strip()
        for ti in test_indicators:
            if ti in stripped.lower():
                require_tests = True
                break

    # Scan for required validators
    validator_indicators = [
        ("ruff", "ruff"),
        ("mypy", "mypy"),
        ("pylint", "pylint"),
        ("black", "black"),
        ("bandit", "bandit"),
        ("pytest", "pytest"),
    ]
    for line in lines:
        stripped = line.strip()
        for keyword, validator_name in validator_indicators:
            if keyword in stripped.lower() and validator_name not in validators:
                validators.append(validator_name)

    return ProjectConstitution(
        protect_paths=protections,
        forbidden_imports=forbidden,
        require_tests=require_tests,
        required_validators=validators,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parser: plan.md (tech-context → routing hints)
# ─────────────────────────────────────────────────────────────────────────────


def parse_plan_md(text: str) -> dict[str, Any]:
    """Extract tech-stack keywords from ``plan.md`` for routing bias.

    Returns a dict with keys like ``tech_stack``, ``language``, etc. — used
    as best-effort hints for ``ROUTING_TABLE`` selection. Never crashes.
    """
    hints: dict[str, Any] = {}
    lower = text.lower()

    # Detect language
    language_map: dict[str, str] = {
        "python": "python",
        "typescript": "typescript",
        "javascript": "javascript",
        "go": "go",
        "rust": "rust",
        "java": "java",
        "kotlin": "kotlin",
        "dart": "dart",
    }
    for lang_key, lang_val in language_map.items():
        if lang_key in lower:
            hints["language"] = lang_val
            break

    # Detect framework / stack
    stack_keywords = [
        "fastapi",
        "django",
        "flask",
        "react",
        "vue",
        "angular",
        "nextjs",
        "nestjs",
        "spring",
        "express",
    ]
    for kw in stack_keywords:
        if kw in lower:
            if "stack" not in hints:
                hints["stack"] = []
            hints["stack"].append(kw)

    return hints


# ─────────────────────────────────────────────────────────────────────────────
# Parser: spec.md (success criteria)
# ─────────────────────────────────────────────────────────────────────────────


def parse_spec_md(text: str) -> list[str]:
    """Extract success criteria / acceptance criteria from ``spec.md``.

    Looks for bullet points under headings containing "criteria", "acceptance",
    "success", or "definition of done". Returns list of criteria strings.
    """
    criteria: list[str] = []
    in_dod = False

    for line in text.split("\n"):
        stripped = line.strip()

        # Detect criteria section headings
        if stripped.startswith("#") and any(
            kw in stripped.lower()
            for kw in ["criteria", "acceptance", "definition of done", "success"]
        ):
            in_dod = True
            continue
        elif stripped.startswith("#") and in_dod:
            # Left criteria section
            in_dod = False
            continue

        if not in_dod:
            continue

        # Collect bullet points
        if stripped.startswith("-") or stripped.startswith("*"):
            bullet = stripped.lstrip("-* ").strip()
            if bullet:
                criteria.append(bullet)

    return criteria


# ─────────────────────────────────────────────────────────────────────────────
# SpecKitAdapter
# ─────────────────────────────────────────────────────────────────────────────


class SpecKitAdapter:
    """Main adapter: loads a Spec-Kit output directory into orchestrator types.

    Typical usage::

        adapter = SpecKitAdapter(file_reader=my_reader)
        artifacts = await adapter.load("/path/to/specs/001-feature/")
        # artifacts.tasks → dict[str, Task] (same as decompose_project())
        # artifacts.constitution → ProjectConstitution
        # artifacts.routing_hints → dict for model selection bias
    """

    _EXPECTED_FILES: list[str] = [
        "tasks.md",
        "spec.md",
        "plan.md",
    ]

    def __init__(self, file_reader: FileReaderPort) -> None:
        self._reader = file_reader

    async def load(self, spec_dir: str | Path) -> SpecArtifacts:
        """Load and parse a Spec-Kit output directory.

        Parameters
        ----------
        spec_dir: Path to a Spec-Kit output directory containing at least
            ``tasks.md`` and ``spec.md`` (``plan.md`` is optional, as is
            ``.specify/memory/constitution.md``).

        Returns
        -------
        ``SpecArtifacts`` with parsed tasks, constitution, and routing hints.

        Raises
        ------
        FileNotFoundError
            If ``tasks.md`` is missing.
        ValueError
            If ``tasks.md`` contains no parseable tasks.
        """
        spec_path = Path(spec_dir)

        # ── tasks.md (required) ──────────────────────────────────────────
        tasks_path = spec_path / "tasks.md"
        try:
            tasks_text = await self._reader.read_text(str(tasks_path))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Spec-Kit directory missing required file: {tasks_path}"
            ) from None

        tasks = parse_tasks_md(tasks_text)

        # ── spec.md (optional, best-effort) ──────────────────────────────
        spec_path_md = spec_path / "spec.md"
        raw_spec_criteria: list[str] = []
        try:
            spec_text = await self._reader.read_text(str(spec_path_md))
            raw_spec_criteria = parse_spec_md(spec_text)
        except FileNotFoundError:
            logger.info("No spec.md found at %s — skipping criteria extraction", spec_path_md)

        # ── plan.md (optional, best-effort) ──────────────────────────────
        plan_path = spec_path / "plan.md"
        routing_hints: dict[str, Any] = {}
        try:
            plan_text = await self._reader.read_text(str(plan_path))
            routing_hints = parse_plan_md(plan_text)
        except FileNotFoundError:
            logger.info("No plan.md found at %s — skipping routing hints", plan_path)

        # ── .specify/memory/constitution.md (optional) ───────────────────
        constitution = ProjectConstitution()
        constitution_path = spec_path / ".specify" / "memory" / "constitution.md"
        try:
            constitution_text = await self._reader.read_text(str(constitution_path))
            constitution = parse_constitution_md(constitution_text)
            logger.info(
                "Loaded constitution from %s (%d rules)",
                constitution_path,
                len([k for k, v in vars(constitution).items() if v]),
            )
        except FileNotFoundError:
            logger.info(
                "No constitution.md found at %s — using empty defaults",
                constitution_path,
            )

        return SpecArtifacts(
            tasks=tasks,
            constitution=constitution,
            routing_hints=routing_hints,
            raw_spec_criteria=raw_spec_criteria,
        )
