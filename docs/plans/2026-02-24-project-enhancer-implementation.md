# Project Enhancer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Before decomposing a project, use an LLM to suggest 3–7 concrete improvements to the description and success criteria, which the user can accept or reject individually.

**Architecture:** A new `orchestrator/enhancer.py` module contains the `Enhancement` dataclass and `ProjectEnhancer` class. Pure logic functions (`_parse_enhancements`, `_apply_enhancements`, `_select_enhance_model`) are decoupled from I/O and fully unit-testable. `ProjectEnhancer.analyze()` makes one `UnifiedClient.call()` and raises no exceptions — it silently falls back to the original spec on any error. The CLI gains a `--no-enhance` flag.

**Tech Stack:** Python 3.10+, dataclasses, json, asyncio, `orchestrator.api_clients.UnifiedClient`, `orchestrator.models.Model`

---

## Background (read before starting)

### Key files

| File | What to know |
|------|-------------|
| `orchestrator/api_clients.py` | `UnifiedClient.call(model, prompt, system, max_tokens, timeout)` → `APIResponse(.text, .cost_usd)` |
| `orchestrator/models.py` | `Model.DEEPSEEK_CHAT`, `Model.DEEPSEEK_REASONER` enums |
| `orchestrator/cli.py:417` | `_async_new_project(args)` — this is where enhancement goes |
| `orchestrator/cli.py:534` | `main()` / `parser.add_argument(...)` — where `--no-enhance` flag goes |
| `orchestrator/__init__.py:19` | Pattern for adding imports and exports |

### How UnifiedClient is used in this codebase

```python
from orchestrator.api_clients import UnifiedClient
client = UnifiedClient()
response = await client.call(
    model=Model.DEEPSEEK_CHAT,
    prompt="...",
    system="...",
    max_tokens=1500,
    temperature=0.3,
    timeout=10,   # seconds
)
print(response.text)    # str
print(response.cost_usd)  # float
```

---

## Task 1: Enhancement dataclass and pure utility functions

**Files:**
- Create: `orchestrator/enhancer.py`
- Test: `tests/test_enhancer.py`

### Step 1: Write the failing tests for pure functions

Create `tests/test_enhancer.py`:

```python
"""Tests for Enhancement dataclass and pure utility functions."""
from __future__ import annotations
import json
import pytest
from orchestrator.enhancer import (
    Enhancement,
    _select_enhance_model,
    _parse_enhancements,
    _apply_enhancements,
)
from orchestrator.models import Model


# ─── _select_enhance_model ────────────────────────────────────────────────────

def test_select_model_short():
    """≤50 words → DeepSeek Chat (fast + cheap)."""
    desc = "Build a FastAPI auth service with JWT tokens"  # 9 words
    assert _select_enhance_model(desc) == Model.DEEPSEEK_CHAT


def test_select_model_long():
    """51+ words → DeepSeek Reasoner (better for complex analysis)."""
    desc = " ".join(["word"] * 51)
    assert _select_enhance_model(desc) == Model.DEEPSEEK_REASONER


def test_select_model_exactly_50_words():
    """Exactly 50 words → DeepSeek Chat (boundary: >50 triggers Reasoner)."""
    desc = " ".join(["word"] * 50)
    assert _select_enhance_model(desc) == Model.DEEPSEEK_CHAT


# ─── _parse_enhancements ─────────────────────────────────────────────────────

VALID_JSON = json.dumps([
    {
        "type": "completeness",
        "title": "Missing: refresh tokens",
        "description": "JWT auth without refresh tokens forces re-login.",
        "patch_description": "with JWT access tokens (24h expiry) and refresh tokens (7d)",
        "patch_criteria": "refresh token endpoint returns 200 with new access token",
    },
    {
        "type": "criteria",
        "title": "Vague success criteria",
        "description": "Tests pass is not measurable.",
        "patch_description": "",
        "patch_criteria": "≥80% test coverage, all endpoints return correct HTTP status codes",
    },
])


def test_parse_enhancements_valid():
    """Valid JSON → list[Enhancement] with correct fields."""
    result = _parse_enhancements(VALID_JSON)
    assert len(result) == 2
    assert result[0].type == "completeness"
    assert result[0].title == "Missing: refresh tokens"
    assert result[0].patch_description == "with JWT access tokens (24h expiry) and refresh tokens (7d)"
    assert result[1].type == "criteria"
    assert result[1].patch_criteria == "≥80% test coverage, all endpoints return correct HTTP status codes"


def test_parse_enhancements_invalid_json():
    """Invalid JSON → empty list, no exception."""
    result = _parse_enhancements("not json at all {{{")
    assert result == []


def test_parse_enhancements_empty_array():
    """Empty JSON array → empty list."""
    result = _parse_enhancements("[]")
    assert result == []


def test_parse_enhancements_missing_keys():
    """Items missing required keys are skipped gracefully."""
    bad_json = json.dumps([{"type": "completeness"}])  # missing title etc.
    result = _parse_enhancements(bad_json)
    assert result == []


# ─── _apply_enhancements ─────────────────────────────────────────────────────

def _make_enhancement(patch_desc: str, patch_crit: str) -> Enhancement:
    return Enhancement(
        type="completeness",
        title="Test",
        description="Test enhancement.",
        patch_description=patch_desc,
        patch_criteria=patch_crit,
    )


def test_apply_patches_description():
    """patch_description is appended to description."""
    e = _make_enhancement("with bcrypt password hashing (cost 12)", "")
    desc, crit = _apply_enhancements("Build an auth service", "tests pass", [e])
    assert desc == "Build an auth service with bcrypt password hashing (cost 12)"
    assert crit == "tests pass"  # unchanged when patch_criteria is empty


def test_apply_patches_criteria():
    """Non-empty patch_criteria is appended to criteria."""
    e = _make_enhancement("", "≥80% coverage")
    desc, crit = _apply_enhancements("Build an auth service", "tests pass", [e])
    assert crit == "tests pass; ≥80% coverage"
    assert desc == "Build an auth service"  # unchanged when patch_description is empty


def test_apply_patches_empty_criteria():
    """Empty patch_criteria leaves criteria unchanged."""
    e = _make_enhancement("with rate limiting (100 req/min per IP)", "")
    _, crit = _apply_enhancements("desc", "tests pass", [e])
    assert crit == "tests pass"


def test_apply_no_accepted():
    """Empty accepted list → description and criteria unchanged."""
    desc, crit = _apply_enhancements("original desc", "original crit", [])
    assert desc == "original desc"
    assert crit == "original crit"


def test_apply_multiple_patches():
    """Multiple accepted enhancements → all appended in order."""
    e1 = _make_enhancement("with bcrypt", "≥80% coverage")
    e2 = _make_enhancement("with rate limiting", "rate limit returns 429")
    desc, crit = _apply_enhancements("Build auth", "tests pass", [e1, e2])
    assert desc == "Build auth with bcrypt with rate limiting"
    assert crit == "tests pass; ≥80% coverage; rate limit returns 429"
```

### Step 2: Run to verify all tests fail

```bash
cd "E:/Documents/Vibe-Coding/Ai Orchestrator/.claude/worktrees/lucid-brahmagupta"
python -m pytest tests/test_enhancer.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'orchestrator.enhancer'`

### Step 3: Create `orchestrator/enhancer.py` with pure functions

```python
"""
Project Enhancer — LLM-powered spec improvement before decomposition.
=====================================================================
Before a project runs, makes one LLM call to suggest 3–7 concrete
improvements to the project description and success criteria.
User accepts/rejects each individually. Accepted patches are injected
before decomposition.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .models import Model

logger = logging.getLogger("orchestrator.enhancer")


@dataclass
class Enhancement:
    """One suggested improvement to the project spec."""
    type: str             # "completeness" | "criteria" | "risk"
    title: str            # ≤8 words, e.g. "Missing: refresh tokens"
    description: str      # Why it matters (1–2 sentences)
    patch_description: str  # Clause to append to project description ("" if none)
    patch_criteria: str   # Clause to append to success criteria ("" if none)


def _select_enhance_model(description: str) -> Model:
    """Auto-select model based on description length.

    >50 words → DeepSeek Reasoner (o1-class, better for complex analysis)
    ≤50 words → DeepSeek Chat (fast + cheap for short descriptions)
    """
    return (
        Model.DEEPSEEK_REASONER
        if len(description.split()) > 50
        else Model.DEEPSEEK_CHAT
    )


def _parse_enhancements(llm_output: str) -> list[Enhancement]:
    """Parse LLM JSON output into Enhancement objects.

    Returns empty list on any parse error — never raises.
    """
    try:
        items = json.loads(llm_output)
        if not isinstance(items, list):
            return []
        result = []
        for item in items:
            try:
                result.append(Enhancement(
                    type=item["type"],
                    title=item["title"],
                    description=item["description"],
                    patch_description=item.get("patch_description", ""),
                    patch_criteria=item.get("patch_criteria", ""),
                ))
            except (KeyError, TypeError):
                logger.warning("Skipping malformed enhancement item: %s", item)
        return result
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Enhancement JSON parse failed: %s", exc)
        return []


def _apply_enhancements(
    description: str,
    criteria: str,
    accepted: list[Enhancement],
) -> tuple[str, str]:
    """Apply accepted enhancements to description and criteria.

    patch_description clauses are appended to description (space-separated).
    patch_criteria clauses are appended to criteria (semicolon-separated).
    Empty patches are skipped.
    """
    for e in accepted:
        if e.patch_description:
            description = f"{description} {e.patch_description}"
        if e.patch_criteria:
            criteria = f"{criteria}; {e.patch_criteria}"
    return description, criteria
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_enhancer.py -v -k "select_model or parse_enhancements or apply"
```

Expected: `12 passed`

### Step 5: Commit

```bash
git add orchestrator/enhancer.py tests/test_enhancer.py
git commit -m "feat: add Enhancement dataclass and pure utility functions"
```

---

## Task 2: ProjectEnhancer class with LLM call

**Files:**
- Modify: `orchestrator/enhancer.py` (add `ProjectEnhancer` class)
- Test: `tests/test_enhancer.py` (add `TestProjectEnhancerAnalyze`)

### Step 1: Write failing tests for `ProjectEnhancer.analyze()`

Add to `tests/test_enhancer.py`:

```python
# ─── ProjectEnhancer.analyze() ───────────────────────────────────────────────

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from orchestrator.enhancer import ProjectEnhancer


class FakeAPIResponse:
    """Minimal stand-in for APIResponse."""
    def __init__(self, text: str, cost_usd: float = 0.001):
        self.text = text
        self.cost_usd = cost_usd


@pytest.mark.asyncio
async def test_analyze_returns_enhancements():
    """analyze() calls LLM and returns parsed Enhancement list."""
    fake_response = FakeAPIResponse(text=VALID_JSON, cost_usd=0.002)
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=fake_response)

    enhancer = ProjectEnhancer(client=mock_client)
    result = await enhancer.analyze(
        description="Build a FastAPI auth service",
        criteria="tests pass",
    )
    assert len(result) == 2
    assert result[0].type == "completeness"
    mock_client.call.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_handles_invalid_json():
    """analyze() returns [] when LLM output is not valid JSON."""
    fake_response = FakeAPIResponse(text="sorry, I cannot help with that")
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=fake_response)

    enhancer = ProjectEnhancer(client=mock_client)
    result = await enhancer.analyze("Build something", "tests pass")
    assert result == []


@pytest.mark.asyncio
async def test_analyze_handles_llm_exception():
    """analyze() returns [] when LLM call raises an exception."""
    mock_client = MagicMock()
    mock_client.call = AsyncMock(side_effect=Exception("network error"))

    enhancer = ProjectEnhancer(client=mock_client)
    result = await enhancer.analyze("Build something", "tests pass")
    assert result == []


@pytest.mark.asyncio
async def test_analyze_selects_reasoner_for_long_desc():
    """Long descriptions (>50 words) use DeepSeek Reasoner."""
    long_desc = " ".join(["word"] * 51)
    fake_response = FakeAPIResponse(text="[]")
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=fake_response)

    enhancer = ProjectEnhancer(client=mock_client)
    await enhancer.analyze(long_desc, "criteria")

    call_kwargs = mock_client.call.call_args
    assert call_kwargs[1]["model"] == Model.DEEPSEEK_REASONER or \
           call_kwargs[0][0] == Model.DEEPSEEK_REASONER


@pytest.mark.asyncio
async def test_analyze_selects_chat_for_short_desc():
    """Short descriptions (≤50 words) use DeepSeek Chat."""
    fake_response = FakeAPIResponse(text="[]")
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=fake_response)

    enhancer = ProjectEnhancer(client=mock_client)
    await enhancer.analyze("Build a FastAPI service", "tests pass")

    call_kwargs = mock_client.call.call_args
    assert call_kwargs[1]["model"] == Model.DEEPSEEK_CHAT or \
           call_kwargs[0][0] == Model.DEEPSEEK_CHAT
```

**Note:** You need `pytest-asyncio`. Install it if not present:
```bash
pip install pytest-asyncio
```

Also add `pytest.ini` or `pyproject.toml` setting if not present:
```ini
# In pytest.ini or setup.cfg [tool:pytest]
asyncio_mode = auto
```

Check if it already exists:
```bash
cat "E:/Documents/Vibe-Coding/Ai Orchestrator/.claude/worktrees/lucid-brahmagupta/pytest.ini" 2>/dev/null || \
cat "E:/Documents/Vibe-Coding/Ai Orchestrator/.claude/worktrees/lucid-brahmagupta/pyproject.toml" 2>/dev/null | head -20
```

### Step 2: Run to verify tests fail

```bash
python -m pytest tests/test_enhancer.py -v -k "analyze" 2>&1 | head -20
```

Expected: `ImportError` or `AttributeError: module 'orchestrator.enhancer' has no attribute 'ProjectEnhancer'`

### Step 3: Add `ProjectEnhancer` class to `orchestrator/enhancer.py`

Append to `orchestrator/enhancer.py` (after the existing functions):

```python
_SYSTEM_PROMPT = """\
You are a senior software architect reviewing a project spec before implementation.
Your job: identify real omissions, vague success criteria, and architectural risks.
Be concrete and brief. No padding.\
"""

_USER_PROMPT_TEMPLATE = """\
PROJECT: {description}
SUCCESS CRITERIA: {criteria}

Return a JSON array of 3–7 improvements. Each item:
{{
  "type": "completeness" | "criteria" | "risk",
  "title": "short label (≤8 words)",
  "description": "why this matters (1–2 sentences)",
  "patch_description": "concise clause to append to project description",
  "patch_criteria": "concise clause to append to success criteria, or empty string"
}}

Rules:
- Only real omissions and risks — no generic advice
- patch_description must be a clause, not a full sentence
- Examples of good patch_description: "with bcrypt password hashing (cost 12)",
  "including Alembic database migrations", "with rate limiting (100 req/min per IP)"
- Examples of bad patch_description: "make sure to add security", "improve the code"
- Return ONLY the JSON array, no markdown fences\
"""

_BUDGET_CAP_USD = 0.10   # hard cap per enhancement call
_TIMEOUT_S = 10          # LLM call timeout in seconds


class ProjectEnhancer:
    """
    Analyzes a project spec before decomposition and returns concrete suggestions.

    Usage:
        enhancer = ProjectEnhancer()
        suggestions = await enhancer.analyze(description, criteria)
        # suggestions is list[Enhancement] — empty on any failure
    """

    def __init__(self, client=None):
        """client: UnifiedClient instance (created fresh if None)."""
        if client is None:
            from .api_clients import UnifiedClient
            client = UnifiedClient()
        self._client = client

    async def analyze(self, description: str, criteria: str) -> list[Enhancement]:
        """Call LLM and return parsed enhancements.

        Returns [] on any error (JSON parse failure, LLM exception, timeout).
        Never raises.
        """
        model = _select_enhance_model(description)
        model_label = "DeepSeek Reasoner" if model == Model.DEEPSEEK_REASONER else "DeepSeek Chat"
        print(f"\n⚡ Analyzing project spec ({model_label})...")

        prompt = _USER_PROMPT_TEMPLATE.format(
            description=description,
            criteria=criteria,
        )

        try:
            response = await self._client.call(
                model=model,
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                max_tokens=1500,
                temperature=0.3,
                timeout=_TIMEOUT_S,
            )

            if response.cost_usd > _BUDGET_CAP_USD:
                logger.warning(
                    "Enhancement call exceeded budget cap ($%.4f > $%.2f) — skipping",
                    response.cost_usd,
                    _BUDGET_CAP_USD,
                )
                return []

            return _parse_enhancements(response.text)

        except Exception as exc:
            logger.warning("Enhancement LLM call failed: %s — skipping", exc)
            return []
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_enhancer.py -v -k "analyze"
```

Expected: `5 passed`

### Step 5: Commit

```bash
git add orchestrator/enhancer.py tests/test_enhancer.py
git commit -m "feat: add ProjectEnhancer.analyze() with mocked LLM tests"
```

---

## Task 3: `_present_enhancements()` interactive prompting

**Files:**
- Modify: `orchestrator/enhancer.py` (add `_present_enhancements` function)
- Test: `tests/test_enhancer.py` (add interaction tests)

### Step 1: Write failing tests for `_present_enhancements`

Add to `tests/test_enhancer.py`:

```python
# ─── _present_enhancements ────────────────────────────────────────────────────

from orchestrator.enhancer import _present_enhancements


def _make_three_enhancements() -> list[Enhancement]:
    return [
        Enhancement("completeness", "Missing: refresh tokens",
                    "JWT auth needs refresh tokens.", "with refresh tokens (7d)", ""),
        Enhancement("criteria", "Vague success criteria",
                    "Tests pass is unmeasurable.", "", "≥80% test coverage"),
        Enhancement("risk", "Missing: password hashing",
                    "Plain-text passwords are insecure.", "with bcrypt (cost 12)", ""),
    ]


def test_present_user_accepts_all(monkeypatch):
    """All 'y' responses → all enhancements returned."""
    monkeypatch.setattr("builtins.input", lambda _: "y")
    accepted = _present_enhancements(_make_three_enhancements())
    assert len(accepted) == 3


def test_present_user_rejects_all(monkeypatch):
    """All 'n' responses → empty list returned."""
    monkeypatch.setattr("builtins.input", lambda _: "n")
    accepted = _present_enhancements(_make_three_enhancements())
    assert accepted == []


def test_present_user_mixed(monkeypatch):
    """Mixed responses → only 'y' ones returned."""
    responses = iter(["y", "n", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    accepted = _present_enhancements(_make_three_enhancements())
    assert len(accepted) == 2
    assert accepted[0].title == "Missing: refresh tokens"
    assert accepted[1].title == "Missing: password hashing"


def test_present_empty_enhancements():
    """Empty list → prints completion message, returns empty list."""
    result = _present_enhancements([])
    assert result == []


def test_present_default_is_yes(monkeypatch):
    """Empty Enter (no input) → treated as 'y' (accept)."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    accepted = _present_enhancements(_make_three_enhancements())
    assert len(accepted) == 3


def test_present_ctrl_c_treated_as_no(monkeypatch):
    """KeyboardInterrupt on any prompt → reject all remaining, return what was accepted so far."""
    call_count = 0
    def mock_input(_):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise KeyboardInterrupt
        return "y"
    monkeypatch.setattr("builtins.input", mock_input)
    accepted = _present_enhancements(_make_three_enhancements())
    assert len(accepted) == 1  # first was accepted, then Ctrl-C on second
```

### Step 2: Run to verify tests fail

```bash
python -m pytest tests/test_enhancer.py -v -k "present"
```

Expected: `ImportError` — `_present_enhancements` not defined yet

### Step 3: Add `_present_enhancements` to `orchestrator/enhancer.py`

Append after `ProjectEnhancer`:

```python
def _present_enhancements(enhancements: list[Enhancement]) -> list[Enhancement]:
    """Present each enhancement interactively; return accepted ones.

    Y/y/Enter → accept. n/N → reject. Ctrl-C → reject all remaining.
    """
    if not enhancements:
        print("  ✓ Spec looks complete — no suggestions.\n")
        return []

    total = len(enhancements)
    print(f"\n  📋 {total} improvement{'s' if total != 1 else ''} found:\n")

    accepted: list[Enhancement] = []
    try:
        for i, e in enumerate(enhancements, start=1):
            type_label = e.type.upper()
            print(f"  [{i}/{total}] {e.type} — {e.title}")
            print(f"        {e.description}")
            if e.patch_description:
                print(f"        Adds: \"{e.patch_description}\"")
            if e.patch_criteria:
                print(f"        Adds criteria: \"{e.patch_criteria}\"")
            try:
                answer = input("        Apply? [Y/n]: ").strip().lower()
            except KeyboardInterrupt:
                print("\n  (interrupted — skipping remaining)")
                break
            if answer in ("", "y", "yes"):
                accepted.append(e)
            print()
    except Exception as exc:
        logger.warning("Unexpected error during enhancement prompts: %s", exc)

    applied = len(accepted)
    print(f"✓ Applied {applied}/{total} enhancements. Running enhanced project...")
    print("─" * 54)
    return accepted
```

### Step 4: Run tests to verify they pass

```bash
python -m pytest tests/test_enhancer.py -v -k "present"
```

Expected: `7 passed`

Run the full test file to confirm nothing regressed:

```bash
python -m pytest tests/test_enhancer.py -v
```

Expected: all tests pass (target: 19+)

### Step 5: Commit

```bash
git add orchestrator/enhancer.py tests/test_enhancer.py
git commit -m "feat: add _present_enhancements() with interactive accept/reject"
```

---

## Task 4: Wire into CLI — `--no-enhance` flag and integration

**Files:**
- Modify: `orchestrator/cli.py` (lines 417–440 and 534–618)
- Test: `tests/test_enhancer.py` (add CLI integration test)

### Step 1: Write failing CLI test

Add to `tests/test_enhancer.py`:

```python
# ─── CLI integration ──────────────────────────────────────────────────────────

import subprocess
import sys


def test_cli_no_enhance_flag():
    """--no-enhance flag causes ProjectEnhancer to be skipped entirely.

    We verify this by patching ProjectEnhancer.analyze and confirming it is
    never called when --no-enhance is passed.
    """
    with patch("orchestrator.enhancer.ProjectEnhancer.analyze") as mock_analyze:
        # Simulate args with no_enhance=True
        import types, asyncio
        from orchestrator.cli import _async_new_project

        args = types.SimpleNamespace(
            project="Build a FastAPI auth service",
            criteria="tests pass",
            budget=8.0,
            time=5400,
            project_id="",
            output_dir="",
            concurrency=3,
            verbose=False,
            raw_tasks=False,
            no_enhance=True,   # <-- the flag
            tracing=False,
            otlp_endpoint=None,
            dependency_report=False,
        )

        # We expect it to call AppBuilder (which we also mock to avoid real network)
        with patch("orchestrator.app_builder.AppBuilder.build", new_callable=AsyncMock) as mock_build:
            from orchestrator.app_builder import AppBuildResult
            mock_build.return_value = MagicMock(success=True, output_dir="/tmp/test")
            asyncio.run(_async_new_project(args))

        # ProjectEnhancer.analyze must NOT have been called
        mock_analyze.assert_not_called()
```

### Step 2: Run to verify it fails

```bash
python -m pytest tests/test_enhancer.py::test_cli_no_enhance_flag -v
```

Expected: `AttributeError: 'Namespace' object has no attribute 'no_enhance'` (or similar — flag not wired yet)

### Step 3: Add `--no-enhance` flag to CLI parser

In `orchestrator/cli.py`, find the block of `parser.add_argument(...)` calls (around line 609–618, before the `args = parser.parse_args()` line). Add after the `--raw-tasks` block:

```python
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        default=False,
        help=(
            "Skip LLM spec enhancement pass and run original project description directly"
        ),
    )
```

### Step 4: Wire `ProjectEnhancer` into `_async_new_project()`

In `orchestrator/cli.py`, modify `_async_new_project()` (starting at line 417).

**Current code (lines 417–440):**
```python
async def _async_new_project(args):
    raw_tasks = getattr(args, "raw_tasks", False)

    if not raw_tasks:
        # Route through AppBuilder (detects app_type automatically)
        import tempfile
        from orchestrator.app_builder import AppBuilder
        output_dir = args.output_dir or tempfile.mkdtemp(prefix="app-builder-")
        print(f"Starting app build (budget: ${args.budget})")
        print(f"Project: {args.project}")
        print(f"Criteria: {args.criteria}")
        print("-" * 60)
        builder = AppBuilder()
        result = await builder.build(
            description=args.project,
            criteria=args.criteria,
            output_dir=Path(output_dir),
        )
```

**Replace with:**
```python
async def _async_new_project(args):
    raw_tasks = getattr(args, "raw_tasks", False)
    no_enhance = getattr(args, "no_enhance", False)

    # ── Enhancement pass (before decomposition) ──────────────────────────────
    description = args.project
    criteria = args.criteria

    if not no_enhance:
        from .enhancer import ProjectEnhancer, _present_enhancements, _apply_enhancements
        enhancer = ProjectEnhancer()
        suggestions = await enhancer.analyze(description, criteria)
        if suggestions:
            accepted = _present_enhancements(suggestions)
            description, criteria = _apply_enhancements(description, criteria, accepted)
    # ─────────────────────────────────────────────────────────────────────────

    if not raw_tasks:
        # Route through AppBuilder (detects app_type automatically)
        import tempfile
        from orchestrator.app_builder import AppBuilder
        output_dir = args.output_dir or tempfile.mkdtemp(prefix="app-builder-")
        print(f"Starting app build (budget: ${args.budget})")
        print(f"Project: {description}")
        print(f"Criteria: {criteria}")
        print("-" * 60)
        builder = AppBuilder()
        result = await builder.build(
            description=description,
            criteria=criteria,
            output_dir=Path(output_dir),
        )
```

Also update the `raw_tasks` branch (lines 442–468) to use `description` and `criteria` instead of `args.project` and `args.criteria`:

Find these lines in the `raw_tasks` branch:
```python
    print(f"Project: {args.project}")
    print(f"Criteria: {args.criteria}")
```
Replace with:
```python
    print(f"Project: {description}")
    print(f"Criteria: {criteria}")
```

And find:
```python
    async for event in orch.run_project_streaming(
        project_description=args.project,
        success_criteria=args.criteria,
```
Replace with:
```python
    async for event in orch.run_project_streaming(
        project_description=description,
        success_criteria=criteria,
```

### Step 5: Run the CLI test

```bash
python -m pytest tests/test_enhancer.py::test_cli_no_enhance_flag -v
```

Expected: `PASSED`

### Step 6: Run the full test file

```bash
python -m pytest tests/test_enhancer.py -v
```

Expected: all tests pass

### Step 7: Quick manual smoke test (optional but recommended)

```bash
cd "E:/Documents/Vibe-Coding/Ai Orchestrator/.claude/worktrees/lucid-brahmagupta"
python -m orchestrator --project "Build a FastAPI auth service" --criteria "tests pass" --no-enhance --budget 0.001 2>&1 | head -10
```

Expected: starts without enhancement prompt (no `⚡ Analyzing...` line).

### Step 8: Commit

```bash
git add orchestrator/cli.py tests/test_enhancer.py
git commit -m "feat: wire ProjectEnhancer into CLI with --no-enhance flag"
```

---

## Task 5: Update `__init__.py` exports

**Files:**
- Modify: `orchestrator/__init__.py`

### Step 1: Add import to `orchestrator/__init__.py`

Find the imports block (around line 19) and add after the existing imports:

```python
from .enhancer import Enhancement, ProjectEnhancer
```

Find the `__all__` list (around line 48) and add before the closing bracket:

```python
    # Project Enhancer — spec improvement before decomposition
    "Enhancement", "ProjectEnhancer",
```

### Step 2: Verify the import works

```bash
python -c "from orchestrator import Enhancement, ProjectEnhancer; print('OK')"
```

Expected: `OK`

### Step 3: Run the full test suite to check nothing is broken

```bash
python -m pytest tests/test_enhancer.py tests/test_hooks_metrics.py tests/test_policy_governance.py -v 2>&1 | tail -20
```

Expected: all pass (or existing failures unrelated to our changes)

### Step 4: Commit

```bash
git add orchestrator/__init__.py
git commit -m "chore: export Enhancement and ProjectEnhancer from orchestrator package"
```

---

## Task 6: Final verification

### Step 1: Run the full enhancer test file

```bash
python -m pytest tests/test_enhancer.py -v
```

Expected output (all 20+ tests):
```
tests/test_enhancer.py::test_select_model_short PASSED
tests/test_enhancer.py::test_select_model_long PASSED
tests/test_enhancer.py::test_select_model_exactly_50_words PASSED
tests/test_enhancer.py::test_parse_enhancements_valid PASSED
tests/test_enhancer.py::test_parse_enhancements_invalid_json PASSED
tests/test_enhancer.py::test_parse_enhancements_empty_array PASSED
tests/test_enhancer.py::test_parse_enhancements_missing_keys PASSED
tests/test_enhancer.py::test_apply_patches_description PASSED
tests/test_enhancer.py::test_apply_patches_criteria PASSED
tests/test_enhancer.py::test_apply_patches_empty_criteria PASSED
tests/test_enhancer.py::test_apply_no_accepted PASSED
tests/test_enhancer.py::test_apply_multiple_patches PASSED
tests/test_enhancer.py::test_analyze_returns_enhancements PASSED
tests/test_enhancer.py::test_analyze_handles_invalid_json PASSED
tests/test_enhancer.py::test_analyze_handles_llm_exception PASSED
tests/test_enhancer.py::test_analyze_selects_reasoner_for_long_desc PASSED
tests/test_enhancer.py::test_analyze_selects_chat_for_short_desc PASSED
tests/test_enhancer.py::test_present_user_accepts_all PASSED
tests/test_enhancer.py::test_present_user_rejects_all PASSED
tests/test_enhancer.py::test_present_user_mixed PASSED
tests/test_enhancer.py::test_present_empty_enhancements PASSED
tests/test_enhancer.py::test_present_default_is_yes PASSED
tests/test_enhancer.py::test_present_ctrl_c_treated_as_no PASSED
tests/test_enhancer.py::test_cli_no_enhance_flag PASSED
```

### Step 2: Verify no regressions in existing tests

```bash
python -m pytest tests/ -x --ignore=tests/test_engine_e2e.py -q 2>&1 | tail -10
```

(Ignoring e2e tests which require real API keys)

Expected: existing tests still pass; only enhancer tests are new.

### Step 3: Final commit

```bash
git add -A
git commit -m "feat: complete project enhancer — LLM spec improvement before execution

Adds ProjectEnhancer that analyzes project description + success criteria
before decomposition, suggesting 3-7 concrete improvements. User accepts
or rejects each interactively. Accepted patches are injected into the
spec before AppBuilder runs.

New: orchestrator/enhancer.py (Enhancement, ProjectEnhancer, _select_enhance_model,
     _parse_enhancements, _apply_enhancements, _present_enhancements)
New: tests/test_enhancer.py (24 tests)
Modified: orchestrator/cli.py (enhancement pass + --no-enhance flag)
Modified: orchestrator/__init__.py (exports)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Quick Reference: File Change Summary

| File | Action | What changes |
|------|--------|-------------|
| `orchestrator/enhancer.py` | **CREATE** | `Enhancement` dataclass, `_select_enhance_model`, `_parse_enhancements`, `_apply_enhancements`, `ProjectEnhancer`, `_present_enhancements` |
| `tests/test_enhancer.py` | **CREATE** | 24 unit tests covering all functions |
| `orchestrator/cli.py` | **MODIFY** | Add `--no-enhance` flag; add enhancement block at top of `_async_new_project()` |
| `orchestrator/__init__.py` | **MODIFY** | Add import + `__all__` entries for `Enhancement`, `ProjectEnhancer` |
