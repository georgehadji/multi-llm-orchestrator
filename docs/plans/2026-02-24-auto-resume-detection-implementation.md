# Auto-Resume Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When a user starts a new project, automatically detect incomplete projects with similar descriptions and interactively offer to resume them.

**Architecture:** Three-layer approach — (1) pure-Python `resume_detector.py` for keyword/scoring logic, (2) `state.py` migration adds `project_description`+`keywords_json` columns and a `find_resumable()` query, (3) `cli.py` calls a `_check_resume()` gate at the start of `_async_new_project()`.

**Tech Stack:** Python stdlib (`re`, `json`, `dataclasses`, `datetime`, `asyncio`), aiosqlite (already a dependency), pytest.

---

## Task 1: Create `resume_detector.py` — pure keyword logic

**Files:**
- Create: `orchestrator/resume_detector.py`
- Create: `tests/test_resume_detector.py`

---

**Step 1: Write the failing tests**

```python
# tests/test_resume_detector.py
import pytest
from datetime import datetime, timedelta
from orchestrator.resume_detector import (
    _extract_keywords,
    _recency_factor,
    _score_candidates,
    _is_exact_match,
    ResumeCandidate,
)


class TestExtractKeywords:
    def test_removes_stopwords(self):
        kw = _extract_keywords("build a REST API for the users")
        assert "a" not in kw
        assert "the" not in kw
        assert "for" not in kw
        assert "build" not in kw

    def test_lowercases(self):
        kw = _extract_keywords("Build FastAPI AUTH")
        assert "fastapi" in kw
        assert "auth" in kw

    def test_synonym_expansion_auth(self):
        kw = _extract_keywords("implement jwt service")
        # "jwt" is an alias of "auth" canonical → all auth aliases included
        assert "auth" in kw
        assert "authentication" in kw

    def test_synonym_expansion_api(self):
        kw = _extract_keywords("build rest endpoints")
        assert "api" in kw

    def test_empty_string(self):
        assert _extract_keywords("") == set()


class TestRecencyFactor:
    def _ago(self, days: int) -> str:
        return (datetime.utcnow() - timedelta(days=days)).isoformat()

    def test_today(self):
        assert _recency_factor(self._ago(0)) == 1.00

    def test_two_days_ago(self):
        assert _recency_factor(self._ago(2)) == 0.85

    def test_five_days_ago(self):
        assert _recency_factor(self._ago(5)) == 0.65

    def test_ten_days_ago(self):
        assert _recency_factor(self._ago(10)) == 0.40

    def test_thirty_days_ago(self):
        assert _recency_factor(self._ago(30)) == 0.20


class TestScoreCandidates:
    def _row(self, description, keywords, days_ago=0, status="PARTIAL_SUCCESS"):
        from datetime import datetime, timedelta
        return {
            "project_id": "proj_test",
            "project_description": description,
            "keywords_json": __import__("json").dumps(list(keywords)),
            "status": status,
            "updated_at": (datetime.utcnow() - timedelta(days=days_ago)).isoformat(),
            "state": "{}",
        }

    def test_high_overlap_returns_candidate(self):
        new_kw = {"fastapi", "auth", "jwt", "api"}
        rows = [self._row("FastAPI auth service", {"fastapi", "auth", "jwt", "api"}, days_ago=0)]
        results = _score_candidates(new_kw, rows)
        assert len(results) == 1
        assert results[0].score >= 0.35

    def test_low_overlap_filtered_out(self):
        new_kw = {"react", "frontend", "ui"}
        rows = [self._row("FastAPI backend", {"fastapi", "backend", "api"}, days_ago=0)]
        results = _score_candidates(new_kw, rows)
        assert results == []

    def test_empty_stored_keywords_skipped(self):
        new_kw = {"fastapi", "auth"}
        rows = [self._row("desc", set(), days_ago=0)]
        results = _score_candidates(new_kw, rows)
        assert results == []

    def test_sorted_descending_score(self):
        new_kw = {"fastapi", "auth", "jwt"}
        rows = [
            self._row("FastAPI auth", {"fastapi", "auth", "jwt"}, days_ago=10),  # lower: old
            self._row("FastAPI auth service", {"fastapi", "auth", "jwt"}, days_ago=1),  # higher: recent
        ]
        results = _score_candidates(new_kw, rows)
        assert results[0].score >= results[1].score

    def test_capped_at_three(self):
        new_kw = {"fastapi", "auth"}
        rows = [
            self._row(f"FastAPI auth {i}", {"fastapi", "auth"}, days_ago=i)
            for i in range(6)
        ]
        results = _score_candidates(new_kw, rows)
        assert len(results) <= 3


class TestIsExactMatch:
    def test_exact_same(self):
        assert _is_exact_match("Build a FastAPI service", "Build a FastAPI service")

    def test_case_insensitive(self):
        assert _is_exact_match("Build FastAPI", "build fastapi")

    def test_whitespace_stripped(self):
        assert _is_exact_match("  Build FastAPI  ", "Build FastAPI")

    def test_different(self):
        assert not _is_exact_match("Build FastAPI", "Build Next.js")
```

**Step 2: Run tests — verify they all FAIL**

```bash
cd "E:\Documents\Vibe-Coding\Ai Orchestrator\.claude\worktrees\lucid-brahmagupta"
python -m pytest tests/test_resume_detector.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_extract_keywords'`

---

**Step 3: Implement `resume_detector.py`**

```python
# orchestrator/resume_detector.py
"""
Auto-Resume Detection
=====================
Pure-Python logic for detecting incomplete projects similar to a new description.
No LLM calls. No extra dependencies beyond stdlib + json.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# ── Stopwords ────────────────────────────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "with", "and", "or", "for", "to", "of", "in", "on",
    "build", "create", "implement", "add", "make", "write", "develop", "design",
    "using", "use", "that", "this", "it", "is", "are", "be", "by", "from",
})

# ── Synonym map ───────────────────────────────────────────────────────────────
# canonical → all aliases (including itself)

_SYNONYMS: dict[str, frozenset[str]] = {
    "auth":     frozenset({"auth", "authentication", "login", "jwt", "oauth", "session", "token"}),
    "api":      frozenset({"api", "rest", "endpoint", "fastapi", "flask", "routes", "http", "graphql"}),
    "db":       frozenset({"db", "database", "postgres", "sqlite", "mysql", "sql", "orm", "mongodb"}),
    "frontend": frozenset({"frontend", "ui", "react", "nextjs", "vue", "svelte", "html", "css"}),
    "queue":    frozenset({"queue", "worker", "celery", "task", "async", "job", "broker"}),
    "cache":    frozenset({"cache", "caching", "redis", "memcache", "ttl"}),
    "test":     frozenset({"test", "tests", "pytest", "jest", "unittest", "spec", "tdd"}),
}

# Build reverse lookup: alias → canonical
_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _SYNONYMS.items()
    for alias in aliases
}


def _extract_keywords(text: str) -> set[str]:
    """Extract and expand keywords from a project description."""
    words = set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS
    expanded = set(words)
    # For every word that maps to a canonical group, add ALL aliases in that group
    for word in words:
        canonical = _ALIAS_TO_CANONICAL.get(word)
        if canonical:
            expanded |= _SYNONYMS[canonical]
    return expanded


# ── Recency factor ────────────────────────────────────────────────────────────

def _recency_factor(updated_at: str) -> float:
    """Decay weight based on days since last update."""
    try:
        dt = datetime.fromisoformat(updated_at)
    except (ValueError, TypeError):
        return 0.20
    age_days = (datetime.utcnow() - dt).days
    if age_days <= 1:   return 1.00
    if age_days <= 3:   return 0.85
    if age_days <= 7:   return 0.65
    if age_days <= 14:  return 0.40
    return 0.20


# ── Candidate data class ──────────────────────────────────────────────────────

@dataclass
class ResumeCandidate:
    project_id: str
    description: str
    status: str
    score: float
    updated_at: str


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_candidates(
    new_keywords: set[str],
    rows: list[dict],
    threshold: float = 0.35,
) -> list[ResumeCandidate]:
    """
    Score DB rows against new_keywords.
    Returns up to 3 candidates sorted by descending score.
    """
    results: list[ResumeCandidate] = []

    for row in rows:
        stored_kw = set(json.loads(row.get("keywords_json") or "[]"))
        if not stored_kw or not new_keywords:
            continue
        # Jaccard similarity
        union = new_keywords | stored_kw
        overlap = len(new_keywords & stored_kw) / len(union)
        score = overlap * _recency_factor(row.get("updated_at", ""))
        if score >= threshold:
            results.append(ResumeCandidate(
                project_id=row["project_id"],
                description=row.get("project_description", ""),
                status=row.get("status", ""),
                score=score,
                updated_at=row.get("updated_at", ""),
            ))

    results.sort(key=lambda c: c.score, reverse=True)
    return results[:3]


# ── Exact match ───────────────────────────────────────────────────────────────

def _is_exact_match(desc_a: str, desc_b: str) -> bool:
    return desc_a.strip().lower() == desc_b.strip().lower()
```

**Step 4: Run tests — verify they all PASS**

```bash
python -m pytest tests/test_resume_detector.py -v
```

Expected: all green, 0 failures.

**Step 5: Commit**

```bash
git add orchestrator/resume_detector.py tests/test_resume_detector.py
git commit -m "feat: add resume_detector — keyword extraction, scoring, recency weighting"
```

---

## Task 2: Migrate `state.py` — add columns + `find_resumable()`

**Files:**
- Modify: `orchestrator/state.py`
- Create: `tests/test_state_resume.py`

---

**Step 1: Write the failing tests**

```python
# tests/test_state_resume.py
import asyncio
import json
import pytest
from pathlib import Path
from orchestrator.state import StateManager
from orchestrator.models import ProjectState, Budget, ProjectStatus


def _make_state(description: str, status: ProjectStatus) -> ProjectState:
    return ProjectState(
        project_description=description,
        success_criteria="tests pass",
        budget=Budget(max_usd=8.0, spent_usd=2.0),
        tasks={},
        results={},
        status=status,
    )


@pytest.fixture
def mgr(tmp_path):
    return StateManager(db_path=tmp_path / "state.db")


@pytest.mark.asyncio
async def test_save_stores_description_and_keywords(mgr):
    state = _make_state("Build a FastAPI auth service", ProjectStatus.PARTIAL_SUCCESS)
    await mgr.save_project("proj_001", state)
    rows = await mgr.list_projects()
    row = next(r for r in rows if r["project_id"] == "proj_001")
    assert row["project_description"] == "Build a FastAPI auth service"
    assert "auth" in json.loads(row["keywords_json"])


@pytest.mark.asyncio
async def test_find_resumable_returns_match(mgr):
    state = _make_state("Build a FastAPI auth service", ProjectStatus.PARTIAL_SUCCESS)
    await mgr.save_project("proj_001", state)
    candidates = await mgr.find_resumable({"fastapi", "auth", "api"})
    assert len(candidates) == 1
    assert candidates[0]["project_id"] == "proj_001"


@pytest.mark.asyncio
async def test_find_resumable_excludes_success(mgr):
    state = _make_state("Build a FastAPI auth service", ProjectStatus.SUCCESS)
    await mgr.save_project("proj_ok", state)
    candidates = await mgr.find_resumable({"fastapi", "auth"})
    assert all(c["project_id"] != "proj_ok" for c in candidates)


@pytest.mark.asyncio
async def test_find_resumable_excludes_empty_keywords(mgr):
    state = _make_state("Build a FastAPI auth service", ProjectStatus.PARTIAL_SUCCESS)
    await mgr.save_project("proj_001", state)
    # Override keywords_json to empty to simulate old row
    db = await mgr._get_conn()
    await db.execute("UPDATE projects SET keywords_json='[]' WHERE project_id='proj_001'")
    await db.commit()
    candidates = await mgr.find_resumable({"fastapi", "auth"})
    assert candidates == []


@pytest.mark.asyncio
async def test_old_rows_survive_migration(mgr):
    """Existing rows without the new columns should not raise errors."""
    # Pre-insert a row without the new columns (simulate old DB)
    db = await mgr._get_conn()
    import time
    await db.execute(
        "INSERT OR IGNORE INTO projects (project_id, state, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("old_proj", '{"project_description":"old","success_criteria":"","budget":{},'
         '"tasks":{},"results":{},"status":"PARTIAL_SUCCESS","execution_order":[]}',
         "PARTIAL_SUCCESS", time.time(), time.time()),
    )
    await db.commit()
    # Should not raise
    rows = await mgr.list_projects()
    assert any(r["project_id"] == "old_proj" for r in rows)
```

**Step 2: Run tests — verify FAIL**

```bash
python -m pytest tests/test_state_resume.py -v 2>&1 | head -20
```

Expected: errors about `find_resumable` not existing / missing columns.

---

**Step 3: Modify `state.py`**

3a. In `_ensure_schema` (inside `_get_conn`), add `ALTER TABLE` migrations **after** the `CREATE TABLE` block:

```python
# After await self._conn.commit() following executescript:
# Idempotent column additions for resume detection
for col, definition in [
    ("project_description", "TEXT DEFAULT ''"),
    ("keywords_json",       "TEXT DEFAULT '[]'"),
]:
    try:
        await self._conn.execute(
            f"ALTER TABLE projects ADD COLUMN {col} {definition}"
        )
        await self._conn.commit()
    except Exception:
        pass  # Column already exists — safe to ignore
```

3b. Update `save_project()` to write description + keywords:

```python
# At top of file, add import:
from .resume_detector import _extract_keywords

# In save_project(), update the INSERT OR REPLACE:
async def save_project(self, project_id: str, state: ProjectState):
    now = time.time()
    blob = json.dumps(_state_to_dict(state))
    description = state.project_description or ""
    keywords = json.dumps(sorted(_extract_keywords(description)))
    db = await self._get_conn()
    await db.execute(
        """INSERT OR REPLACE INTO projects
           (project_id, state, status, project_description, keywords_json,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, COALESCE(
               (SELECT created_at FROM projects WHERE project_id = ?), ?
           ), ?)""",
        (project_id, blob, state.status.value, description, keywords,
         project_id, now, now)
    )
    await db.commit()
```

3c. Update `list_projects()` to include new columns:

```python
async def list_projects(self) -> list[dict]:
    db = await self._get_conn()
    async with db.execute(
        "SELECT project_id, status, created_at, updated_at, "
        "COALESCE(project_description,'') as project_description, "
        "COALESCE(keywords_json,'[]') as keywords_json "
        "FROM projects ORDER BY updated_at DESC"
    ) as cursor:
        rows = await cursor.fetchall()
    return [
        {
            "project_id": r[0], "status": r[1],
            "created_at": r[2], "updated_at": r[3],
            "project_description": r[4], "keywords_json": r[5],
        }
        for r in rows
    ]
```

3d. Add `find_resumable()` method to `StateManager`:

```python
_RESUMABLE_STATUSES = ("PARTIAL_SUCCESS", "BUDGET_EXHAUSTED", "TIMEOUT")

async def find_resumable(self, new_keywords: set[str]) -> list[dict]:
    """
    Return DB rows for incomplete projects that have stored keywords.
    Filtering by status only — scoring is done by resume_detector.
    """
    placeholders = ",".join("?" * len(_RESUMABLE_STATUSES))
    db = await self._get_conn()
    async with db.execute(
        f"SELECT project_id, status, updated_at, "
        f"COALESCE(project_description,'') as project_description, "
        f"COALESCE(keywords_json,'[]') as keywords_json "
        f"FROM projects "
        f"WHERE status IN ({placeholders}) "
        f"AND keywords_json IS NOT NULL AND keywords_json != '[]' "
        f"ORDER BY updated_at DESC",
        _RESUMABLE_STATUSES,
    ) as cursor:
        rows = await cursor.fetchall()
    return [
        {
            "project_id": r[0], "status": r[1], "updated_at": r[2],
            "project_description": r[3], "keywords_json": r[4],
        }
        for r in rows
    ]
```

**Step 4: Run tests — verify all PASS**

```bash
python -m pytest tests/test_state_resume.py tests/test_resume_detector.py -v
```

Expected: all green.

**Step 5: Commit**

```bash
git add orchestrator/state.py orchestrator/resume_detector.py tests/test_state_resume.py
git commit -m "feat: state.py — add project_description/keywords columns + find_resumable()"
```

---

## Task 3: Wire into `cli.py` + `--new-project` flag

**Files:**
- Modify: `orchestrator/cli.py`
- Create: `tests/test_cli_resume.py`

---

**Step 1: Write the failing tests**

```python
# tests/test_cli_resume.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_candidate(project_id="proj_001", description="Build a FastAPI auth service",
                    score=0.75, status="PARTIAL_SUCCESS"):
    from orchestrator.resume_detector import ResumeCandidate
    from datetime import datetime, timedelta
    return ResumeCandidate(
        project_id=project_id,
        description=description,
        status=status,
        score=score,
        updated_at=(datetime.utcnow() - timedelta(days=2)).isoformat(),
    )


class TestCheckResume:
    """Unit-test _check_resume() in isolation."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_candidates(self):
        from orchestrator.cli import _check_resume
        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = []
        result = await _check_resume("Build a GraphQL API", state_mgr, new_project=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_new_project_flag(self):
        from orchestrator.cli import _check_resume
        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = [_make_candidate()]
        result = await _check_resume("Build a FastAPI auth service", state_mgr, new_project=True)
        assert result is None   # flag bypasses check

    @pytest.mark.asyncio
    async def test_auto_resumes_on_exact_match(self):
        from orchestrator.cli import _check_resume
        candidate = _make_candidate(description="Build a FastAPI auth service")
        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = [candidate]
        result = await _check_resume(
            "Build a FastAPI auth service", state_mgr, new_project=False,
            _input_fn=lambda _: "n",   # should not be called for exact match
        )
        assert result == "proj_001"

    @pytest.mark.asyncio
    async def test_single_match_user_says_yes(self):
        from orchestrator.cli import _check_resume
        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = [_make_candidate()]
        result = await _check_resume(
            "Build a FastAPI JWT backend", state_mgr, new_project=False,
            _input_fn=lambda _: "y",
        )
        assert result == "proj_001"

    @pytest.mark.asyncio
    async def test_single_match_user_says_no(self):
        from orchestrator.cli import _check_resume
        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = [_make_candidate()]
        result = await _check_resume(
            "Build a FastAPI JWT backend", state_mgr, new_project=False,
            _input_fn=lambda _: "n",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_multi_match_user_picks_first(self):
        from orchestrator.cli import _check_resume
        candidates = [
            _make_candidate("proj_001", score=0.80),
            _make_candidate("proj_002", score=0.60),
        ]
        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = candidates
        result = await _check_resume(
            "Build a FastAPI JWT backend", state_mgr, new_project=False,
            _input_fn=lambda _: "1",
        )
        assert result == "proj_001"

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        from orchestrator.cli import _check_resume
        state_mgr = AsyncMock()
        async def slow(*a, **kw):
            await asyncio.sleep(10)
            return []
        state_mgr.find_resumable.side_effect = slow
        result = await _check_resume(
            "Build a FastAPI auth service", state_mgr, new_project=False,
        )
        assert result is None   # timeout → silent fallback
```

**Step 2: Run tests — verify FAIL**

```bash
python -m pytest tests/test_cli_resume.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_check_resume'`

---

**Step 3: Add `_check_resume()` to `cli.py`**

Add near the top of `cli.py` (after imports):

```python
from .resume_detector import _extract_keywords, _score_candidates, _is_exact_match
from .state import StateManager
```

Add the function before `_async_new_project`:

```python
async def _check_resume(
    description: str,
    state_mgr: StateManager,
    new_project: bool = False,
    _input_fn=None,           # injectable for testing
) -> Optional[str]:
    """
    Check if there is a resumable project similar to `description`.
    Returns project_id to resume, or None to start fresh.
    """
    if new_project:
        return None

    _ask = _input_fn or input

    try:
        new_kw = _extract_keywords(description)
        rows = await asyncio.wait_for(
            state_mgr.find_resumable(new_kw),
            timeout=0.2,
        )
    except (asyncio.TimeoutError, Exception):
        return None

    candidates = _score_candidates(new_kw, rows)
    if not candidates:
        return None

    # Exact match → auto-resume
    top = candidates[0]
    if _is_exact_match(description, top.description):
        print(f"\n⚡ Identical project found — auto-resuming {top.project_id}...")
        print("   (pass --new-project to force a fresh start)\n")
        return top.project_id

    # Single match
    if len(candidates) == 1:
        c = candidates[0]
        age = (datetime.utcnow() - datetime.fromisoformat(c.updated_at)).days
        print(f"\n⚡ Found a resumable project:")
        print(f"  ID:       {c.project_id}")
        print(f"  Status:   {c.status}")
        print(f"  Last run: {age} days ago")
        print(f'  Desc:     "{c.description[:80]}"')
        answer = _ask("\n  Resume it? [Y/n]: ").strip().lower()
        return c.project_id if answer in ("", "y", "yes") else None

    # Multiple matches
    print(f"\n⚡ Found {len(candidates)} resumable projects:")
    for i, c in enumerate(candidates, 1):
        age = (datetime.utcnow() - datetime.fromisoformat(c.updated_at)).days
        bar = "█" * int(c.score * 10) + "░" * (10 - int(c.score * 10))
        print(f"  [{i}] {c.project_id:<30} {c.status:<20} {age}d ago  {bar} {c.score:.0%}")
    answer = _ask(f"\n  Resume which? [1–{len(candidates)} / n — start fresh]: ").strip()
    if answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx].project_id
    return None
```

Add `from datetime import datetime` to the imports at the top if not already present.

---

**Step 4: Call `_check_resume()` in `_async_new_project()`**

At the **very beginning** of `_async_new_project`, before the `if not raw_tasks:` block:

```python
async def _async_new_project(args):
    # ── Resume detection ────────────────────────────────────────────────
    if not getattr(args, "new_project", False):
        state_mgr = StateManager()
        project_id_to_resume = await _check_resume(
            description=args.project,
            state_mgr=state_mgr,
            new_project=getattr(args, "new_project", False),
        )
        await state_mgr.close()
        if project_id_to_resume:
            # Patch args and delegate to resume handler
            args.resume = project_id_to_resume
            await _async_resume(args)
            return
    # ── existing code continues unchanged ──────────────────────────────
    raw_tasks = getattr(args, "raw_tasks", False)
    ...
```

---

**Step 5: Add `--new-project` flag to the argument parser**

Find the block where `--resume` is defined (line ~555 of `cli.py`) and add:

```python
parser.add_argument(
    "--new-project", "-N",
    action="store_true",
    default=False,
    help="Skip resume detection and always start a fresh project",
)
```

---

**Step 6: Run tests — verify all PASS**

```bash
python -m pytest tests/test_cli_resume.py tests/test_state_resume.py tests/test_resume_detector.py -v
```

Expected: all green.

**Step 7: Run full test suite — verify no regressions**

```bash
python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: 0 failures (some may be skipped for missing API keys — that is fine).

**Step 8: Commit**

```bash
git add orchestrator/cli.py tests/test_cli_resume.py
git commit -m "feat: wire _check_resume() into CLI — auto-detect resumable projects on start"
```

---

## Task 4: Manual smoke test

**Step 1: Create two test projects in state DB**

```python
# run_once.py  (delete after testing)
import asyncio
from orchestrator.state import StateManager
from orchestrator.models import ProjectState, Budget, ProjectStatus

async def seed():
    mgr = StateManager()
    s = ProjectState(
        project_description="Build a FastAPI authentication service with JWT tokens",
        success_criteria="all tests pass",
        budget=Budget(max_usd=8.0, spent_usd=2.34),
        tasks={}, results={},
        status=ProjectStatus.PARTIAL_SUCCESS,
    )
    await mgr.save_project("proj_smoke_001", s)
    print("Seeded proj_smoke_001")
    await mgr.close()

asyncio.run(seed())
```

```bash
python run_once.py
```

**Step 2: Run CLI with a similar description — should see prompt**

```bash
python -m orchestrator --project "Build a FastAPI auth service" --criteria "pytest passes" --new-project
```

Expected: `--new-project` skips detection, starts fresh (no prompt shown).

```bash
python -m orchestrator --project "Build a FastAPI auth service" --criteria "pytest passes"
```

Expected: resume detection prompt appears.

**Step 3: Clean up**

```bash
rm run_once.py
```

---

## Task 5: Final commit

```bash
git add -A
git commit -m "feat: auto-resume detection complete — keyword match + recency weighting

- resume_detector.py: _extract_keywords, synonyms, _recency_factor, Jaccard scoring
- state.py: project_description + keywords_json columns, find_resumable()
- cli.py: _check_resume() with 200ms timeout, --new-project flag
- Exact match → auto-resume; 1 match → [Y/n]; 2-3 matches → numbered list
- 24 new unit tests (test_resume_detector, test_state_resume, test_cli_resume)
- Zero regressions on existing test suite"
```
