# Design: Auto-Resume Detection on Project Start

**Date:** 2026-02-24
**Status:** Approved
**Scope:** When the user starts a new project, automatically detect incomplete projects and offer to resume

---

## Problem Statement

Currently the user must manually run `--list-projects` to find incomplete runs, then explicitly pass `--resume <project_id>`. This is friction-heavy. Interrupted projects (PARTIAL_SUCCESS, BUDGET_EXHAUSTED, TIMEOUT) are often abandoned simply because the user forgets they exist.

---

## Approved Design

### Approach: Keyword overlap + recency weighting (Approach C + R1–R7 refinements)

No LLM calls. No external dependencies. Pure Python + SQLite.

---

## Architecture

```
CLI._async_new_project()
  │
  ├── 1. Extract keywords from new description
  │         _extract_keywords(description) → set[str]
  │         + synonym expansion (SYNONYMS dict)
  │
  ├── 2. Query DB for incomplete projects
  │         StateManager.find_resumable(keywords) → list[ResumeCandidate]
  │         SELECT project_id, project_description, keywords_json,
  │                status, created_at, updated_at
  │         WHERE status IN ('PARTIAL_SUCCESS','BUDGET_EXHAUSTED','TIMEOUT')
  │         (200ms hard timeout — falls back to [] on failure)
  │
  ├── 3. Score each candidate
  │         overlap  = Jaccard(new_keywords, stored_keywords)
  │         score    = overlap × recency_factor(updated_at)
  │         filter   score ≥ 0.35
  │         sort     descending
  │
  └── 4. Prompt user (or auto-resume)
            0 matches   → proceed silently
            exact match → auto-resume (print 1 line)
            1 match     → single [Y/n] prompt
            2–3 matches → numbered list [1/2/.../n]
```

---

## Components

### R1 — DB Schema: add columns to `projects` table

`StateManager._ensure_schema()` adds two new columns:

```sql
project_description TEXT DEFAULT ''
keywords_json       TEXT DEFAULT '[]'   -- JSON array of extracted keywords
```

Written at `save_project()` time. Old rows have empty strings (gracefully ignored by scoring).

### R2 — `_extract_keywords(text: str) → set[str]`

Pure Python utility (no dependencies):

```python
STOPWORDS = {"a","an","the","with","and","or","for","to","of","in","on","build","create","implement","add","make"}

SYNONYMS = {
    "auth":       {"authentication","login","jwt","oauth","session"},
    "api":        {"rest","endpoint","fastapi","flask","routes","http"},
    "db":         {"database","postgres","sqlite","mysql","sql","orm"},
    "frontend":   {"ui","react","nextjs","vue","svelte","html","css"},
    "queue":      {"worker","celery","task","async","job","broker"},
    "cache":      {"redis","memcache","caching","ttl"},
    "test":       {"pytest","jest","unittest","spec","tdd"},
}

def _extract_keywords(text: str) -> set[str]:
    words = set(re.findall(r'[a-z0-9]+', text.lower()))
    words -= STOPWORDS
    expanded = set(words)
    for canonical, aliases in SYNONYMS.items():
        if words & ({canonical} | aliases):   # any hit → add all
            expanded |= {canonical} | aliases
    return expanded
```

### R3 — Recency factor

```python
def _recency_factor(updated_at: str) -> float:
    age_days = (datetime.utcnow() - datetime.fromisoformat(updated_at)).days
    if age_days <= 1:   return 1.00
    if age_days <= 3:   return 0.85
    if age_days <= 7:   return 0.65
    if age_days <= 14:  return 0.40
    return 0.20
```

### R4 — Scoring & filtering

```python
@dataclass
class ResumeCandidate:
    project_id: str
    description: str
    status: str
    score: float
    tasks_done: int
    tasks_total: int
    spent_usd: float
    updated_at: str

def _score_candidates(new_kw: set[str], rows: list[dict]) -> list[ResumeCandidate]:
    results = []
    for row in rows:
        stored_kw = set(json.loads(row["keywords_json"] or "[]"))
        if not stored_kw:
            continue
        overlap = len(new_kw & stored_kw) / len(new_kw | stored_kw)  # Jaccard
        score   = overlap * _recency_factor(row["updated_at"])
        if score >= 0.35:
            results.append(ResumeCandidate(..., score=score))
    return sorted(results, key=lambda c: c.score, reverse=True)[:3]
```

### R5 — Exact match → auto-resume

```python
def _is_exact_match(desc_a: str, desc_b: str) -> bool:
    return desc_a.strip().lower() == desc_b.strip().lower()
```

If first candidate is exact match → auto-resume, print one line, no prompt.

### R6 — `--new-project` CLI flag

```
--new-project   Skip resume check and always create a fresh project
```

Short alias: `-N`

### R7 — 200ms hard timeout

```python
try:
    candidates = await asyncio.wait_for(
        state_mgr.find_resumable(new_keywords),
        timeout=0.2,
    )
except asyncio.TimeoutError:
    candidates = []
```

---

## CLI User Experience

### Case: No match
```
$ python -m orchestrator --project "Build a GraphQL API" --criteria "..."
(no output — proceeds immediately)
```

### Case: 1 match
```
$ python -m orchestrator --project "Build a FastAPI auth service" --criteria "..."

⚡ Found a resumable project:

  ID:       proj_2026_02_20_fastapi_auth
  Status:   PARTIAL_SUCCESS  (7 / 12 tasks done)
  Budget:   $2.34 of $8.00 used
  Last run: 4 days ago
  Match:    "Build a FastAPI auth service with JWT tokens"

  Resume it? [Y/n]:
```

### Case: Multiple matches
```
⚡ Found 2 resumable projects:

  [1] proj_fastapi_auth     PARTIAL_SUCCESS   (7/12)  $2.34  4 days ago  ██████░░░░ 58%
  [2] proj_api_service      BUDGET_EXHAUSTED  (9/12)  $7.89  8 days ago  ████░░░░░░ 39%

  Resume which? [1 / 2 / n — start fresh]:
```

### Case: Exact match
```
⚡ Identical project found — auto-resuming proj_fastapi_auth...
   (pass --new-project to force a fresh start)
```

---

## Modified Files

| File | Change |
|------|--------|
| `orchestrator/state.py` | Add `project_description`, `keywords_json` columns; `find_resumable()` method |
| `orchestrator/cli.py` | Call `_check_resume()` inside `_async_new_project()`; add `--new-project` flag |

## New Files

| File | Role |
|------|------|
| `orchestrator/resume_detector.py` | `_extract_keywords`, `_recency_factor`, `_score_candidates`, `ResumeCandidate`, `_is_exact_match` |

---

## Testing Strategy

| Test | Description |
|------|-------------|
| `test_extract_keywords` | Stopwords removed, synonyms expanded |
| `test_recency_factor` | Correct decay per age bucket |
| `test_score_candidates` | Jaccard + recency; threshold filter; sort order |
| `test_exact_match` | Case-insensitive detection |
| `test_no_match` | Score < 0.35 → empty list |
| `test_db_migration` | New columns added without breaking existing rows |
| `test_timeout_fallback` | Returns [] within 200ms on slow DB |
| `test_cli_new_project_flag` | `--new-project` skips check entirely |
| `test_cli_single_match_prompt` | Interactive [Y/n] prompt |
| `test_cli_multi_match_prompt` | Numbered list [1/2/n] |
| `test_cli_auto_resume` | Exact match triggers auto-resume |

---

## Implementation Order

1. `resume_detector.py` — pure logic, no dependencies (testable in isolation)
2. `state.py` — schema migration + `find_resumable()` method
3. `cli.py` — wire up `_check_resume()` + `--new-project` flag
4. Tests — full coverage per table above
