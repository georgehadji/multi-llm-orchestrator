# Category 2: Version Control & State Management

> **Focus:** Checkpoints, versioning, releases, and project state management — ensuring no work is ever lost and every change is traceable.
> **Phases:** 7 | **Est. Days:** 12-17 | **Weight:** 9% of total roadmap

---

## Overview

This category provides the **safety net** for the orchestrator — ensuring every code change is tracked, every state is snapshot-able, and every version is rollback-able. Phases range from manual checkpoints through automatic versioning to managed app releases with semantic versioning.

| # | Phase | Source | Description | Days | Dependencies |
|---|-------|--------|-------------|------|-------------|
| 1 | 1 | Replit | **Checkpoints** — named snapshots (full state + files + AI context), bidirectional rollback, auto-triggers | 2-3 | — |
| 2 | V6 | v0 | **Versions as First-Class** — automatic version on every code change, diff comparison, revert, chain navigation | 3-4 | 1, U3 |
| 3 | N2 | Newly | **Chat-Level Restore Points** — restore project to before any specific prompt was processed | 1-2 | 1 |
| 4 | N5 | Newly | **Auto-Commit Messages** — descriptive git commit messages per task (fast template-based or LLM-summarized diff) | 1 | 1, 4 |
| 5 | R2 | Retool | **App Release Management** — semantic versioning (MAJOR.MINOR.PATCH), draft/published state, diff comparison, AI release notes | 2-3 | V6 |
| 6 | W4 | Bolt.new | **Project vs Site Separation** — dev workspace separate from published version, explicit publish with validation gate | 2-3 | V6 |
| 7 | D5 | Dyad | **Copy Project for Experimentation** — duplicate entire project (codebase + state + versions), merge back selectively | 1-2 | — |

---

## Capability Progression

```
Checkpoints (manual snapshots)
    │
    ├── Auto-Versions (every code change tracked)
    ├── Restore Points (per-prompt granularity)
    ├── Auto-Commits (descriptive git messages)
    │
    ├── Release Management (semantic versioning + draft/publish)
    ├── Project vs Site (dev/published separation)
    │
    └── Copy Project (safe experimentation with merge)
```

## Key Innovations

- **Checkpoints capture AI context** — not just code state, but conversation history for full context restoration
- **Versions as first-class** means every single AI edit is automatically versioned with metadata
- **Release management** brings semantic versioning to generated projects — not just development versions
- **Project vs Site separation** prevents accidental deployment of unfinished code
- **Copy project** enables safe experimentation without affecting the original
