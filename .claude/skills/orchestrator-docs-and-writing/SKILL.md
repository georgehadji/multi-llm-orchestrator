---
name: orchestrator-docs-and-writing
description: Load this skill BEFORE creating or editing any Markdown documentation in this repo, before writing an implementation plan/audit report/ADR, or when asked "which doc do I update for X", "is this doc stale", "why does the docs index say 14 files", or "does CLAUDE.md's debugging guide link work". Covers the docs-of-record hierarchy (CLAUDE.md, README.md, USAGE_GUIDE.md, CAPABILITIES.md, docs/README.md, docs/CODEBASE_MINDMAP.md), the docs/ bloat problem (173 files vs the index's claimed 14), the confirmed-missing docs/debugging/DEBUGGING_GUIDE.md, the ponytail-integration doc/code contradiction, house style (date-stamps, commit citations, no oversell), and templates for plans/audits/ADRs. Symptom keywords: "update the docs", "which file is the source of truth", "docs/README.md is wrong", "stale documentation", "write an ADR", "write an implementation plan", "audit report template", "docs are out of date", "CLAUDE.md says X but code does Y".
---

# Orchestrator Docs and Writing

Owns: the docs-of-record hierarchy, the docs/ bloat problem, house style for
written artifacts, and templates for plans/audits/ADRs. Does **not** own the
incident chronicle (see `orchestrator-failure-archaeology`) or the change-control
gate process (see `orchestrator-change-control`) — this skill tells you *how to
write and where to file* a document; those skills tell you *what happened* and
*what process governs a change*.

All facts below verified 2026-07-08 against the repo at
`E:\Documents\Vibe-Coding\Ai Orchestrator`, branch `feat/response-healing`.
Re-verification commands are in §7.

## 1. Docs-of-record hierarchy

These are the **living documents** — the ones a reader should trust as current
and that must be updated when behavior changes. Everything else in `docs/` is
historical record (§2).

| Doc | Path | Verified state (2026-07-08) | Owns |
|-----|------|------------------------------|------|
| Agent manifest | `CLAUDE.md` (repo root) | Present, 8.8K. Last touched `af357973` (2026-06-23). | Instructions to Claude Code / AI agents working this repo. Editing this file is an **architectural-class change** per `orchestrator-change-control` — it defines the Four Unbreakable Rules and CI gate list agents are expected to obey. |
| Quick start | `README.md` (repo root) | Present, 24.1K. | Install, quick start, feature overview, includes a "ponytail" integration section (see §3 for a doc/code contradiction here). |
| CLI & API reference | `USAGE_GUIDE.md` (repo root) | Present, 60.2K. | Command-line usage, Python API. |
| Feature inventory | `CAPABILITIES.md` (repo root) | Present, 48.5K. | Enumerates shipped features. |
| Docs index | `docs/README.md` | Present, 4.8K, **dated 2026-03-25 — stale**, see §2. | Meant to be the navigation hub for `docs/`. |
| Master architecture reference | `docs/CODEBASE_MINDMAP.md` | Present, 130.3K. | THE architecture reference. There is also a `mindmap` skill (`.claude/skills/mindmap/SKILL.md`) whose entire body is `cat`-ing this exact file — load `mindmap` instead of re-reading it here; don't duplicate its content into new docs. |
| Debugging guide | `docs/debugging/DEBUGGING_GUIDE.md` | **DOES NOT EXIST.** `docs/debugging/` is not a directory. CLAUDE.md line 207 cites it as `[docs/debugging/DEBUGGING_GUIDE.md](docs/debugging/DEBUGGING_GUIDE.md)` — this is a dead link, confirmed by a prior session and re-confirmed here. | Nothing — the gap is real. If you need debugging guidance, use `orchestrator-debugging-playbook` (symptom-to-triage) or `orchestrator-failure-archaeology` (incident history) instead; do not invent content and backfill this path without flagging it as a new doc, and do not silently leave CLAUDE.md pointing at nothing once you notice it. |

**Honest gap:** `docs/debugging/DEBUGGING_GUIDE.md` is cited but absent. Two ways
to resolve it, both legitimate — pick one, don't silently ignore it:
1. Fix the CLAUDE.md link to point at the two skills that now cover this
   ground (`orchestrator-debugging-playbook`, `orchestrator-failure-archaeology`).
2. Actually write `docs/debugging/DEBUGGING_GUIDE.md` if a single evergreen
   doc (not a skill) is genuinely wanted, then update the CLAUDE.md link only
   if the target changes.
Either way, editing CLAUDE.md is an architectural-class change — do not do it
as a drive-by inside an unrelated PR.

## 2. The docs/ bloat problem

`docs/README.md` (dated 2026-03-25) claims:

```
| Category | Files | Lines | Status |
| Total (Optimized) | 14 | ~10,000 | ✅ Complete |
```

Verified actual counts (2026-07-08):

| Location | File count | Nature |
|----------|-----------:|--------|
| `docs/**/*.md` (recursive) | 173 | everything under docs/ |
| `docs/*.md` (top level only) | 68 | plans, audits, guides, mixed together |
| `docs/plans/` | 26 | implementation plans |
| `docs/archive/` | 88 | old status/completion/bug-report docs, several are `.json` not `.md` |
| `docs/categories/` | 12 | numbered category briefs (`01-autonomous-execution.md` … `12-scoping-safety.md`) |
| `docs/adr/` | 1 file (`ADR-001.md`), containing multiple ADR entries (ADR-001, ADR-002, … concatenated) |

The index's "14 files, docs/core, docs/features, docs/production" structure
does not exist on disk — that structure was aspirational/from an earlier
reorg that never fully landed, or was itself never true. **Do not trust
`docs/README.md`'s file listing or counts without re-verifying** (§7 gives the
command). If you touch `docs/README.md`, either fix the count/structure claims
to match reality or explicitly mark the numbers as historical.

### The WORM rule

This repo has two fundamentally different kinds of Markdown in `docs/`, and
conflating them is the root cause of the bloat:

1. **Living docs** — the hierarchy in §1. Small, fixed set. Get edited in
   place when reality changes.
2. **WORM documents (write-once, read-many)** — implementation plans, audit
   reports, architecture-score reports, "COMPLETE"/"SUMMARY" status docs.
   These are a historical record of a point-in-time decision or a
   point-in-time audit result. **Do not edit them after the fact to keep them
   "current"** — that destroys the historical record and is how you get a
   173-file pile where nobody can tell what's still true. If a plan is
   superseded, write a new plan/doc and (optionally) add a one-line "Status:
   superseded by X" note at the top of the old one — don't rewrite its body.

`docs/archive/` exists and is exactly the right pattern (old
STANDUP/COMPLETE/BUG_REPORT docs moved out of the active view). `docs/plans/`
holds plan documents that are mostly still sitting at top-level docs/ too —
if you write a new plan, put it in `docs/plans/`, not loose in `docs/`.
Genuinely stale/superseded top-level docs (multiple duplicate
`ARCHITECTURE_SCORE_IMPROVEMENT_PLAN*.md`, multiple `OPTIMIZATIONS_*COMPLETE.md`)
are candidates for a move to `docs/archive/` — that's a cleanup task, not
something to do silently as a side effect of an unrelated change.

## 3. Known doc-vs-reality contradictions — do not propagate

When a doc contradicts code or git history, **code and history win**, and the
correct action is to fix the doc — not to write new docs that repeat the
stale claim.

### Ponytail "integration" claim

`README.md` (lines ~133–159) states:

> "The orchestrator integrates the **ponytail** extension... You can invoke
> ponytail directly via commands or by using specific keywords in your
> prompts... `python -m orchestrator --project "Build a script to parse CSVs,
> be lazy and use ponytail"`"

Verified reality (2026-07-08):
- `grep -rli "ponytail" orchestrator/` (case-insensitive, source tree) —
  **zero matches** in any `.py` file. The only hit is a stale `.pyc` in
  `orchestrator/__pycache__/`, which does not correspond to any live source
  reference (`orchestrator/persona_modes.py`, the module that pyc shadows,
  defines `Persona` enum values `STRICT/CREATIVE/BALANCED/ANALYTICAL/
  CONVERSATIONAL/EXPERT/HELPFUL` — no `PONYTAIL` member, no ponytail logic).
- Ponytail exists as **(a)** Claude Code tooling only —
  `.claude/skills/ponytail/`, `ponytail-audit/`, `ponytail-help/`,
  `ponytail-review/` — these operate on the *assistant* writing code, not on
  the orchestrator's own runtime.
- **(b)** An unwired prototype at `packages/orchestrator-persona/` (a
  separate `pyproject.toml`, `src/orchestrator_persona/{persona.py,
  persona_modes.py}`). Confirmed: `packages/orchestrator-persona` is **not**
  listed as a dependency in the root `pyproject.toml` (`grep -n persona
  pyproject.toml` → 0 matches) and its module name (`orchestrator_persona`)
  is distinct from the production `orchestrator/persona.py` /
  `orchestrator/persona_modes.py` modules — it was never merged in.

So: passing "be lazy, use ponytail" as project text to
`python -m orchestrator --project "..."` does nothing special — it's just
prompt text the LLM sees, not a recognized flag or mode. **Rule for this
skill's consumers:** never write or imply that ponytail is wired into
orchestrator runtime behavior. If you're asked to document ponytail, describe
it as Claude Code tooling that helps *write* the orchestrator's code, and
flag the README section as needing a correction (this is a `docs:` fix, not
a feature to build on).

## 4. Templates

### 4.1 Implementation plan

Use for any non-trivial (3+ step) feature/refactor before writing code, per
CLAUDE.md's "Plan Mode Default". Representative real plans:
`docs/plans/AGENTIC_SYSTEM_IMPLEMENTATION_PLAN.md`,
`docs/ARCHITECTURAL_REMEDIATION_PLAN.md`. Minimum shape:

```markdown
# <Feature/Change Name> — Implementation Plan

**Date:** YYYY-MM-DD · **Author:** <name/agent> · **Branch:** `<branch>`

## 1. Problem / Motivation
What's broken or missing, with evidence (failing test, incident, user request).

## 2. Scope
In scope / explicitly out of scope.

## 3. Design
Architecture-layer impact (cite `docs/CODEBASE_MINDMAP.md` if touching
engine.py/models.py/container.py — mandatory per orchestrator-architecture-contract).
New modules and where they live (no new root-level orchestrator/*.py files).

## 4. Steps
Numbered, each independently testable. RED→GREEN→REFACTOR per step where code is involved.

## 5. Risks / Rollback
What could break; how to revert.

## 6. Test Plan
Specific test files/markers to add or run.
```

File location: `docs/plans/<NAME>.md` for anything substantial (matches
existing convention). This document is WORM once execution starts — track
progress in commits/PR, not by rewriting the plan.

### 4.2 Audit report

Mirror `docs/implementation_audit_report.md` — the only real audit report
found in this repo that consistently uses this exact shape (verified
2026-07-08, dated 2026-06-24, branch `feat/response-healing`):

```markdown
# <Subject> Audit Report — <Final|vN>

**Date:** YYYY-MM-DD · **Auditor:** <name> · **Branch:** `<branch>` · **Commits:** N · **Tests:** N · **Contracts:** N/5

## 1. Executive Summary
One paragraph: what changed, whether gates pass, one-line verdict.
**Verdict:** APPROVED | BLOCKED | APPROVED WITH FOLLOW-UPS

## 2. Plan Compliance
| Plan Item | Status | Commit |
|-----------|:------:|--------|
| ... | ✅/❌ | `<short-hash>` |

## 3. Architecture Compliance
State file/dependency counts and `lint-imports` contract result (N KEPT, N broken).

## 4. Code Quality
Prose notes on patterns used, DI, error handling.

## 5. Testing
| Suite | Tests | Result |
|-------|:-----:|--------|

## 6. Risks
Explicit risk list or "None" with justification.

## 7. Corrections
Follow-up items still open, or "None".

## 8. Final Verdict — **<VERDICT>**
```

File location: `docs/<name>_audit_report.md` or `docs/archive/` if it's
retrospective on already-merged work. Cite every commit hash; verify with
`git show <hash> --stat` before citing it, don't trust memory.

### 4.3 Incident / chronicle entry

**Owned by `orchestrator-failure-archaeology`** — do not create a parallel
incident log here. If you discover a new incident (root cause found, gate
that fired, silent failure resolved), add an entry there using its existing
entry format (symptom → root cause → evidence (commit hash) → status). Load
that skill directly for the format and existing entries.

### 4.4 ADR (Architecture Decision Record)

`docs/adr/` exists — one file, `ADR-001.md`, which actually contains multiple
sequential ADR entries concatenated (ADR-001, ADR-002, ... separated by `---`)
rather than one file per ADR. Verified format from the real file:

```markdown
# ADR-NNN: <Decision Title>

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded by ADR-MMM

## Context
Why this decision was needed — the pain, tightly-coupled code, etc.

## Decision
What was decided, in concrete terms (module/interface names).

## Consequences
What changes as a result — what's now possible/impossible, what tests exist.
```

Follow the existing convention: append new entries to `docs/adr/ADR-001.md`
separated by `---` rather than creating `ADR-002.md` as a new file, unless a
maintainer has already changed that convention (check `ls docs/adr/` before
assuming — re-verify count with the command in §7).

## 5. House style

- **Date-stamp every volatile fact.** Anything that can drift (file counts,
  test counts, "current" state descriptions, "N contracts pass") needs a
  `YYYY-MM-DD` next to it. This SKILL.md's facts are stamped 2026-07-08 —
  don't copy a fact forward into a new doc without re-verifying and
  re-stamping it.
- **Cite commit hashes for claims about what changed or when**, e.g. "fixed
  in `11deb573`" — not "recently fixed". Verify the hash exists and matches
  the claim with `git show <hash> --stat` before writing it down; don't
  transcribe a hash you haven't checked.
- **No oversell.** Don't write "fully integrated", "production-ready",
  "complete" unless you've verified it against code (see §3 for what happens
  when a doc oversells — the ponytail claim). Prefer precise, falsifiable
  statements: "wired into X at line Y" beats "seamlessly integrated".
  Unverified or aspirational claims must be labeled `UNVERIFIED` or
  `PROPOSED`, never stated as fact.
- **Conventional commit types** for anything touching docs:
  `docs: <description>` — e.g. `docs: fix dead debugging-guide link in
  CLAUDE.md`, `docs: correct docs/README.md file count claim`. Use `docs:`
  even for skill authoring commits under `.claude/skills/`.
- **Tables and checklists over prose** for anything a reader will scan under
  time pressure (this mirrors the authoring brief's own house style — see
  `.claude/skills/_authoring/AUTHORING_BRIEF.md`).
- **Windows vs CI note:** dev machine is Windows 11 (PowerShell primary), CI
  is `ubuntu-latest`. When a doc includes shell commands, note where behavior
  differs (path separators, `lint-imports`/grimp Windows Rust-panic history —
  see `orchestrator-build-and-env`) rather than assuming one shell for both.

## 6. When to update which doc

| Change type | Update |
|-------------|--------|
| New CLI flag, new env var, new public API behavior | `USAGE_GUIDE.md` delta; `README.md` if it's a headline feature |
| New feature end users would look for | `CAPABILITIES.md` |
| New module, new layer, changed dependency direction, anything touching `engine.py`/`models.py`/`container.py`/`.importlinter` | `docs/CODEBASE_MINDMAP.md` (architectural-class change — see `orchestrator-architecture-contract` and `orchestrator-change-control` for the review/process side) |
| New Four-Unbreakable-Rule, new CI gate, new agent-facing instruction | `CLAUDE.md` (architectural-class change, mandatory human review per `orchestrator-change-control`) |
| Root-cause found for a prior bug/incident | append to `orchestrator-failure-archaeology`'s chronicle, not a new standalone doc |
| One-off point-in-time plan or audit | new file under `docs/plans/` or `docs/` per §4, WORM afterward |
| Doc found to contradict code | fix the doc in the **same commit** as the code change if you caused the drift, or a standalone `docs:` commit if you're just correcting stale text |

## 7. When NOT to use this skill

- Need the incident/root-cause history itself, not how to write one up → `orchestrator-failure-archaeology`.
- Need to know if a change requires human review / what gate classification applies → `orchestrator-change-control`.
- Need architecture facts to write into a doc (layering, DI, the Four Rules) → `orchestrator-architecture-contract` and the `mindmap` skill (which loads `docs/CODEBASE_MINDMAP.md` directly — don't hand-copy its content).
- Need the flag/env-var catalog to document → `orchestrator-config-and-flags`.
- Need to reproduce a dev environment before writing setup docs → `orchestrator-build-and-env`.
- Debugging a live symptom (not writing about one) → `orchestrator-debugging-playbook`.
- Need theory/rationale (why routing/eval/caching work the way they do) for a doc → `llm-orchestration-reference`.
- Need to prove a claim with a script/tool before writing it down → `orchestrator-diagnostics-and-tooling`.

## 8. Provenance and maintenance

Re-verify before trusting any count/path claim in this skill — things here
drift fast (that's the whole problem this skill documents):

```bash
# Docs-of-record existence check
ls CLAUDE.md README.md USAGE_GUIDE.md CAPABILITIES.md docs/README.md docs/CODEBASE_MINDMAP.md
ls docs/debugging/DEBUGGING_GUIDE.md 2>&1   # expect: No such file or directory (as of 2026-07-08)

# docs/ file count vs docs/README.md's claimed count
find docs -name "*.md" | wc -l               # was 173 on 2026-07-08
find docs -maxdepth 1 -name "*.md" | wc -l    # was 68 on 2026-07-08
grep -n "Total" docs/README.md                # check the claimed count hasn't been fixed

# ponytail doc/code contradiction re-check
grep -rli "ponytail" orchestrator/ --include="*.py"   # expect: 0 matches
grep -n "persona" pyproject.toml                       # expect: 0 matches (orchestrator-persona not a dependency)

# ADR file structure
ls docs/adr/

# CLAUDE.md debugging-guide link — last touched
git log -1 --format="%ad %h" -- CLAUDE.md
```

If any of these disagree with the numbers in this file, fix this file in the
same PR — don't let a docs-about-docs skill go stale too.
