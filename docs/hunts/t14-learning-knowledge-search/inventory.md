# T14 — learning/, knowledge/, nexus_search/, pattern_learner/, context_mgmt/, analysis/ — Inventory

Sixth of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`, continuing the
AUTONOMOUS DEFECT-HUNT PROTOCOL V7 across backend files with no individual disposition
recorded in earlier tiers.

## Phase 0 — Scope

71 files by direct `find` count (the plan estimated ~72 via a directory-`wc -l` approximation;
the 1-file gap is immaterial, noted for the record per this hunt's transparency practice
rather than silently accepted): `learning/` (14), `knowledge/` (6), `nexus_search/` (21,
including `agents/`, `optimization/`, `providers/` subpackages), `pattern_learner/` (5),
`context_mgmt/` (7), root `memory_tier.py` (1), `ingest/` (2), `analysis/` (15). Framed by the
plan as "auxiliary intelligence/retrieval subsystems — real but not on the critical execution
path for a single run."

## Phase 1-3 — Survey, candidates, trigger/innocence

A background agent surveyed all 71 files, giving full depth to four pre-identified leads
(two 3-way basename collisions in `analysis/`, a `knowledge/` vs `learning/` same-named-file
pair, `learning/log_config.py`, and `nexus_search/`'s internal collisions) plus an exhaustive
**direct-import test on all 71 modules individually** — the same mechanism that surfaced
several of T12's ARA-chain and T13's `knowledge/` findings, applied here systematically from
the start rather than discovered incidentally. All findings below were independently
re-verified from source (diffs re-run, imports re-executed, callers re-grepped) before any
fix was applied.

## Phase 4 — Fixes (VERIFIED DEFECT)

### C1 — `knowledge/knowledge_base.py`'s shim imported a nonexistent name

**File:** `orchestrator/knowledge/knowledge_base.py` (converted to a wildcard shim).

`from ..knowledge_base import get_knowledge_base, KnowledgeBase, KnowledgeEntry` — root
`knowledge_base.py` defines `KnowledgeType`, `KnowledgeArtifact`, `Pattern`, `KnowledgeBase`,
`get_knowledge_base`, but never `KnowledgeEntry` (very likely a stale name from before a
rename to `KnowledgeArtifact` that this shim was never updated for). Empirically reproduced:
`from orchestrator.knowledge.knowledge_base import KnowledgeBase` raised `ImportError:
cannot import name 'KnowledgeEntry'`. `orchestrator/knowledge/__init__.py` bypasses this
broken shim entirely (imports straight from root), so `import orchestrator.knowledge`
succeeds regardless — but anything importing `orchestrator.knowledge.knowledge_base` directly
would fail. The one real (dead) consumer, `knowledge/docs_generator.py`, is transitively
fixed by this change (verified: now imports cleanly).

**Fix:** converted the enumerated import to a wildcard shim (`from ..knowledge_base import
*`), matching this hunt's dominant convention and eliminating the whole class of
wrong-enumerated-name bugs going forward (a wildcard shim can't go stale the way an explicit
name list can when root renames something).

### C2 — `knowledge/knowledge_graph.py`'s shim imported two nonexistent names

**File:** `orchestrator/knowledge/knowledge_graph.py` (converted to a wildcard shim).

`from ..knowledge_graph import ModelPerformanceGraph, KnowledgeGraph` — root
`knowledge_graph.py` defines `PerformanceKnowledgeGraph` (note the swapped word order) and no
bare `KnowledgeGraph` at all. Empirically reproduced the same `ImportError` shape as C1.
`orchestrator/learning/knowledge_graph.py` was investigated as a possible third instance of
the same name but confirmed genuinely different (relational task-pattern learning graph vs.
root's NetworkX-based semantic model graph) — cleared, not a duplicate.

**Fix:** same wildcard-shim conversion as C1.

### C3 — `learning/federated_learning.py` was an unshimmed duplicate with a stripped import

**File:** `orchestrator/learning/federated_learning.py` (converted to a shim).

Diffed against canonical, live `orchestrator/federated_learning.py`: identical in every line
except one — `from .feedback_loop import CodebaseFingerprint, OutcomeStatus,
ProductionOutcome` was replaced with `# REMOVED: from .feedback_loop import ...`, while
`FederatedLearningOrchestrator.contribute_insight()` still references `OutcomeStatus`
directly. **Trigger:** calling `.contribute_insight()` on this specific copy →
`NameError: name 'OutcomeStatus' is not defined`. Confirmed dead: the only real caller of
federated-learning functionality anywhere (`nash/stable_orchestrator.py:37`) imports the root
module, not this one; this copy is reachable only through `learning/__init__.py`'s own
wildcard export, which nothing outside the package reads.

**Fix:** converted to a `from ..federated_learning import *` shim rather than merely restoring
the stripped import in place — this is a full, otherwise-identical duplicate file, and this
hunt's established, more durable fix for that shape is a shim (a shim cannot re-diverge the
way two independently-maintained copies can, which is exactly what happened here).

### C4 — `learning/transfer_learning.py` was an unshimmed duplicate with a stripped import justified by a false comment

**File:** `orchestrator/learning/transfer_learning.py` (converted to a shim).

Same shape as C3, one difference worth flagging on its own: the stripped import
(`from .meta_orchestrator import (ExecutionArchive, ProjectTrajectory, StrategyProposal,
StrategyType)`) was replaced with the comment `# meta_orchestrator types removed (module does
not exist)` — **that comment is itself false**: `orchestrator/meta_orchestrator.py` exists in
the current tree and defines all four names root still imports and uses. Whoever forked this
file either checked a stale tree or misread the situation, then left a confidently-wrong
comment that would mislead the next person to look at it. `_create_routing_proposal()`/
`_create_budget_proposal()` reference `StrategyType`/`StrategyProposal` directly.
**Trigger:** calling `TransferLearningEngine.apply_pattern()` with a `MODEL_ROUTING` or
`BUDGET_ALLOCATION` pattern on this copy → `NameError`. Confirmed dead: real callers
(`meta/monitoring.py`, `meta_integration.py`, `commands/meta.py`) all resolve to root.

**Fix:** same shim conversion as C3.

### C5 — `performance.py::QueryOptimizer.build_selective_query()` built raw SQL from unvalidated input

**Files:** `orchestrator/performance.py` (fixed), `orchestrator/analysis/performance.py`
(converted to a shim after the fix landed).

`build_selective_query(table, columns, where, order_by, limit)` interpolated every one of its
parameters directly into an f-string SQL query with zero validation:
```python
col_str = ", ".join(columns) if columns else "*"
query = f"SELECT {col_str} FROM {table}"
...
query += f" ORDER BY {order_by}"
```
A `table`, `columns` entry, or `order_by` value containing SQL syntax would be executed
verbatim. **Currently dead**: `grep -rn "QueryOptimizer"` across the whole repository
(including tests) returns only the two class definitions — zero constructors, zero callers,
confirmed independently.

**What makes this a live-vs-dead-copy divergence, not just a dead-code gap:** the sibling
`analysis/performance.py` copy — otherwise byte-identical to root except for the expected
import-depth adjustment — had **already been independently hardened**: an `_ALLOWED_TABLES`
frozenset, an `_validate_identifier()` regex + allowlist check applied to `table`, each
`columns` entry, and `order_by`, plus `int(limit)` coercion. That fix was never backported to
the live root copy, which remained vulnerable. This is the third instance this tier of the
"fix landed on the dead copy of an unshimmed duplicate, never propagated to the live one"
shape — the same root cause as C3/C4, just with a completed security fix on the wrong side
instead of a stripped import.

**Fix:** ported the exact hardening (allowlist, identifier validation, `int()` coercion) from
`analysis/performance.py` into the canonical, live `orchestrator/performance.py`. Verified the
fix blocks a semicolon-injected table name, an unlisted table name, and a comma-injected
column name, while a normal query (`table="tasks", columns=["id","name"],
where={"status":"active"}, order_by="created_at", limit=10`) still builds correctly. Once
root carried the same protection, `analysis/performance.py` became a pure, safe duplicate —
converted to a `from ..performance import *` shim (its only other consumer, its own package's
`__init__.py` wildcard export, is unaffected; `orchestrator.analysis` still imports cleanly).

## Phase 4 — Residual, surveyed but not fixed

### 7 further clean-but-unshimmed duplicate pairs, zero current divergence

`analysis/assumption_gate.py`, `analysis/leaderboard.py`, `analysis/progress.py`,
`analysis/progress_writer.py`, `analysis/progressive_output.py`, `analysis/projections.py`,
`analysis/visualization.py` — each confirmed to differ from its root counterpart *only* by
the expected import-depth adjustment, with root confirmed live (via `application/
project_runner.py`, `entrypoints/cli_dispatch.py`, or `orchestrator/__init__.py`) and the
`analysis/` copy confirmed dead in each case. Also `analysis/cross_project_learning.py`
(both copies dead — `CrossProjectLearning` has zero instantiations anywhere). Not converted
to shims this tier: unlike C3-C5, none of these 8 pairs has actually diverged yet — this
tier's fix budget was spent on the 3 pairs that *had* diverged (C3, C4, C5), which is where
real (if currently dormant) risk already materialized. Recorded here as a standing structural
risk given the demonstrated pattern (3 confirmed divergences in this tier alone, plus 10+
across T1-T13) — a good candidate for a future dedicated cleanup pass, not urgent enough to
force into this tier's scope.

### `orchestrator/leaderboard.py` vs `orchestrator/analysis/leaderboard.py` — latent split-brain singleton

Each file maintains its own independent module-level singleton (`_leaderboard`). The
`analysis/` singleton is on a confirmed live path today (`analysis/pareto_frontier.py` →
`commands/nash.py`'s `nash` CLI subcommand). Root's singleton is reachable only via
`engine_core/outcome_router.py` ← `router_integration.py`, and nothing imports
`router_integration.py` except itself and a lint script — fully dead today. **Net effect: only
one singleton is ever populated currently, so no active split-brain bug exists** — but if
`outcome_router.py`'s evident intent (Nash-stable, production-weighted model scoring) is ever
wired live, its writes would be invisible to `nash`'s cost-quality analysis and vice versa,
since they'd be two distinct Python objects. `[REQUIRES HUMAN REVIEW]` if/when that wiring is
considered — not fixed here since the root side is currently dead.

### Minor, low-priority residuals

- `nexus_search/nexus_client.py:140,147` — two consecutive `except Exception` blocks where the
  second is unreachable dead code (the first already catches everything); both behave
  identically. Cosmetic, zero functional impact.
- `memory_tier.py`'s per-file scan loops (`_touch_memory`, `delete_project_memories`) `continue`
  past a file that fails to load with no skip-counter — the same shape as this hunt's Pattern
  4, but much lower stakes than a validator/scanner (a best-effort cache-touch/delete
  operation, not a security or correctness gate). `[UNK]` — only matters if a COLD-tier file
  is both corrupted and happens to match a deletion target's id.
- 4 of `nexus_search/optimization/`'s 7 modules (`adaptive_depth.py`, `circuit_breaker.py`,
  `llm_classifier.py`, `query_expansion.py`) are fully unwired (absent from the package's own
  `__init__.py` exports, never called by `core.py`). Matches this hunt's already-broadly-
  documented "built but unwired" pattern (T12's application/ findings, T13's
  `frontend_security.py`) — recorded as one more data point, not a fresh discovery, and
  calibrated low priority since `nexus_search` itself degrades gracefully when unavailable.
- `analysis/entity_rls.py` — a complete, unwired SQL row-level-security policy generator for
  *generated applications* (not the orchestrator's own DB). Same family as T13's
  `design/frontend_security.py` — explicitly not claimed as a new discovery.
- `context_mgmt/sources.py::ContextSourceManager`, `context_mgmt/system.py::ContextSystem` —
  defined and exported but no confirmed external instantiation (unlike sibling classes in the
  same package that are genuinely wired). Informational only, header-depth only.

## Phase 4 — Cleared (innocent)

`learning/log_config.py` — confirmed a deliberately tiny, self-contained logger-namespacing
helper (`logging.getLogger(f"orchestrator.learning.{name}")`), not a diverged fork of root's
`log_config.py`. Records still propagate to the root `"orchestrator"` logger's handlers
(including the secrets-masking filter), since Python's logging hierarchy is name-prefix-based
— no observability or secrets-masking bypass. `ide_backend/log_config.py` checked on the same
question, also cleared (its own complete, independent, non-forked config).

`nexus_search/`'s three internal 5-way basename collisions (`config.py`, `core.py`,
`providers/base.py`) — each confirmed to be five genuinely disjoint implementations across
unrelated subsystems (a search config vs. a profiler config vs. a feature-flag config, etc.),
basename coincidence in a ~900-file codebase, not a bug. `nexus_search/` as a whole confirmed
live and reachable (from `engine.py`'s post-analysis step, the ARA reasoning pipeline's
default-enabled `nexus_enabled=True`, `enhancer.py`, `architecture_advisor.py`, and its own
standalone `nexus_cli.py`), with every real call site guarding both the import and the actual
HTTP calls — a genuine, deployable, but soft-optional subsystem (a
`nexus-search-docker-compose.yml` exists at repo root), not vaporware and not orphaned.

`orchestrator/agents/metrics.py`, `orchestrator/meta/performance.py` — confirmed genuinely
different purposes from the `analysis/`/root `metrics.py`/`performance.py` pair despite
sharing a basename (per-agent observability; meta-optimization batch/cache utilities).

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used (Phase 1-3 survey delegated to one background agent run, which
ran an exhaustive direct-import test across all 71 files as part of its own methodology).
`fix_revisions`: all 5 fixes (C1-C5) correct on first pass, RED→GREEN verified on the first
attempt. No survey severity claim required correction this tier — the agent's report already
applied the "check for an existing developer comment before calling something high-impact"
lesson from T13 itself (explicitly checking C4's stale comment and correctly reporting it as
false rather than accepting it at face value).
