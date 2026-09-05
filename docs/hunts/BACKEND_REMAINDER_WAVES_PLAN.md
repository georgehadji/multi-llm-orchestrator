# Backend Remainder Sweep — Waves Plan (T9+)

Continuation of `docs/DEFECT_HUNT_PLAN.md`'s AUTONOMOUS DEFECT-HUNT PROTOCOL V7.
T0–T7 are closed (`docs/hunts/INVENTORY.md`); T8 ("remainder, coverage-ordered")
is in flight against an 8-item residual backlog carried over from T0–T7's own
triage notes. This document defines what comes **after** T8: a prioritized
sequence of further tiers (T9, T10, …) covering backend files no prior tier
has individually named, read, or fixed.

This is a scoping document, not a new protocol. Every tier below still runs
the exact 8-phase runbook in `docs/DEFECT_HUNT_PLAN.md` §4, and is bound by
its §8 termination rules (`hunt_iterations` ≤ 3/tier, `fix_revisions` ≤ 1/fix,
a declared per-tier budget cap). This file only answers: *which files, in
what order, and why.*

## Method — how "not run previously" was measured

**VERIFIED (reproducible):**
```
find orchestrator -name "*.py" -not -path "*/tests/*" | wc -l   → 892
grep -oE '`[a-zA-Z0-9_./]+\.py`' docs/hunts/INVENTORY.md | sort -u | wc -l → 53
```
Diffing the full backend file list against every `.py` path individually
named in `INVENTORY.md` (as fixed, cleared, or explicitly flagged residual
across T0–T7) leaves **845 files with no individual disposition on record**.

**Caveat (this is a floor, not a ceiling):** T6's broad-except survey read
~115 files in detail and T7's subprocess/exec census touched 54 files at the
grep level — but those surveys reported aggregate counts, not itemized file
lists, so some of the 845 may have already been eyeballed without earning a
named entry in the inventory. Treat "845 unexamined" as the conservative,
data-backed lower bound on remaining work, not a claim that all 845 are
untouched by any human or agent attention ever.

**INFERENCE:** grouping 845 files into priority waves by directory/subsystem,
using the same risk ordering the original plan already used for T1–T7 (money
→ trust boundary → persistence → concurrency → resilience → error-handling →
execution surface), generalized to whole subsystems instead of individual
files. Severity within each wave is not yet verified — that's each tier's own
Phase 1–3 job.

## Why waves, not one T9 that reads 845 files

`DEFECT_HUNT_PLAN.md` §8 is explicit that a remainder tier "is explicitly
PARTIAL and never completes." Reading 845 files to T0–T7's depth (full read,
trigger+innocence check, fix, ≥1 real RED→GREEN test, full gate suite) is not
a single-tier task — T0–T7 each spent a full tier on a few thousand lines.
Committing to "read all 845 now" would either blow every budget cap in §8 or
produce shallow, unverifiable claims — exactly what this protocol exists to
prevent. Instead: bounded waves, each independently closeable, each leaving
an honest coverage statement, run sequentially for as long as the session
continues this work.

## Wave order and rationale

| Wave | Subsystems | Files (approx) | Why this order |
|---|---|---|---|
| **T9** | `safety/` (18, incl. `sandbox.py`, `secure_execution.py`), `security/` (5), `plugin/`+`plugins/` (11), `reference_monitor.py`, `red_team.py`, `guardrails.py`, `agent_safety.py`, `input_validation.py`, `gateway/` subpackage (3), `hierarchy.py` | ~40 | Direct continuation of T7's own named gap ("52 files remain unexamined, including `safety/sandbox.py`, `safety/secure_execution.py`") — sandbox/isolation code is the highest-consequence category left (arbitrary-code-execution blast radius) and was explicitly flagged `[UNK]`, not cleared. |
| **T10** | `cost_optimization/` (14, already flagged dead-but-imported by T1's C6), `costing/` remainder (2), `rate_limiter.py`, `token_optimizer.py`, `token_budget.py`, `provisioned_throughput.py` | ~20 | Money-adjacent; T1 explicitly deferred verifying whether `cost_optimization/`'s "imported everywhere, instantiated nowhere" claim still holds and whether any path silently bypasses budget accounting. |
| **T11** | `infrastructure/` (51, `state.py` excepted), `engine_core/` (47, `container.py` excepted), `domain/` (12), `events/`+`unified_events/` (10) | ~130 | The hexagonal-architecture core (driven adapters + Mediator wiring + domain ports + event bus, per CLAUDE.md's own pattern table) has the largest single unexamined footprint and the largest blast radius if a Mediator-wiring or port-contract bug exists — architecture-critical, not yet touched at all beyond `container.py`/`state.py`. |
| **T12** | `application/` (42, `evaluator.py`/`budget_enforcer.py` excepted), `planning/` (3), `routing/` (2), `reasoning/` (6), `agents/`+`agents.py` (16), `supervisor/` (7), `delegation/` (3), `meta/` (7), `nash/` (6) | ~90 | Application-layer orchestration logic (task decomposition, agent coordination, multi-agent delegation) — the layer CLAUDE.md calls out as Strategy/Mediator-adjacent; a defect here changes *what the orchestrator decides to do*, not just how it logs a failure. |
| **T13** | `generators/` (32), `appbuilder/` (5), `codebase/` (7) + `code_executor.py`/`code_validator.py`/`code_post_processor.py`, `design/` (33), `scaffold/` (10), `output/` (4), `quality/` (21) | ~145 | The website/app-generation product surface itself — largest wave, but lower architectural blast radius than T9–T12 (bugs here affect generated-output quality, not orchestrator integrity/security). `generators/website_validator.py`'s false-clean secret scan (T6 residual) lives here and should be picked up as this wave's first candidate. |
| **T14** | `learning/` (14), `knowledge/` (6), `nexus_search/` (21), `pattern_learner/` (5), `context_mgmt/` (7), `memory*` (3), `ingest/` (2), `analysis/` (15) | ~72 | Auxiliary intelligence/retrieval subsystems — real but not on the critical execution path for a single run. |
| **T15** | `integrations/` remainder (11), `vcs/` (7), `connectors/` (3), `ide_backend/` (16), `dashboard_core/` (6), `kanban/` (3), `hitl/` (3), `commands/` (24), `cli*.py` (4), `entrypoints/` (3) | ~78 | External-facing adapters and CLI/dashboard surfaces — driving/driven adapters in hexagonal terms, exercised by users directly but isolated by the ports they sit behind. |
| **T16** | Everything remaining: `operations/` remainder (30), `verification/` (6), `testing/` (4), `tools/` (3), `skills/` (4), `workspace*` (6), `project*` (13), `services/` remainder (5), `crosscutting/` (2), misc root-level (`telemetry.py`, `tracing.py`, `monitoring.py`, `metrics.py`, `logging.py`, `policy_engine.py`, `policy.py`, `policy_dsl.py`, …) | ~75 | Final catch-all — cross-cutting observability/policy modules, lower individual blast radius, swept last per the plan's own "remainder" convention. |

Wave file counts are approximate (directory-level `wc -l`, not yet
individually triaged) and will be corrected by each wave's own Phase 0 delta.

## Per-wave execution (unchanged from the base protocol)

Each wave (T9, T10, …) runs `docs/DEFECT_HUNT_PLAN.md` §4 verbatim:

0. Delta — re-run the census against the tree the prior wave leaves behind.
1. Survey the wave's file set (delegate to a background Explore/general-purpose
   agent per the T6/T7/T8 pattern — read-only, no fixes, no test files).
2. Rank candidates by the plan's severity heuristic (money > security/trust
   boundary > silent-failure > everything else).
3. For each candidate pursued: confirm live reachability, attempt genuine
   innocence (isolating handler? dead code? already covered elsewhere?)
   before calling it a defect.
4. Record every candidate's disposition in a per-wave `inventory.md`
   (fixed / cleared / `[REQUIRES HUMAN REVIEW]` / `[UNK]`) — cleared and
   residual items are recorded, never silently dropped.
5–7. Fix + self-review + ≥1 real RED→GREEN test per VERIFIED DEFECT taken to
   a fix, following the session's TDD discipline (RED confirmed against the
   pre-fix tree, GREEN after).
8. Run the full gate suite (`black`, `ruff`, `lint-imports`,
   `check_root_module_freeze.py`, `check_test_markers.py`, `mypy` on the
   specified core paths, `bandit -lll -r orchestrator/`,
   `pytest tests/ -q -m "unit or integration"`), write the wave's own
   `coverage.md` with an honest PARTIAL statement, append a summary row to
   `docs/hunts/INVENTORY.md`, commit, push, update PR #28.

Standing constraints carried over unchanged: never fabricate testimonials/
credentials/medical claims (n/a to backend, kept for completeness), never
hardcode passwords, a check the tooling cannot measure is reported as
outstanding rather than scored as a pass, TDD without exceptions, no new
root-level modules, `engine.py`/`models.py`'s Unbreakable Rules hold.

## Termination

This plan does not have a "done" state by design — it is explicitly a
continuation of a protocol whose own §8 says the remainder "never completes."
Waves are executed sequentially, one at a time, for as long as this session
continues the work; each wave's coverage.md states plainly what was and was
not verified, and the next wave's Phase 0 re-confirms this document's file
counts are still accurate before proceeding (files move, get deleted, or get
fixed by unrelated work between waves).

## Immediate next step

Begin **T9** (safety/execution surface) — Phase 0/1, delegated to a
background investigation agent, run concurrently with T8's still-open
investigation (disjoint file sets: T8 = 8 named residual candidates from
T0–T7's own triage; T9 = the `safety/`/`security/`/`plugin*`/`gateway/`
subsystem census above). T8 closes independently on its own inventory once
its agent returns.
