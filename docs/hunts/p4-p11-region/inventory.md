# P4–P11 — V4 precision audit, waves 4–11 + cross-wave sweeps

**Base:** `master` @ `f390414`. **Scope:** the 193 files `docs/hunts/v4_waves.tsv` assigns to
P4–P11 (~91k LOC), plus the cross-wave mechanical sweeps described in §Method.

## Method, and the honest limit of this coverage claim

These eight waves were **not** run as eight independent K=8 human-depth reads. The V4 wave
plan's own §8 warns that "11 waves is a floor, not an estimate of effort"; running P4–P11 at
P1/P2 depth is not achievable in one session, and claiming otherwise would be the exact
dishonesty the wave plan's §8 was written to prevent.

What was actually done, and what it does and does not license:

1. **Full mechanical sweep across all 218 P3–P11 files** with seven AST detectors targeting
   the shapes that produced every P1/P2 finding: un-awaited coroutine, `async with` over an
   async factory, unread parameter, unread underscore-local (invisible to ruff F841),
   zero-caller public method, broad-except-without-log, and structurally-unraisable `try`.
   Candidates were then filtered against the wrappers that make a bare coroutine legitimate
   (`create_task`, `gather`, `ensure_future`, …) and each survivor was read in source.
2. **Whole-repo import census** (821 modules), classified into ORPHAN / TEST-ONLY /
   MAIN-ONLY / ENTRYPOINT / SHIM, with dynamic entry-point discovery accounted for.
3. **Whole-repo import execution** — every one of the 821 modules actually imported.
4. **Targeted deep reads** of the highest-priority money, safety and execution files.

**This licenses:** total coverage *within those detector shapes* and *total* coverage of the
import-integrity class (every module was really imported). **It does not license** a
clean-claim for the taxonomy classes V4 would reach by reading each file in full —
injection, resource lifecycle, edge-case arithmetic, and domain-logic errors in the ~91k LOC
of P4–P11 remain **largely unread**, as they did after T22's own "~10,000 of 10,571 lines
remain unread" statement. Treat P4–P11 as *swept*, not as *audited*.

---

## PX-IMPORT1 — 12 modules cannot be imported at all — VERIFIED, CRITICAL, total coverage

Executed, not inferred: `pkgutil.walk_packages` over `orchestrator/`, importing each of
**821 modules**. Twelve raise at import time, so every symbol in them is unreachable by any
caller under any configuration:

| module | LOC | failure |
|---|---|---|
| `engine_core/sagas.py` | 850 | `No module named 'orchestrator.engine_core.unified_events'` |
| `integrations/mcp_server.py` | 653 | `No module named 'orchestrator.integrations.log_config'` |
| `website_generator.py` | 584 | `No module named 'orchestrator.component_registry'` |
| `engine_core/outcome_router.py` | 580 | `No module named 'orchestrator.engine_core.feedback_loop'` |
| `agents/persona_modes.py` | 443 | `No module named 'orchestrator.agents.modes'` |
| `cli_website.py` | 330 | `No module named 'orchestrator.component_registry'` |
| `engine_core/modes.py` | 316 | `No module named 'orchestrator.engine_core.models'` |
| `engine_core/escalation.py` | 291 | `No module named 'orchestrator.engine_core.models'` |
| `router_integration.py` | 254 | `cannot import name 'get_adaptive_router'` |
| `engine_core/evaluation.py` | 250 | `No module named 'orchestrator.engine_core.models'` |
| `integrations/compat.py` | 70 | `No module named 'orchestrator.integrations.dashboard_core'` |
| `persona_modes.py` | 11 | (shim to `agents/persona_modes`) |

**Root cause — one systematic mistake, verified.** Eleven of the twelve are files that were
moved from `orchestrator/X.py` into a subpackage (`engine_core/`, `integrations/`, `agents/`)
during a reorganisation **without their single-dot relative imports being re-pointed**. Inside
`orchestrator/engine_core/sagas.py`, `from .log_config import get_logger` no longer means
`orchestrator.log_config` — it means `orchestrator.engine_core.log_config`, which does not
exist. Every intended target does exist one level up (verified for all eight distinct
targets: `log_config`, `unified_events/core`, `adaptive_router`, `feedback_loop`, `models`,
`dashboard_core/mission_control`, `design/component_registry`, `engine_core/modes`), so the
fix in each case is `.` → `..`.

Two have a different cause and need more than a dot:
- `website_generator.py:17` / `cli_website.py` import `from .component_registry import
  get_registry`; that module now lives at `orchestrator/design/component_registry.py`
  (moved when the component registry was revived). This is the **root** `website_generator.py`,
  not the live `generators/website_generator.py` that P1 audited — the twin was left behind.
- `router_integration.py:24` imports `get_adaptive_router` from `adaptive_router.py`, which
  defines only `ModelState` and `AdaptiveRouter` (verified). The factory function does not
  exist anywhere in the repo — a missing symbol, not a misrouted path.

**The missing gate.** `grep -rln "walk_packages|iter_modules" tests/` returns **nothing**:
there is no test that imports every module. All eleven CI gates can pass with 12 modules
permanently unloadable, because nothing ever tries to load them. This is the single
highest-leverage gap found in this sweep — a ~10-line test closes the entire class and
prevents recurrence.

**Interaction with the census.** These 12 are a subset of the ORPHAN/MAIN-ONLY sets from
P3-ORPHAN0, and the causality runs both ways: a module nothing imports is a module whose
broken import nobody notices, and a module that cannot be imported can never acquire a
caller. This is the mechanism by which 18% of the backend became unreachable.

---

## PX-BUS1 — `get_event_bus()` bound un-awaited at 8 sites in 5 files — VERIFIED, systemic

`unified_events/core.py:1058` declares `async def get_event_bus()`. Eight call sites bind
its return value synchronously, so the name holds a bare coroutine:

| site | code | first use of the bad value |
|---|---|---|
| `projections.py:610, :625` | `bus = event_bus or get_event_bus()` | passed to `ModelPerformanceProjection(bus)` / `BudgetProjection(bus)` |
| `analysis/projections.py:612, :628` | same (twin file) | same |
| `engine_core/sagas.py:392` | `self.event_bus = event_bus or get_event_bus()` | `await self.event_bus.publish(event)` at `:624` → `AttributeError: 'coroutine' object has no attribute 'publish'` |
| `dashboard_core/core.py:171` | `self.event_bus = event_bus or get_event_bus()` | `async for event in self.event_bus.subscribe():` at `:230` → `AttributeError` |
| `cli_nash.py:502, :532` | `bus = get_event_bus()` | `bus.get_event_history(...)` → `AttributeError`, caught by the command's own `except Exception` and printed as `Error: 'coroutine' object has no attribute 'get_event_history'` |

This is the same shape as P1-5 and P2-S2-2/2b, now at eight further sites — evidence that
patching call sites one at a time is not converging. The `or` idiom is what makes it easy to
get wrong: `event_bus or get_event_bus()` reads as a null-coalesce and silently produces a
coroutine only on the fallback branch, so any test that injects a bus passes.

**Refinement of P2-UEB1 (correction to the prior wave's reachability claim).** P2 recorded
that `sagas.py:624`'s events are "silently queued forever" because the bus was never
`start()`ed. With `sagas.py` unable to import at all (PX-IMPORT1) and `self.event_bus` a
coroutine when no bus is injected, saga events do not reach a queue — the call raises before
that. P2's mechanism was right about the bus; its claim that this path was live was not.

## PX-DIAG1 — `ProjectDiagnostics.diagnose()` calls a method that does not exist — VERIFIED, dormant

**File:** `orchestrator/operations/diagnostics.py:412`.

```python
state_mgr = StateManager()
state = state_mgr.load_state(self.project_id)
```

`StateManager` (`infrastructure/state.py:232`) defines `load_project`,
`load_latest_checkpoint` and `load_circuit_breaker_state` — there is **no `load_state`**
(verified by grep over both the canonical module and its `state.py` shim). Every call to
`diagnose()` raises `AttributeError`. The line immediately above it reads
`# FIXED: from ..state import StateManager` — the import was corrected at some point, the
method name was not. The correct call is almost certainly `await
state_mgr.load_project(self.project_id)`, which is *also* an `async def`, so the fix needs an
`await` too.

**Reachability:** `ProjectDiagnostics` and `.diagnose()` have zero callers repo-wide.
Noted for honesty: this file was audited in P2 and modified again in R1-C2 (PR #29) without
this being caught — a deep read of a file is not a guarantee against a defect one line
outside the region under examination.

## PX-ROUTER1 — un-awaited coroutine makes a fallback unreachable — VERIFIED, dormant

**File:** `orchestrator/router_integration.py:131, :202`.

```python
return self.adaptive.preferred_model(healthy, task_type) or healthy[0]
```

`AdaptiveRouter.preferred_model` is `async def` (`adaptive_router.py:163`). The un-awaited
call yields a coroutine, which is always truthy, so `or healthy[0]` is **unreachable** and a
coroutine object is returned where a `Model` is expected. Dormant only because the module
cannot be imported at all (PX-IMPORT1) — fixing the import without also fixing these two
lines would turn a dead module into a live crash.

## Cleared (innocent) — checked, no defect

- `plugin/plugin_isolation.py:402,415,418,442` `process.is_alive()` — this is
  `multiprocessing.Process.is_alive`, a sync stdlib method. The detector matched the
  unrelated `async def is_alive` in `engine_core/health.py`.
- `unified_events/core.py:671` `handler(event)` — every `on_*` handler on the projections
  defined in that file is a plain `def`; the `async def` handlers belong to the separate
  `Projection` hierarchy in `projections.py`, which this `apply()` never dispatches to.
- `integrations/mcp_server.py:119,298` / `mcp_server.py:110,264` `@self.server.list_tools()`
  — MCP SDK decorator factories, not calls to this module's own async methods.
- `infrastructure/streaming_resilient.py:358` `gc.collect()`, and the several
  `sys.path.insert` / `list.insert` hits — stdlib name collisions with
  `pattern_store.insert` / `ports.collect`.
- `api_server.py:597,695` `run_fn=lambda: orch.run_project_with_tasks(...)` — the lambda
  defers the coroutine for an awaiting caller; correct by construction.
- `orchestrator/preflight.py` (root) is referenced and live; only its `quality/preflight.py`
  twin is orphaned, so `CLAUDE.md`'s Validation row cites a valid path.
