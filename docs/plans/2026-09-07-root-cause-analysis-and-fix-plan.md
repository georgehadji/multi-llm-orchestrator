# Root-Cause Analysis & Fix Plan — the five problems underneath the symptoms

**Date:** 2026-09-07 · **Base:** `master` @ `f390414` + the Phase 0–3a fixes on
`claude/llm-orchestrator-website-factory-luk61w` (PR #30)
**Method:** every number below comes from a script or a cited file/line, not an impression.
Reproduce with `scripts/reachability_census.py` and the measurements in §7.
**Relationship to prior plans:** `2026-09-06-outstanding-remediation-plan.md` enumerated 16
escalated *items*; `2026-09-07-reachability-and-remediation-plan.md` sequenced the *fixes* for
what P3–P11/T23–T24 found. This document asks the different question — **what causes keep
generating these symptoms** — and is the one to work from when deciding where effort goes.

Tags: **VERIFIED** (measured or read today) · **INFERENCE** · **HYPOTHESIS** · **UNKNOWN**.

---

## 1. Selection criterion

The five below are ranked by **dissolution power**: how many independent, already-observed
defects stop being possible once the cause is removed. A problem that is merely large but
generates nothing else ranks lower than a small one that keeps producing bugs.

Three defect hunts (T0–T22, V4 P1–P2, V4 P3–P11/T23–T24) have now found ~130 verified defects
in this repository. Sorted by cause rather than by symptom, **the overwhelming majority trace
to five origins**, and two of those five are each responsible for a defect shape that has now
been "fixed" three separate times without converging.

---

## 2. RC1 — A flat→package migration was started, never finished, and never gated

**Dissolution power: highest.** Accounts for 11 of the 12 unloadable modules, all 7
module/package collisions, 34 duplicate pairs, and at least 5 previously-fixed defects that
recurred in a twin.

### The evidence (VERIFIED)

Modules were moved from `orchestrator/X.py` into sub-packages (`engine_core/`, `integrations/`,
`generators/`, `quality/`, `operations/`, …). The move left three distinct kinds of wreckage:

| artefact | count | measured how |
|---|---|---|
| Same-name root/sub-package pairs, non-shim | 98 | filename match, shims excluded |
| — effectively identical copies (≥98% similar) | **26** | `difflib.SequenceMatcher` ratio |
| — **same module, diverged** (80–98%) | **8** (2,938 LOC) | same |
| — different modules sharing a filename | 64 | same — **not debt**, see below |
| Modules whose relative imports still point at the old flat location | **11** | actual import, PX-IMPORT1 |
| Modules shadowed by a same-named package, unimportable by construction | **7** | P3-SHADOW1 |

The 8 genuinely-diverged pairs, largest first: `docker_generator.py` (95.5% similar),
`mcp_server.py` (92.7%), `preflight.py` (97.0%), `triggers.py` (92.9%), `progress.py` (94.6%),
`progress_writer.py` (97.2%), `session_lifecycle.py` (97.9%), `remediation.py` (94.9%).

**The measurement trap, stated because it nearly produced a false claim here.** A basename
match is *not* evidence of duplication. `orchestrator/models.py` (1,387 LOC),
`supervisor/models.py` (79) and `nexus_search/models.py` (273) share a filename and are
entirely different modules; so are `product_manager.py` (677 LOC of `RICEScore`/`Feature`
domain models) and `agents/product_manager.py` (55 LOC of `ProductManagerAgent`). Sixty-four
of the 98 pairs are this. **Any remediation that deletes "the root twin" by filename will
destroy unrelated code** — similarity must be measured, not assumed.

### Why it keeps generating defects (INFERENCE, strongly supported)

A diverged pair means every fix must be applied twice, and nothing enforces that. Observed
three times already:

- P1-4 fixed a `NameError` in `infrastructure/streaming.py`; P2-S2-1 found the identical bug
  still present in the root `streaming.py`.
- P1-5 fixed an un-awaited `get_event_bus()` in one twin; P2-S2-2b found it in the other; this
  session found a **third** instance in `streaming.py::StreamingPipeline` that both previous
  fixes had missed.
- `ContentBrief` was removed from `design_system` and only the canonical
  `generators/website_generator.py` was updated; the root twin has been unloadable ever since.

### Root cause

**No invariant states which copy is canonical, and no gate detects drift.** T17 swept duplicate
pairs once, manually. Nothing prevents the next one.

### Safest fix

1. **Gate first, converge second.** Add a similarity-based duplicate gate (`>=80%` same-name
   pairs must be an allowlisted shim or fail), seeded with today's 34 as an explicit,
   shrink-only allowlist — same shape as the shadowing gate already shipped in
   `tests/unit/test_no_module_package_shadowing.py`.
2. **Collapse the 26 identical copies** by replacing the root half with a re-export shim
   (the pattern this repo already uses 78 times) — behaviour-preserving, no decision needed.
3. **Converge the 8 diverged pairs one at a time**, each as its own commit with a diff review.
   These are the only ones needing judgment.
4. **Never delete by filename.** Require a measured similarity ≥80% before treating two files
   as the same module.

### Paradigm/pattern note

The re-export shim is the right tool for step 2 and this repo already applies it well. For
step 3, converge toward the **sub-package copy as canonical** (T17's precedent) and keep the
root name alive only as a shim, so `orchestrator.X` stays a stable public path while the
implementation lives at its architectural layer.

---

## 3. RC2 — Nothing verified that code could be loaded at all

**Dissolution power: very high. Already fixed this session** — recorded because it is the
cause that made RC1's damage invisible for so long, and because the fix generalises.

### The evidence (VERIFIED)

Twelve modules raised on `import` — 850-LOC `engine_core/sagas.py` down to a 70-LOC shim —
while **5/5 import-linter contracts passed, ruff and black were clean, and 2,812 tests
passed**. `grep -rln "walk_packages\|iter_modules" tests/` returned nothing: no test had ever
imported the package.

### Root cause

Every one of the eleven CI gates verifies a **property of code that runs**. None verified
**that the code is reachable**. The two failure modes then compound into a ratchet: a module
nothing imports is a module whose broken import nobody notices, and a module that cannot be
imported can never acquire a caller. That loop is the mechanism by which 179 files / 44,449
LOC (18% of the backend) became unreachable.

### Fix (shipped)

`tests/unit/test_import_integrity.py` walks the package and imports all 821 modules,
distinguishing internal breakage (fails) from a missing optional dependency (ignored) via
`ImportError.name`. Importing for real is the ground truth — a static checker would have to
model `TYPE_CHECKING`, conditional imports and `container.py`'s entry-point indirection, and
would still be a model.

### The generalisable lesson

**Add the cheap gate that would have caught the class, not just the fix for the instance.**
This is the template for RC1's duplicate gate, RC3's await gate, and RC5's wiring gate.

---

## 4. RC3 — The async/sync boundary is invisible at runtime

**Dissolution power: high.** One shape, ≥11 occurrences, "fixed" in three separate waves
without converging.

### The evidence (VERIFIED)

A coroutine object is **truthy**, carries no runtime marker, and satisfies every guard the
codebase uses. So an un-awaited call to an `async def` degrades silently rather than failing:

| defect | site | what the truthiness broke |
|---|---|---|
| P1-5 | `infrastructure/streaming.py` | `self.event_bus` bound to a coroutine |
| P2-S2-2/2b | `streaming.py`, twin | same, in a live class |
| P2-NASH1 | `nash/infrastructure_v2.py` | `async with` over an async factory → guaranteed `TypeError` |
| **PX-BUS1** | **8 sites in 5 files** | `event_bus or get_event_bus()` — the `or` idiom hides it, because any test that injects a bus passes |
| **P3-TE3** | `application/task_executor.py:199` | `not <coroutine>` is False → **both cache layers skipped**, and `<coroutine object …>` string-interpolated into the LLM prompt |
| **PX-ROUTER1** | `router_integration.py:131,202` | `f() or fallback` — fallback unreachable |

`P3-TE3` is the sharpest illustration: a missing `await` did not crash, it silently disabled
caching and corrupted the prompt — for CODE_REVIEW tasks, injecting a coroutine repr directly
after the text *"Do NOT claim the code was not provided."*

### Root cause

Async singleton accessors were written **`async` only to take a lock**, then called from
synchronous constructors that structurally cannot await. `UnifiedEventBus.get_instance()` is
`async` purely for an `asyncio.Lock` around a double-checked singleton whose construction
(`cls()`) is itself synchronous — so the async-ness was never essential, but it made every
sync caller a latent bug.

### Safest fix (shipped for the event bus; generalise)

1. **Give every async singleton a sync accessor** where the async-ness is incidental —
   `get_event_bus_sync()` is done. Non-breaking: the async one keeps working.
2. **Gate the shape** — `test_no_unawaited_get_event_bus` walks every module's AST and fails on
   any call not under an `Await`. Shipped.
3. **Generalise the gate** to any `async def` whose name is called un-awaited outside the
   legitimate wrappers (`create_task`, `gather`, `ensure_future`, …). The detector built for
   the P3–P11 sweep already does this at ~90% precision; promoting it to a gate needs only the
   ambiguity filter it already has.

### Paradigm note

The real design error is using **`async` for mutual exclusion of a pure-construction critical
section**. In a single-threaded event loop there is no await point between the `is None` check
and the assignment, so the lock buys nothing. Prefer a synchronous factory plus an explicit
`await resource.start()` at a lifecycle hook — which is exactly what `engine.py::__aenter__`
already does for this bus (P2-UEB1).

---

## 5. RC4 — The test suite structurally cannot detect RC3

**Dissolution power: high, and it explains RC3's recurrence.** This is the cause *behind* a
cause.

### The evidence (VERIFIED)

`application/task_executor.py` has a dedicated test file, `tests/unit/test_task_executor.py`.
It could not fail on P3-TE3, by construction: its fixtures pass `MagicMock()` collaborators
(`:30-64`), and a `MagicMock`'s `get_dependency_context(...)` returns another **truthy**
`MagicMock` — behaving identically to the truthy coroutine at every branch the bug corrupts.

Repo-wide: **48 test files use `MagicMock`, 50 use `AsyncMock`, and 0 use `autospec` /
`create_autospec`.** Without autospec a mock accepts any attribute and any signature, so it
cannot detect a call to a method that does not exist (PX-DIAG1's `StateManager.load_state`) nor
a sync call to an async method (RC3).

### Root cause

Mocks are hand-shaped rather than derived from the real interface, so the test asserts what the
author believed the collaborator does, not what it does.

### Safest fix

1. **`create_autospec(RealClass, spec_set=True)` for collaborators**, which makes a
   `MagicMock` of an `async def` an `AsyncMock` automatically and rejects unknown attributes.
   Fixes RC3 and PX-DIAG1's class in one move.
2. **Do not mass-rewrite.** Convert opportunistically: any test touching a module a wave
   flags, plus new tests. A 48-file sweep is high-risk, low-yield churn.
3. **Add `-W error::RuntimeWarning` for `coroutine ... was never awaited`** in pytest config —
   Python already emits it, the suite currently discards it. This session's run showed the
   warning firing in `test_hunt_p2_deep.py` and being ignored. Nearly free, catches the class
   at runtime rather than by review.

**Recommend starting with the RuntimeWarning promotion** — it is a config line and would have
caught P1-5, P2-S2-2, P3-TE3 and PX-BUS1 without touching a single test.

---

## 6. RC5 — Defensive coding converts missing wiring into silent degradation

**Dissolution power: high.** The single mechanism behind the entire "wiring gap" family and
the highest-severity live defect found this session.

### The evidence (VERIFIED)

Two idioms, applied for robustness, systematically hide absence:

**`getattr(obj, "field", default)`** — 61 sites. The exemplar is P3-COST3:
`ProjectRunner.run_job` reads `getattr(spec, "job_id", "") or ""`. `policy.py::JobSpec` has
never declared `job_id`, so this silently yields `""` forever. Downstream, `can_afford_job`
reserved unconditionally while `charge_job` released via `_reservations.pop("")` → `0.0`, so
**every job permanently leaked its whole budget estimate from the org cap** until the hierarchy
refused all work — reporting a still-low `_org_spent`, which makes the failure look like the
caller's fault. A plain attribute access would have raised `AttributeError` on the first run.

**Broad `except` around a whole operation** — the same shape produces the wiring family:
`--agent-profile` parsed at 4 sites and applied at none (P1-2); `decomposition_model` ignoring
its argument (P2-M2-3); `TieredModelRouter` methods with zero callers; `BatchClient` polling a
batch id the provider never issued, with the failure swallowed by `except Exception: pass`
(P3-BATCH2).

### Root cause

**Absence is represented by a value that is indistinguishable from a legitimate one.** `""`,
`None`, `0.0` and a truthy `MagicMock` all mean "fine, carry on". Nothing distinguishes
"configured to be empty" from "never wired".

### Safest fix

1. **Make absence loud at trust boundaries.** Where a field is required for correctness, read
   it directly and let `AttributeError` fire, or validate once in `__post_init__` (the pattern
   `JobSpec` already uses for `project_description`). Keep `getattr` for genuinely optional
   fields only.
2. **Fix the asymmetry, not the caller.** P3-COST3's real bug was that `can_afford_job` took a
   reservation it had no key to release — wrong on its own terms whatever the caller passes.
   Shipped: a `_release_reserved()` helper both settlement paths share, and `job_id`/`team`
   added to `JobSpec` so spend can be attributed.
3. **A wiring gate.** Promote `scripts/reachability_census.py` to a test with a shrink-only
   allowlist of the 179 known-unreachable files, so the count can go down but never up. This
   is the RC2 template applied to reachability-in-the-large.

---

## 7. What was measured, and how to re-derive it

| claim | command |
|---|---|
| 179 unreachable files / 44,449 LOC | `python scripts/reachability_census.py` |
| 12 unloadable modules | `pytest tests/unit/test_import_integrity.py` |
| 7 module/package collisions | `pytest tests/unit/test_no_module_package_shadowing.py` |
| 98 same-name pairs → 26 identical / 8 diverged / 64 unrelated | `difflib.SequenceMatcher` over each pair (§2) |
| 48 MagicMock vs 50 AsyncMock vs 0 autospec | `grep -rl` over `tests/` |
| 61 defensive `getattr` sites | `grep -rn "getattr(spec\|getattr(config\|getattr(settings\|getattr(self\._"` |

---

## 8. The plan

Ordered so that each phase makes the next safe. Every phase obeys the Four Unbreakable Rules,
ends green on the full gate suite (`black`, `ruff`, `bandit`, `lint-imports` 5/5,
`check_root_module_freeze`, CI's blocking mypy invocation, full `pytest`), and is one item per
commit.

### Stage A — Gates (no product decision; do first)

Each is cheap, each closes a whole class, and each makes the later stages safe.

| gate | closes | status |
|---|---|---|
| Import integrity | RC2 | **shipped** |
| No module/package shadowing | RC1 (part) | **shipped** |
| No un-awaited `get_event_bus` | RC3 (part) | **shipped** |
| `RuntimeWarning: never awaited` → error in pytest config | RC3 + RC4 | **do next — highest value/effort ratio in this document** |
| Similarity-based duplicate gate, shrink-only allowlist | RC1 | do next |
| Reachability census as a shrink-only gate | RC5 | do next |
| Generalised un-awaited-coroutine AST gate | RC3 | after the above |

### Stage B — Converge the duplicates (RC1)

1. Collapse the **26 identical copies** to re-export shims. Mechanical, behaviour-preserving.
2. Converge the **8 diverged pairs** one commit each, sub-package copy canonical.
3. **Do not touch the 64 same-name-different-module files.** They are not duplicates.
4. Resolve the 2 remaining shadowing collisions (see §9 — one has changed since the decision).

### Stage C — Live correctness (largely shipped)

P3-COST3 / COST2 / COST1, PX-BUS1, PX-DIAG1, PX-ROUTER1, the 12 imports — all landed on PR #30.
Remaining: P3-TE1 (TDD-first generation reports `cost_usd=0.0` for real LLM calls) and P3-TE2
(semantic-cache hit fabricates `score=0.85` and `deterministic_check_passed=True`). Both are on
`TaskExecutor`, which is constructed only in tests — fix before any decision to wire it.

### Stage D — Wire the two subsystems you chose

**`safety/guardrails.py`** — fix before wiring, in this order:
- P3-GUARD1: declare `psutil` (it is in no extra) **or** make the memory check fail closed.
  Returning `passed=True` for a check that did not run is the bug, not the missing package.
- P3-GUARD2: the kill switch returns "not activated" for 5s after any check and does not latch,
  while its sibling `KillSwitch` class in the same file *does* latch. Make them agree.
- P3-GUARD3: move kill-switch files off world-writable `/tmp` to `~/.orchestrator_cache/`, and
  check ownership before honouring one. `check_and_exit()` answers the force file with
  `os._exit(1)`, so today any local user can hard-kill the orchestrator.
- P3-GUARD4: correct the docstring — "budget never exceeded" is detection after the fact, not
  prevention. The real hard limit is `BudgetHierarchy.can_afford_job`.
- **Then** wire it, at `engine.py::__aenter__` — the lifecycle hook that already owns
  `event_bus.start()` and `telemetry_store.drain_queue()`.

**Multi-agent (`agents/`)** — wiring is bounded:
- Export the 9 unreferenced implementations from `agents/__init__.py`.
- Add a default `AgentRole → AgentBase` factory so `AgentOrchestrator` can be constructed with
  a populated dict; it currently takes one by injection and nothing in the product ever calls
  it with a non-empty one.
- **Priority within this:** `_decompose_goal` targets `AgentRole.INVESTIGATOR`, whose
  implementation is one of the unexported nine — so even a correctly-wired caller gets a silent
  "no agent for role" failure today. That is a 3-line fix and the highest-value part.

### Stage E — Delete what you are not wiring

Batch API (~730 LOC), `projections.py` + twin (~1,270), leaderboard and
`cross_project_learning` + twins (~2,900). Each is unreachable, each carries defects that have
never executed.

---

## 9. Two things that changed since the decisions you gave

**`gateway.py` needs no rescue — it is already preserved.** You chose "move their contents into
the packages". For `agents.py` (`TaskChannel`/`AgentPool`) that is right: the content is unique.
But `orchestrator/gateway.py` is **byte-identical** to `orchestrator/integrations/gateway.py`
(`diff` returns nothing), which is importable and not shadowed. Moving it into
`orchestrator/gateway/` would create a *third* copy of the same `APIGateway`. **Recommend
deleting `orchestrator/gateway.py`** — the intent behind your choice (don't lose distinct code)
is already satisfied. Only `agents.py` needs the move, to `agents/pool.py`.

**"Delete the root twin" must be similarity-gated.** Applied by filename it would delete
`product_manager.py` (677 LOC of `RICEScore`/`Feature` domain models, nothing to do with
`agents/product_manager.py`'s 55-LOC `ProductManagerAgent`), plus 63 other unrelated files.
Verified safe under the similarity rule: `gateway.py`, `export_manager.py`,
`advanced_query_processing.py`, `pre_submission_testing.py`, `query_expander.py` — where the
only difference is relative-import depth. `tenancy.py` is also a true duplicate but
`tests/unit/test_hunt_t2_credentials.py:173` deliberately parametrises over **both** paths, so
deleting it requires updating that test in the same commit.

---

## 10. Where this analysis is weak

- **Dissolution power is an argued ranking, not a measurement.** RC1 and RC3 are supported by
  counted recurrences; RC4's causal claim (that mocks are *why* RC3 recurred) is INFERENCE from
  one clearly-established case (P3-TE3), not a survey of all 48 MagicMock files.
- **The similarity thresholds (80% / 98%) are judgment.** They separate the cases cleanly here —
  the gap between the 8 diverged pairs (80–98%) and the 64 unrelated ones (<80%) is wide — but
  a pair near the boundary would need a human read.
- **UNKNOWN: whether one migration or several caused RC1.** The mechanism is certain; the
  history is not. This matters because other move-damage an import test cannot see — signature
  drift, silently divergent behaviour between twins — may still be present.
- **The 179-file dead surface is a floor, not a ceiling.** The census counts a module live if
  anything imports it, including a re-export `__init__.py` no caller uses:
  `application/task_executor.py` is "referenced" by exactly that mechanism while being
  constructed only in tests.
- **P4–P11 were swept, not audited.** Total coverage for seven detector shapes and for import
  integrity; the remaining V4 taxonomy classes are unread across ~91k LOC. Silence about a file
  in this document is not a clean claim about it.
