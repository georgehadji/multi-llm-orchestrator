# Reachability & Remediation Plan — fixing what P3–P11 and T23–T24 found

**Date:** 2026-09-07 · **Base:** `master` @ `f390414` (post-merge of PR #29)
**Inputs:** `docs/hunts/p3-deep-region/inventory.md`, `docs/hunts/p4-p11-region/inventory.md`,
`docs/hunts/t23-remaining-region-depth/inventory.md`, `docs/hunts/ESCALATION_REGISTER.md`
**Supersedes nothing.** `docs/plans/2026-09-06-outstanding-remediation-plan.md` remains valid;
its 10 still-open Group A/B items are carried into this plan's Phase 4 rather than restated.

## 0. House rules

WORM document, same convention as its predecessor — don't edit in place as items land; track
disposition in commits. Claims are tagged **VERIFIED** / **INFERENCE** / **HYPOTHESIS** /
**UNKNOWN**. Every VERIFIED claim below names a file and line that can be re-checked in one
command, or a script in `§9` that reproduces the measurement.

---

## 1. What the waves actually found, and why it reframes the request

The ask was to *"wire up the dead code."* The evidence says that instruction, executed
literally, would make the system worse. That deserves stating plainly before any plan.

**VERIFIED — four measurements, each independently reproducible:**

| measurement | result |
|---|---|
| Product files unreachable from any live entry point | **179 files / 44,449 LOC — 18% of the backend** |
| Modules that raise on `import` | **12** (850-LOC `sagas.py` down to a 70-LOC shim) |
| Modules shadowed by a same-named package, unimportable by construction | **7** |
| Gate suite status while all of the above is true | **5/5 import-linter contracts KEPT, ruff clean, black clean** |

The last row is the finding that matters most. This repository has eleven CI gates, an
isolated-diff mypy check, 2,812 passing tests, and a defect-hunt programme that has closed 22
tiers and two precision waves — and none of it detects that a twelfth of its own modules
cannot be loaded. The gates all verify *properties of code that runs*. Nothing verifies that
the code **is reachable at all**.

**The two failure modes compound.** A module nothing imports is a module whose broken import
nobody notices; a module that cannot be imported can never acquire a caller. That loop is the
mechanism by which 18% of this backend became unreachable, and it explains why the dead
regions are disproportionately full of defects that have *never executed*:

- `application/task_executor.py:199` omits an `await` on an `async def`, so a coroutine object
  — always truthy — disables both cache layers and gets string-interpolated into the LLM
  prompt (P3-TE3). Its test suite cannot catch it: the fixture passes `MagicMock()`, whose
  return value is truthy in exactly the same way.
- `router_integration.py:131` has the same shape, making an `or` fallback unreachable
  (PX-ROUTER1) — in a module that cannot be imported anyway.
- `operations/diagnostics.py:412` calls `StateManager.load_state()`, a method that does not
  exist on that class (PX-DIAG1).
- `get_event_bus()` is bound un-awaited at **8 sites across 5 files** (PX-BUS1), the same
  shape P1-5 and P2-S2-2 already fixed twice.

**INFERENCE (well-supported):** these defects survive *because* the code is dead. Executing
any of these paths raises immediately — `AttributeError` on a coroutine, `ModuleNotFoundError`
at import. They are not subtle. They are simply never run.

**Therefore:** "wire up the dead code" is not a refactor, it is ~44k LOC of unexecuted code
being switched on. Wiring it as-is converts 12 silent absences into 12 loud crashes, and turns
`safety/guardrails.py` from "no guardrails" into "guardrails that report passing without
checking" (P3-GUARD1/2/4).

**The steelman of wiring it up anyway**, which I take seriously: this code was written on
purpose, it represents real capability (`CLAUDE.md` advertises several of these subsystems as
core), deleting it discards genuine work, and a codebase where 18% is dead is a codebase whose
architecture diagram is fiction. All true. The plan below therefore does not refuse to wire —
it sequences wiring *behind* the gate that makes wiring safe, and forces a per-subsystem
decision instead of a blanket one.

---

## 2. Three approaches

### Approach A — Wire everything up
Restore all 179 unreferenced files to live paths: fix the 12 imports, resolve the 7 shadowing
collisions in favour of the richer implementation, register the 9 unexported agents, wire
`guardrails`, `BatchClient`, `projections`, the leaderboard, and the rest.

*Assumes:* every subsystem was intended to ship and is merely unfinished.
*Cost:* the largest by far. Each subsystem needs its latent defects found and fixed **before**
activation, and none has a test suite that exercises it under real collaborators.
*Risk:* highest. `BatchClient` alone would introduce a 300-second stall on the
evaluation/critique path (P3-BATCH1) and a polling loop that can never succeed (P3-BATCH2).
*Failure mode:* a long, destabilising branch that trades a quiet 18% for a loud one.

### Approach B — Delete everything unreferenced
Remove all 179 files, shrink the backend by 18%, and let `git` be the archive.

*Assumes:* unreferenced ⇒ unwanted.
*Cost:* lowest, and it collapses the audit surface every future wave pays for.
*Risk:* discards genuine unfinished work with no record of intent. Some of these files are
recent (`design/component_registry.py`'s revival left `website_generator.py` behind mid-move,
VERIFIED) — deleting the stranded half destroys work that was 90% done.
*Failure mode:* deleting the wrong half of a duplicate pair, or a subsystem someone is
actively mid-way through.

### Approach C — Gate first, then triage by subsystem ← **recommended**
Land the reachability gate that makes the class impossible to reintroduce; fix the mechanical
breakage that gate exposes; fix the live defects that are real today regardless of any
decision; **then** take one wire-or-delete decision per subsystem, with the gate now
preventing silent regression either way.

*Assumes:* only that the decision is worth making deliberately and cheaply reversibly.
*Cost:* moderate, and front-loaded onto the highest-value work.
*Risk:* lowest — every phase is independently shippable and independently revertable.
*Why it beats A:* it refuses to activate code before its defects are known.
*Why it beats B:* it deletes only what a maintainer has declined to claim, one subsystem at a
time, instead of in a single irreversible sweep.

**Recommendation: C.** The decisive argument is ordering, not scope: A and B both *end* with a
decision about 44k LOC that neither can make safely today, because today there is no gate that
would tell you if you got it wrong. C installs that gate in Phase 0, at a cost of about ten
lines.

---

## 3. The plan

Every phase obeys the Four Unbreakable Rules (`CLAUDE.md`): no new logic in `engine.py`, no
behaviour in `models.py`, TDD RED→GREEN per item, no new root-level modules. Every phase ends
green on the full gate suite: `black --check`, `ruff check`, `bandit -r orchestrator
--severity-level high`, `lint-imports` (5/5), `check_new_root_files.py`, mypy isolated-diff,
full `pytest`. One item per commit, matching the repo's T/P-tier convention.

### Phase 0 — The reachability gate *(no decision needed; do this first)*

**Optimal solution: one test, ~10 lines, walking the package.**

```python
# tests/unit/test_import_integrity.py
def test_every_module_imports():
    import importlib, pkgutil, orchestrator
    failures = {}
    for m in pkgutil.walk_packages(orchestrator.__path__, "orchestrator."):
        try:
            importlib.import_module(m.name)
        except Exception as exc:
            failures[m.name] = f"{type(exc).__name__}: {exc}"
    assert not failures, failures
```

*Why this and not something bigger:* a static import-graph checker would need to model
`TYPE_CHECKING`, conditional imports, and the entry-point indirection `container.py` uses.
Actually importing is the ground truth, costs ~30s, and cannot be fooled. It is also the
cheapest thing that would have caught all 12.

**Land it green, not red** — fix the 12 in Phase 1 first, in the same PR, so the gate never
enters the tree failing. Ship the gate and the fixes together.

**Prerequisite check (VERIFIED):** the walk imports every module, including the ones with
side-effectful bodies. Three currently log at import (`skills`, `ide_orchestrator_server`,
`ide_test`) but none block; if any later acquires a blocking body, the fix is that body, not
the gate.

### Phase 1 — Restore importability *(mechanical, no decision)*

All 12 fixes are mechanical and independently verifiable by the Phase 0 gate.

| fix | files | change |
|---|---|---|
| Sub-package relative imports never re-pointed after a move | 9 | `.` → `..` |
| `component_registry` moved to `design/` | 2 | `.component_registry` → `.design.component_registry` |
| `get_adaptive_router` does not exist | 1 | see below |

**Optimal solution for the nine:** re-point the dot, nothing more. Every intended target
exists one level up — VERIFIED for all eight distinct targets (`log_config`,
`unified_events/core`, `adaptive_router`, `feedback_loop`, `models`,
`dashboard_core/mission_control`, `design/component_registry`, `engine_core/modes`). Contract
compliance checked: none of these targets is `orchestrator.infrastructure`, so contracts 4 and
5 stay KEPT.

**`router_integration.py` is the exception — do not paper over it.** It imports a
`get_adaptive_router` factory that has never existed (`adaptive_router.py` defines only
`ModelState` and `AdaptiveRouter`). Writing the missing factory would *activate* a module that
also contains PX-ROUTER1's un-awaited-coroutine bug at two sites. **Optimal: fix the import
by adding the trivial factory AND fix the two `await`s in the same commit**, or leave the
module out of Phase 1 and delete it in Phase 4. Do not add the factory alone.

### Phase 2 — Live defects, fixed regardless of any decision

These are on code that runs today. They need no product decision and should not wait for one.

**2a. `BudgetHierarchy` reservation leak — HIGH, live money path (P3-COST3).**
Every `run_job()` permanently leaks `spec.budget.max_usd` from the org cap, on both the
success and failure paths, until the hierarchy refuses all work.

*Options considered:* (i) add `job_id`/`team` to `JobSpec`; (ii) make
`BudgetHierarchy` release anonymous reservations; (iii) generate a synthetic job id in
`run_job`.

*Optimal: (ii) then (i), in that order, as two commits.* (ii) is the real bug — `can_afford_job`
increments `_reserved_usd` **unconditionally** (`cost.py:248`) while `charge_job` releases via
`_reservations.pop(job_id)` (`:363`), an asymmetry that is wrong on its own terms whatever the
caller passes. Fixing it stops the leak for every caller, including future ones. (i) then makes
the hierarchy *useful* by letting callers attribute spend, and is purely additive
(`job_id: str = ""`, `team: str = ""` on `policy.py::JobSpec`). (iii) is rejected: a synthetic
id papers over the asymmetry and silently fragments cross-run attribution.

RED test: two `run_job()` calls against an `org_max` that only fits one, asserting the second
is not refused. Fails today for the predicted reason.

**2b. `reset_spend()` is undone by a restart — MEDIUM (P3-COST2).**
`_save_to_db()` only ever `INSERT OR REPLACE`s surviving keys; there is no `DELETE` anywhere in
`cost.py` (VERIFIED), so reset team/job rows are restored on next start.
*Optimal: `DELETE FROM budget_hierarchy WHERE key = ?` for removed keys*, rather than the
tempting `DELETE`-all-then-reinsert, which would lose data if the process dies mid-write.

**2c. `remaining("job")` ignores reservations — LOW (P3-COST1).** One-line symmetry fix;
`self._reservations[key]` already holds the value. Ship with 2b.

**2d. The `get_event_bus()` shape, fixed once instead of eight times (PX-BUS1).**
*Options:* (i) patch all 8 call sites; (ii) add a sync accessor; (iii) make the singleton
getter sync.
*Optimal: (iii), with (i) as the mechanical consequence.* `get_event_bus()` is `async def`
only so it can `await` a lazily-created singleton's `start()`. Splitting it into a sync
`get_event_bus()` returning the instance and an explicit `await bus.start()` at the two real
lifecycle hooks (`engine.py::__aenter__` already calls `start()` — P2-UEB1) removes the trap
permanently. (i) alone leaves the next caller free to make the same mistake, which the
evidence says they will: it has now happened 11 times across four waves.
*Risk:* signature change on a public accessor. Mitigate by keeping an `async` alias for one
release. **This is the one Phase-2 item that touches live architecture — Architectural class,
mandatory human review.**

**2e. `PX-DIAG1` — `StateManager.load_state()` does not exist.** Correct to
`await state_mgr.load_project(...)` (also `async`, so the `await` is part of the fix).

### Phase 3 — Structural cleanup *(one confirmation, then mechanical)*

**3a. Delete the 7 shadowed modules (P3-SHADOW1, register N6).** Five are 7–11-line
deprecation shims whose warnings can *never* fire — they are pure misinformation and their
deletion is behaviour-preserving by construction. The two substantial ones need one line of
confirmation that the package is the intended survivor; the evidence already says yes for
`gateway` (`commands/gateway.py` imports `..gateway.run`).

*Optimal: delete, don't rename.* Renaming `gateway.py` → `gateway_legacy.py` would preserve
477 LOC that no longer has a caller and re-enter it into the audit surface — trading a
structural error for a maintenance cost. **Add a gate** (extend `check_new_root_files.py`)
asserting no `X.py`/`X/` collision, so the class cannot recur.

**3b. Dead duplicate pairs where *both* halves are dead** — `gradual_rollout.py` (713),
`feedback_loop.py` (678), `export_manager.py` (434) and their `operations/` twins, plus
`leaderboard.py`, `projections.py`, `cross_project_learning.py` and theirs.
*Optimal: delete the root twin first* (T17 established the sub-package copy as canonical),
which halves the surface before any wire-or-delete decision is taken on the survivor.

### Phase 4 — Wire or delete, one subsystem at a time *(decisions required)*

With Phase 0's gate in place, either choice is now safe: a wired subsystem stays importable,
and a deleted one cannot silently leave a stranded half.

**Default: delete. Wire by exception.** The reasoning is in §1 — every file kept "for later"
is a file that will fail when wired, and meanwhile costs every future wave. But the default is
rebuttable per subsystem, and these five deserve an explicit answer (full detail and
trade-offs in `ESCALATION_REGISTER.md` N1–N5):

| subsystem | LOC | recommendation | why |
|---|---|---|---|
| `safety/guardrails.py` | 590 | **fix, then wire** | The only one whose *absence* is a real risk. But fix P3-GUARD1/2/4 first — wiring it unchanged gives guardrails that report passing without checking. Declare `psutil` or make the memory check fail *closed*. |
| Multi-agent (`agents/`) | ~1,400 | **decide, then wire narrowly** | If kept: export the 9 implementations and add a default role→agent factory. The coordinator already targets `INVESTIGATOR`, whose implementation is unexported — a 3-line fix that makes the subsystem work. |
| Batch API (`cost_optimization/`) | ~730 | **delete or rewrite — do not wire** | Cannot deliver its advertised 50% saving: hangs 300s below 10 queued requests, and polls a batch id the provider never issued. |
| `projections.py` + twin | ~1,270 | **delete** | Zero callers, both twins, plus the un-awaited-bus bug. Event-sourced read models nothing reads. |
| Leaderboard + `cross_project_learning` + twins | ~2,900 | **delete** | Four files, two duplicate pairs, zero callers on any. |

Then work the 10 carried Group A/B items from the previous plan, unchanged, as their decisions
resolve. **A1 and A2 first** — they are the only Security-class items open, and A1's blast
radius is currently nil (`slack_integration` is unwired), which makes now the cheapest moment
it will ever be to fix.

---

## 4. Sequencing and gates

| Phase | Contents | Gate to start | Class |
|---|---|---|---|
| **0+1** | Reachability gate + 12 import fixes, one PR | none | Standard |
| **2** | Live defects (2a–2e) | none, except 2d | 2a–2c, 2e Standard · **2d Architectural (human review)** |
| **3** | Shadowed modules, dead twins, collision gate | one-line confirmation on `gateway.py`/`agents.py` | Standard |
| **4** | Wire-or-delete per subsystem; then A1, A2, B1–B7, B9 | that subsystem's decision recorded | Architectural / Security (human review) |

Phases 0+1, 2 and 3 need **no product decision** and can ship immediately. That is deliberate:
they are also where the measured value is — the gate, the money-path leak, and the structural
errors.

---

## 5. Where this plan is weak

Stated plainly, per the same discipline the predecessor plan used:

- **P4–P11 were swept, not audited.** VERIFIED total coverage for seven detector shapes and
  for import integrity across all 218 files; the remaining V4 taxonomy classes — injection,
  resource lifecycle, edge-case arithmetic, domain-logic errors — are **largely unread** in
  those ~91k LOC. A full-depth P4–P11 would very likely find more. Do not read this plan's
  silence about a file as a clean claim about it.
- **The census is an import-graph measurement, not a semantic one.** It counts a module as
  live if anything imports it — including a re-export `__init__.py` that no caller uses.
  `application/task_executor.py` is "referenced" by exactly that mechanism while never being
  constructed outside tests. **INFERENCE: the true dead surface is larger than 179 files**, not
  smaller. The 179 is a floor.
- **"Delete by default" is a judgment, not a measurement.** It follows from evidence about
  defect density in dead code, but a maintainer with intent I cannot see may rightly overrule
  it per subsystem. That is why Phase 4 is per-subsystem and not a sweep.
- **UNKNOWN: why the 12 imports broke.** The mechanism is certain (moved files, un-repointed
  dots); whether one reorganisation or several caused it is not established, and it bears on
  whether other classes of move-damage exist that an import test would not catch — signature
  drift, for instance.
- **2d changes a public async signature.** Lowest-risk framing offered (sync accessor + alias),
  but it is the one item here that could break an out-of-tree caller.

---

## 6. Reproducing the measurements

Every number in §1 comes from a script, not an estimate. All three are in this session's
scratchpad and should be committed under `scripts/` if the maintainer wants them as gates:

- **Import execution** (the 12): `pkgutil.walk_packages` + `importlib.import_module` over
  `orchestrator/` — becomes Phase 0's test verbatim.
- **Dead-module census** (the 179): AST import graph over `orchestrator/` + `scripts/` +
  root launchers, `tests/` counted separately, with `pyproject` entry points, the
  `orchestrator.pipeline.stages` group and `container.py::_FALLBACK_ENTRY_POINTS` counted as
  real references, and `docs/` excluded from the operational-reference check.
- **Shadowing collisions** (the 7): for each `orchestrator/X/__init__.py`, test whether
  `orchestrator/X.py` also exists; confirm with `importlib` which one `orchestrator.X`
  resolves to.
