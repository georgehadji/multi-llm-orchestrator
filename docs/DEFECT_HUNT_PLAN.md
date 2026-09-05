# Sequential Defect-Hunt Plan — Multi-LLM Orchestrator Backend

Applying **AUTONOMOUS DEFECT-HUNT PROTOCOL V7 (PROACTIVE)** across the backend, in bounded tiers.

Status: **DRAFT — not yet executed.** Every `[VF]` below was measured against the working tree
at `92b30f9` (master, post-#27). Nothing here is a defect finding; this is the allocation plan
that decides *where the hunts run and in what order*.

---

## 0. Why this document exists

V7 Phase 0 carries a blocking condition:

> If runtime version or entry point cannot be determined, **OR if no scope can be bounded**
> → emit `[BLOCKED: missing environment/scope data]` and do not proceed.

"The whole backend" does not bound a scope. Measured:

| Measure | Value | Basis |
|---|---|---|
| Python files under `orchestrator/` | **892** | `find orchestrator -name '*.py' \| wc -l` `[VF]` |
| Lines under `orchestrator/` | **259,444** | `find … -exec cat {} + \| wc -l` `[VF]` |
| Root-level modules (depth 1) | **257** (64,818 lines) | frozen baseline = 256 + `__init__` `[VF]` |
| Modules containing `async def` | **405** | grep `[VF]` |
| `gather` / `create_task` / `Lock` sites | **205** | grep `[VF]` |
| `except Exception` / bare `except` sites | **977** | grep `[VF]` |
| `subprocess.` / `eval(` / `exec(` sites | **235** | grep `[VF]` |
| Test modules (`test_*.py`) | **177** | `[VF]` |
| `xfail` / `skip` markers in suite | **34** | `[VF]` |

A single V7 run over 259k lines terminates only by exhausting its budget in the first
subsystem it opens, and then emits a coverage statement admitting it audited ~1% of the
surface. That is a *valid* V7 outcome (PARTIAL), but a wasteful one, because the 1% is
chosen by directory-listing order rather than by risk.

**So the plan's job is decomposition:** turn one unbounded hunt into a *sequence of eight
bounded hunts*, each of which independently satisfies Phase 0, ordered so that the earliest
budget buys the largest risk reduction.

---

## 1. PHASE 0 CENSUS (shared across all tiers)

Instantiate once; each tier declares only its *delta* (scope, budget, taxonomy prune).

```
Language:       Python
Runtime:        3.11.15 in this environment  [VF: `python --version`]
                CI type-checks with --python-version=3.12  [VF: .github/workflows/ci.yml]
                ⚠ DISCREPANCY — see §7.1, this is itself a Tier-0 candidate
Framework(s):   asyncio (stdlib), aiosqlite, FastAPI/uvicorn (api_server), argparse (CLI)
Test framework: pytest 7+ with pytest-asyncio (mode=auto), pytest-cov  [VF: pyproject.toml]
                --strict-markers is ON; 15 registered markers
Entry point(s): orchestrator.cli:main            (console script `orchestrator`, `mllm`)  [VF: resolves]
                orchestrator.cli_dashboard:main  (console script `dashboard`)             [VF: resolves]
                orchestrator/__main__.py → entrypoints/cli_dispatch.py (argparse + dynamic subcommands)
                orchestrator/api_server.py       (HTTP)
                orchestrator/gateway.py          (HTTP API gateway facade)
                orchestrator.pipeline.stages     (setuptools entry-point plugin surface)
Invariants:     Four Unbreakable Rules (CLAUDE.md) + 5 import-linter contracts (.importlinter)
                — enumerated in §6; these are the *documented* invariants V7 Phase 3b defends against
Build/lockfile: pyproject.toml, editable install; no lockfile  [VF]
```

### 1.1 Threat model for THIS system

Ranked by cost of the failure, not by frequency:

1. **Money loss / budget escape** — the orchestrator spends real USD on provider APIs. A budget
   check that fails open bills the user. Two independent mechanisms (`Budget` per-run,
   `BudgetHierarchy` cross-run) that can disagree.
2. **Credential exposure** — 43 modules touch `api_key` `[VF]`. A key reaching a log, a
   telemetry payload, a generated site, or a git commit is unrecoverable.
3. **Silent wrong result** — an orchestrator that reports success on work it did not do is the
   failure mode this codebase has already been bitten by repeatedly (the WF-100 work exists
   because of it). Ranked *above* crashes: a crash is visible.
4. **State corruption / lost work** — `state.db` is the crash-recovery record. A partial write
   or a bad resume destroys a paid-for run.
5. **Resource exhaustion** — 405 async modules, 205 concurrency primitives; leaked tasks,
   unclosed sessions, unreleased semaphores.
6. **Command injection / path traversal** — 235 `subprocess`/`eval`/`exec` sites, plus a website
   generator that writes to disk from LLM-authored content.
7. **Crash** — lowest rank. Loud and diagnosable.

### 1.2 Taxonomy prune

From V7's default catalogue, keep all eight, but **weight** them for this system:

| Class | Weight | Why |
|---|---|---|
| 4. Error & exception paths | **HIGHEST** | 977 broad-except sites; swallowed failure = silent wrong result (threat 3) |
| 2. Resource lifecycle | **HIGH** | 405 async modules; leak-on-error-path is the dominant async bug |
| 3. Concurrency | **HIGH** | 205 primitives; check-then-act on budget is a money bug |
| 8. Contract / dependency | **HIGH** | four provider SDKs normalized behind one adapter; ordering/nullability assumptions |
| 5. Trust boundary / input | **HIGH** | 235 exec sites + LLM output written to disk |
| 6. Type & serialization | MEDIUM | schema drift across provider responses; `None` propagation |
| 7. State machine | MEDIUM | circuit breaker, resume detection, policy enforcement modes |
| 1. Boundary & arithmetic | MEDIUM | cost arithmetic and token accounting only; elsewhere low value |

---

## 2. Strategy selection

Four decompositions were considered. All four are defensible; they differ in what the first
unit of budget buys.

### A. Layer-ordered (hexagonal rings, inside out)
`domain/` → `application/` → `engine_core/` → `infrastructure/` → adapters.

- **Assumption:** defects in inner rings have the largest blast radius, so audit them first.
- **Pro:** maps 1:1 onto the architecture and onto the import-linter contracts; fixes never
  fight the layer rules because you are always inside one ring.
- **Con:** the inner rings are the *smallest and best-defended* part of this codebase —
  `domain/` is 3,213 lines and `models.py` is enforced pure. Early budget is spent where the
  static smell density is lowest.
- **Failure mode:** finishes three tiers before touching a line of code that spends money.

### B. Threat-model-ordered (money → secrets → state → concurrency → …)
- **Assumption:** MRADO — allocate to maximum expected risk reduction per unit budget.
- **Pro:** the first hunt opens `budget.py`/`cost.py`; the first finding, if any, is a money bug.
- **Con:** a threat cuts across rings. A single budget-escape defect may span `models.py`
  (dataclass), `cost.py` (hierarchy), and `engine.py` (the check site) — colliding with V7's
  "≤ 1 function modified" rule on almost every fix.
- **Failure mode:** a high rate of `[CONSTRAINT-FORCED ESCALATION]`.

### C. Coverage-gap-ordered (least-tested × highest blast radius first)
- **Assumption:** untested code is where defects survive.
- **Pro:** purely empirical, no judgement needed, and the repo has `fail_under = 0` `[VF]` so
  the gaps are real and unpoliced.
- **Con:** coverage measures *attention*, not risk. A 0%-covered dead module scores maximum
  and yields nothing. Also: 892 files means the ranking has a long tail of noise.
- **Failure mode:** budget spent proving that unreachable code is unreachable.

### D. Entry-point-reachability-ordered (BFS from `cli:main`, `api_server`, `gateway`)
- **Assumption:** V7 caps priority on DEAD/UNKNOWN reachability, so start from what is provably
  reachable.
- **Pro:** every region enters Phase 1 with `REACHABLE from [entry] via [path]` already
  established — the single most expensive field to fill in honestly.
- **Con:** misses the setuptools `orchestrator.pipeline.stages` plugin surface and anything
  reached only by dynamic dispatch (`discover_command_modules`). Those are exactly the paths
  where a previous CI repair found *three dead class names* — dynamic reachability is a known
  weak spot in this repo `[VF: commit 54a5a4e]`.

### Recommendation — hybrid, and why

**Order by B. Filter by D. Constrain fixes by A. Break ties by C.**

- **B sets the tier sequence** because the threat model is the only ranking that knows a dollar
  from a stack trace.
- **D is applied inside each tier** as a Phase 1 gate: a region enters the hunt queue only with
  its reachability field filled from a real entry point. Regions reachable only dynamically get
  `UNKNOWN` and are deprioritized-not-discarded, per V7 1b.
- **A binds Phase 5**: the fix must land in the ring that owns the logic. If the causal fix
  belongs in `engine.py`, it does *not* go in `engine.py` — see §6.2.
- **C orders regions within a tier** where B and D tie.

The one thing the hybrid must accept up front: **B guarantees cross-ring defects**, so
`[CONSTRAINT-FORCED ESCALATION]` will be common in T1–T3. That is a policy artifact, not a
risk signal, and V7 already has the tag for it. Budget for it rather than being surprised.

---

## 3. The tier ladder

Eight tiers. Each is one complete V7 run (Phases 0→8) and one pull request.

| # | Tier | Primary surface | Approx. lines | Threat | Budget (candidates) |
|---|---|---|---|---|---|
| **T0** | Census repair | `CLAUDE.md`, CI config, marker/xfail inventory | — | census integrity | 6 |
| **T1** | Money | `budget.py`, `cost.py`, `cost_tracker.py`, `cost_analytics.py`, `cost_optimization/` | ~8k | 1 | 14 |
| **T2** | Credentials & trust boundary | `security/`, `api_clients.py`, `gateway.py`, `api_server.py`, `generators/secrets_*`, the 43 `api_key` modules | ~12k | 2, 6 | 14 |
| **T3** | Persistence & resume | `state.py`, `state_mgmt/`, `checkpoints`, `async_event_store.py` | ~5k | 4 | 14 |
| **T4** | Concurrency & resource lifecycle | `engine.py`, `engine_core/`, the 205 primitive sites | ~10k | 5, 3 | 14 |
| **T5** | Resilience state machines | `resilience.py`, `rate_limiter.py`, `adaptive_router.py` | ~4k | 5, 7 | 10 |
| **T6** | Error-path sweep (class-wide, not region-wide) | the 977 broad-except sites, ranked | cross-cutting | 3 | 16 |
| **T7** | Execution & filesystem surface | the 235 `subprocess`/`eval`/`exec` sites, `generators/` disk writes | cross-cutting | 6 | 12 |
| **T8** | Remainder, coverage-ordered | everything untriaged | ~200k | — | PARTIAL by design |

Totals: **100 candidates across T0–T7**. T8 is explicitly open-ended and expected to terminate
on budget, emitting a PARTIAL coverage statement — which V7 §8 declares a valid honest result.

### 3.1 Measured coverage per tier `[VF]`

Run: `pytest tests/ -m "unit or integration" --cov=orchestrator`, 850 files, at `92b30f9`.

| Tier | Statements | Covered | Read |
|---|---|---|---|
| T1 money | 2,155 | **38.7%** | best-defended of the high-threat tiers, still 6 in 10 statements unexercised |
| T2 credentials | 1,353 | **23.0%** | second-least covered; threat rank 2 |
| T3 persistence | 1,918 | **19.0%** | **least covered in the whole ladder**, and it owns crash recovery |
| T4 concurrency | 4,192 | 40.2% | headline number flatters it — coverage counts lines, not interleavings |
| T5 resilience | 440 | 54.5% | small and comparatively well-tested |
| T7 generators | 6,868 | 50.6% | lifted by the WF-100 and template work |
| **Backend total** | **85,721** | **28.7%** | — |

Two adjustments follow from this, and they are the only places measurement overrode judgement:

1. **T3's budget rises from 12 to 14 candidates.** It is the least-covered tier (19.0%) *and*
   it owns the artifact that makes a crashed run recoverable. Lowest coverage on the fourth
   threat is a worse combination than moderate coverage on the first.
2. **T4's 40.2% is treated as `[UNK]`, not as reassurance.** Line coverage cannot exercise an
   interleaving; a fully line-covered `asyncio.gather` says nothing about a race. T4 keeps its
   full budget regardless of the number.

Coverage did **not** reorder the tiers. It is a tie-break input (§2, strategy C) and the gate
is off (`fail_under = 0`, §7.4), so it measures attention, not correctness.

### 3.2 Why this order and not another

- **T0 first** because V7 Phase 0 consumes documented invariants, and this repo's documented
  invariants are *already known to contain a falsehood* — see §7.2. A census built on a false
  invariant produces false innocence defenses in Phase 3b, which is the worst possible error
  under this protocol (a real defect cleared as innocent).
- **T1 before T2** because a budget escape bills the user on every run, whereas a credential
  leak requires a second condition (a log reaching somewhere).
- **T6 after T1–T5** because a broad-except sweep over 977 sites is only tractable once the
  high-value regions are already triaged; most of the 977 will be legitimate
  (`# noqa: BLE001 — one site must not kill the batch` is a real pattern here `[VF]`), and
  distinguishing them needs the region context the earlier tiers build.
- **T8 last and unbounded** because it is the honest home for "we did not get there".

---

## 4. Per-tier runbook

Identical for every tier. One pass, no skipping, no proceeding past a blocked phase.

### Step 1 — Phase 0 delta
Declare only what changes: in-scope file list (explicit, not a glob), out-of-scope with reason,
`budget_spent = 0`, candidate cap from §3, taxonomy prune for this tier.
Re-verify the shared census is still true (the tree moves between tiers).

### Step 2 — Phase 1 surface map
For every in-scope module, emit the `R[N]` block. **Reachability is filled from a real entry
point or it is `UNKNOWN`** — no guessing. Practical method:
- static: `grep` the import graph inward from `cli_dispatch.py` / `api_server.py` / `gateway.py`;
- dynamic: `discover_command_modules()` and the setuptools `orchestrator.pipeline.stages`
  group must be enumerated explicitly, because static import-following misses both `[VF]`.
Then ≥3 atomic assertions about the map itself, each tagged.

### Step 3 — Phase 2 candidates
Generate to the tier's cap. Every candidate names the violated property and a file:line.
Rejected on sight: "this looks fragile", "consider adding validation".

### Step 4 — Phase 3 trigger + innocence
Both halves, always. Notes specific to this repo:
- **No live provider calls.** Trigger tests must not hit OpenAI/Anthropic/Google/DeepSeek/
  OpenRouter — CI has no keys and this sandbox blocks `openrouter.ai` `[VF: 2 pre-existing
  test failures]`. Use a **fake transport at the adapter seam** (`UnifiedClient`'s injected
  client), never a mock of the unit under test — V7 3a forbids mocking away the mechanism, and
  a transport fake does not.
- **Async triggers** run under `pytest-asyncio` (mode=auto, already configured `[VF]`).
- **Concurrency candidates** need the N≥100 harness and report a rate, tagged STATISTICAL.
  Budget ~10× the wall-clock of a deterministic candidate; do not let one race eat a tier.
- **Innocence attempt must consult the import-linter contracts.** "This layer cannot reach that
  adapter" is a *machine-checked* innocence defense here, and it is the strongest kind.

### Step 5 — Phase 4 inventory
The table. Cleared candidates are recorded, not deleted — this repo has re-litigated the same
false alarms before, and the record is what stops it.

### Step 6 — Phases 5–7 fix, self-review, tests
Fix constraints from V7 **plus** the architecture constraints in §6. Then the ≥3 tests.
The proof-of-defect test **is** the repo's mandated RED test — V7 Phase 3a and CLAUDE.md's
TDD rule are the same step, written down twice. Do not write it twice.

### Step 7 — Phase 8 verdict + gates
Before the tier closes, all of these must pass — they are the repo's actual CI gates `[VF]`:

```bash
black --line-length=100 --check orchestrator/ tests/
ruff check orchestrator/
lint-imports                                   # 5 contracts, all KEPT
python scripts/check_root_module_freeze.py     # 256 modules, no additions
python scripts/check_test_markers.py
mypy orchestrator/domain/ orchestrator/application/ orchestrator/engine_core/container.py \
     --ignore-missing-imports --no-strict-optional --python-version=3.12 --follow-imports=silent
bandit -lll -r orchestrator/
pytest tests/ -q -m "unit or integration"
```

Then the mandatory coverage & residual-risk statement, scoped to *this tier's* surface.

### Step 8 — Deliver
One branch, one PR per tier, named `hunt/tN-<tier>`. PR body = the Phase 4 inventory + the
Phase 8 coverage statement verbatim. Merge before opening the next tier, so each hunt's Phase 0
census is taken against a tree that already contains the previous tier's fixes.

---

## 5. Deliverables

Per tier, committed:

```
docs/hunts/tN-<tier>/inventory.md    Phase 4 table + per-candidate D[N] blocks
docs/hunts/tN-<tier>/coverage.md     Phase 8 coverage & residual-risk statement
tests/unit/test_hunt_tN_*.py         proof-of-defect + boundary + no-regression tests
<the fixes themselves>               minimal diffs, in the owning ring
```

Plus one running file, `docs/hunts/INVENTORY.md`, appended per tier: every VERIFIED DEFECT,
every CLEARED candidate, and the accumulating residual-UNKNOWN set. The cleared list is the
part that compounds — it is what prevents tier N+3 from re-raising tier N's false alarm.

---

## 6. Architecture constraints binding the protocol

These **override** V7 where they conflict. They are the project's rules, not the protocol's.

### 6.1 The five import-linter contracts `[VF: .importlinter]`
1. `domain` / `models` / `exceptions` must not import `infrastructure`, `application`,
   `engine_core`, `engine`.
2. `application` must not import `infrastructure`.
3. `application` must not import `engine`.
4. `engine_core` pipeline modules (`pipeline`, `pipeline_executor`, `pipeline_runner`,
   `project_planner`, `state_coordinator`, `stages`) must not import `infrastructure`.
   `container` is exempt — it is the composition root.
5. Root modules must not import `infrastructure` (documented shims excepted).

A fix that would break a contract is not a fix. `lint-imports` runs in Phase 6 vector 4.

### 6.2 The Four Unbreakable Rules `[VF: CLAUDE.md]` — and where they collide with V7

| Rule | Collision with V7 | Resolution |
|---|---|---|
| `engine.py` = Mediator; new logic goes in a service module | V7 wants the ≤1-function fix at the causal site. If the causal site is `engine.py`, the minimal fix *adds logic to the Mediator*. | The fix goes to the owning service; tag `[CONSTRAINT-FORCED ESCALATION]` and say it is architectural, not risk. |
| `models.py` = pure data, no I/O, no asyncio, no behavior | A validation fix at a dataclass is the natural minimal fix, and it is forbidden here. | Reclassify as a **caller/contract defect** per V7 3b's own rule ("a defect that only fires when a documented invariant is violated is a defect of the caller"). Fix at the caller. |
| TDD without exceptions | None — V7 Phase 3a *is* the RED test. | Write it once. |
| No new root-level `orchestrator/*.py` | A fix needing a new module cannot create one at depth 1. | New code goes in the owning subpackage. `check_root_module_freeze.py` enforces it (256 baseline). |

### 6.3 The dual-budget rule
`Budget` (per-run, `models.py`) and `BudgetHierarchy` (cross-run, `cost.py`) are **independent
and simultaneously active** `[VF: CLAUDE.md]`. In T1, a candidate that conflates them is a
protocol error, not a finding. State this in T1's Phase 0 delta so the innocence attempt has it.

---

## 7. Known collisions, discrepancies, and pre-registered innocents

Recording these now prevents budget being spent rediscovering them.

### 7.1 Runtime version discrepancy `[VF]` — candidate, not yet a defect
This environment runs **Python 3.11.15**; CI's mypy gate passes `--python-version=3.12`, and
`docs`/tooling reference 3.12. V7's Phase 5 output template also assumes 3.12.
**Open question `[UNK]`:** does the CI *test* job run 3.11, 3.12, or both, and does any module
use 3.12-only syntax that 3.11 accepts only by luck? Resolve in T0 — it changes what "the
trigger fired" means for every subsequent tier.

### 7.2 A documented invariant that is false `[VF]`
`CLAUDE.md` states, twice (Testing Strategy, and Known Limitations):

> Pre-existing failures in `tests/stress_test.py` (S2, S6, S7) — documented, not blocking.

**`tests/stress_test.py` does not exist**, and no file matching `*stress*` exists under
`tests/` `[VF: ls, find]`. This is exactly the class of census corruption V7 Phase 0 is
supposed to catch: a *documented* invariant that Phase 3b would accept as a valid innocence
defense. **T0 must resolve it** — either the file was deleted and the doc is stale, or the
tests were never committed. Until resolved, no tier may cite CLAUDE.md's Known Limitations as
an innocence defense.

### 7.3 Pre-registered known-innocent — do not spend budget here
- `tests/unit/test_openrouter_model_audit.py::test_audit_against_live_catalogue_is_clean`
- `tests/unit/test_openrouter_model_audit.py::test_runtime_only_ids_resolve_via_endpoints`

Both fail with `Tunnel connection failed: 403 Forbidden` `[VF]`. **Environmental** — the sandbox
blocks `openrouter.ai`. Not a defect of this code. Any tier observing them re-classifies
immediately and moves on.

### 7.4 The coverage gate is off `[VF]`
`fail_under = 0` in `pyproject.toml`, described in CLAUDE.md as "temporarily relaxed". Coverage
therefore measures attention, not policy. It is a legitimate *tie-break* input (strategy C) and
an illegitimate *risk* input. Do not treat a 0%-covered module as a defect.

### 7.5 The 977 broad excepts are not 977 defects
This codebase uses `except Exception` deliberately at isolation boundaries, with a written
reason (`# noqa: BLE001 — one site must not kill the batch`). T6 ranks them by whether the
handler **loses information** (swallows and continues with a wrong value) versus **isolates**
(records the failure as an outcome). Only the first class is a candidate.

---

## 8. Termination, counters, and what "done" means

Three independent counters, per V7 §8:

- `hunt_iterations` — cap **3 per region tier**; exhausting advances the hunt queue.
- `fix_revisions` — cap **1**; exceeding → `[REQUIRES HUMAN REVIEW]`.
- `budget_spent` — the per-tier candidate cap from §3. Reaching it stops that tier and emits
  its coverage statement.

**The whole programme terminates when T0–T7 have each emitted a coverage statement.** T8 is
explicitly PARTIAL and never "completes".

**The clean claim this programme is permitted to make, and no more:**

> Regions [R…] across tiers T0–T7 were audited for taxonomy classes [1–8 as weighted in §1.2].
> Within that surface × class product, no VERIFIED defect remains unfixed.

It may **not** claim the backend is defect-free. 259,444 lines minus the ~39k in T0–T7 leaves
roughly **85% of the backend `[UNK]`** — not `[VF]`-correct, simply unexamined. Per V7's
epistemic honesty clause, absence of found defects over unaudited surface is `[UNK]`.

---

## 9. Execution order summary

```
T0 census repair       → resolve §7.1 and §7.2 first; everything downstream depends on it
T1 money               → budget escape, double-charge, hierarchy/per-run disagreement
T2 credentials         → key in a log / payload / generated artifact / commit
T3 persistence         → partial write, bad resume, lost paid work
T4 concurrency         → leaked task, unreleased semaphore, check-then-act on shared state
T5 resilience          → circuit-breaker and rate-limiter state machines
T6 error paths         → the information-losing subset of the 977
T7 exec & filesystem   → the 235 sites + LLM-authored content written to disk
T8 remainder           → coverage-ordered, PARTIAL by design
```

---

## UNCERTAINTY ACKNOWLEDGMENT

- **Finding most likely to be a false positive:** none yet — this plan contains zero findings by
  construction. The measurement most likely to mislead is the **977 broad-except count**: it is
  a raw grep, includes tests and comments, and §7.5 argues most are legitimate. Treat it as an
  upper bound on T6's surface, not as a defect count.
- **Real defect most likely missed:** anything reachable **only** through the setuptools
  `orchestrator.pipeline.stages` plugin group or through `discover_command_modules()`. Static
  import-following does not see either, and this repo has already shipped **three dead class
  names** in exactly that surface `[VF: commit 54a5a4e]`. T4's Phase 1 must enumerate both
  dynamically or it will mark live code DEAD.
- **What requires runtime validation:** every concurrency candidate in T4/T5 (rates, not
  booleans); resume-detection correctness in T3 (the file-mtime heuristic CLAUDE.md flags);
  whether the CI test matrix actually runs 3.11, 3.12, or both (§7.1).
- **What static analysis cannot determine:** async interleaving order, aiosqlite write
  durability under crash, real provider response shapes (four SDKs, no live calls permitted),
  and whether a `except Exception` handler's caller can tolerate the value it substitutes.
- **What additional input would most increase confidence:**
  1. The CI workflow's **test-job Python matrix** — resolves §7.1 and fixes the meaning of
     "trigger fired" for all eight tiers.
  2. A **captured real response payload per provider** (redacted), enabling honest transport
     fakes in T2/T6 without live calls.
  3. Whether `tests/stress_test.py` was **deleted or never committed** (§7.2).
  4. A statement of which budget mechanism is **authoritative** when `Budget` and
     `BudgetHierarchy` disagree — currently `[UNK]`, and T1's most important question.
