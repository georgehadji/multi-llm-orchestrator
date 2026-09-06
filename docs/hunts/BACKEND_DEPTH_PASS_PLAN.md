# Backend Depth-Pass Plan — Waves T17–T24

Second continuation of the AUTONOMOUS DEFECT-HUNT PROTOCOL V7 across this
repository's backend. Successor to `docs/DEFECT_HUNT_PLAN.md` (T0–T8) and
`docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` (T9–T16). Drafted 2026-09-06 against
commit `56276c7`.

---

## 1. What "not previously run" actually means now — read this before anything else

The request that produced this plan was to hunt "the parts of the backend the
protocol was not run on previously." **Taken as breadth, that set is now empty**,
and saying otherwise would be false: all 892 backend `.py` files received at
least a Phase-1 purpose-and-reachability pass across T0–T16, and every file was
import-checked. There is no virgin territory left to point at.

**Taken as depth, the remainder is large, and it is precisely documented** —
because every wave's own Phase-8 coverage statement explicitly disclaimed uniform
depth. Quoting the ledger's own disclaimers:

| Tier | Its own recorded depth debt |
|---|---|
| T6 | 18 of ~923 broad-except sites ranked; "~862 other broad-except sites … not touched by this survey" |
| T7 | "does not claim any of the other 52 subprocess/eval/exec-touching files were audited" |
| T11 | "Tier C files (the majority) received purpose+reachability only" |
| T12 | `agents/` role classes, `reasoning/ara_pipelines.py` (class/def inventory only), `nash/`, `meta/` internals "not read start-to-end" |
| T13 | most of `generators/wf100/` (incl. `checks.py`, header-only), `design/` Hallmark subsystem, `scaffold/` templates "read at header/reachability depth only" |
| T14 | `nexus_search/agents/`, `pattern_learner/`, `context_mgmt/` "read at header/reachability depth only" |
| T15 | `ide_backend/` internals (**could not even be imported** — missing optional dep), `dashboard_core/` non-chat views, 4 named `integrations/` files |
| T16 | ~19 `operations/` files, `testing/first_generator.py`, 5 `project_mgmt/`/`workspace/` canonicals |

So this plan is a **depth pass**, not a breadth pass. That distinction is the
whole design, and the honesty clause in §9 depends on it.

---

## 2. The measured remainder

Every number below was measured against the working tree at `56276c7`, not
estimated. The commands are given so any of them can be re-run and falsified.

### R1 — Cross-cutting pattern instances (mechanically enumerable)

The T0–T16 programme confirmed 61 defects. The large majority came from seven
recurring *shapes*, not from region-specific logic errors. Those shapes were
sampled, never exhausted. What remains, repo-wide:

| Pattern | Measured remainder | How measured |
|---|---|---|
| **#1 Duplicate root/subpackage pairs** | **188** same-name pairs → 26 sub-side shims, 97 root-side shims (both resolved), **65 where *both sides carry their own definitions*** — the true divergence candidates | AST: module body with zero `ClassDef`/`FunctionDef` = shim |
| **#4 Silent failure** | **923** `except Exception`/bare-except sites; **77** with an immediate `pass`/`continue` and no logging | `grep -rn` + 1-line lookahead |
| **subprocess/exec surface** | **70** files touching `subprocess`/`os.system`/`eval(`/`exec(`/`create_subprocess`; T7 audited 2, T9 covered the `safety/` subset | `grep -rln` |
| **#3/#6 Wiring gaps** | not yet countable — needs a detector (see T20) | — |

The 65 divergence candidates are the highest-value item in this plan. Pattern #1
alone produced the T5 circuit-breaker HALF_OPEN bug, T13's duplicated false-clean
security scan, T14's un-backported SQL-injection hardening, and T16's four
`operations/` pairs. Several of the 65 show the exact near-identical-fork
signature (equal definition counts on both sides): `design/frontend_security.py`
(24/24), `design/component_library.py` (14/14), `generators/docker_generator.py`
(12/12), `infrastructure/streaming.py` (12/12), `operations/feedback_loop.py`
(11/11), `analysis/leaderboard.py` (9/9).

> **Honesty note:** 65 is a *candidate* count, not a defect count. A pair where
> both sides define things is not automatically a defect — some are legitimately
> independent modules that merely share a name. The wave's job is the pairwise
> diff that decides which.

### R2 — Region depth debt

Regions whose own coverage statement recorded reachability-only treatment:

| Region | Files | Lines | Note |
|---|---:|---:|---|
| `reasoning/` | 6 | 5,450 | incl. `ara_pipelines.py` at 4,288 — def-inventory only |
| `ide_backend/` | 16 | 5,152 | **never importable in this sandbox** (missing `fastapi`) |
| `generators/wf100/` | 8 | 5,121 | incl. `checks.py` at 2,251 — header-only |
| `meta/` | 7 | 3,354 | |
| `nash/` | 6 | 2,885 | |
| `context_mgmt/` | 7 | 2,450 | |
| `agents/` | 15 | 1,998 | individual role classes |
| `testing/first_generator.py` | 1 | 1,903 | body never read |
| `dashboard_core/` | 6 | 1,433 | non-chat views |
| `scaffold/` | 10 | 1,409 | template files |
| `pattern_learner/` | 5 | 992 | |
| `nexus_search/agents/` | 3 | 491 | |
| `operations/` remainder | ~19 | — | per T16's own disclosure |

≈ **110 files / ≈ 33,000 lines** at reachability depth only.

### R3 — Escalation register

**14** `[REQUIRES HUMAN REVIEW]` items accumulated across T0–T16 and never
dispositioned. The largest: the policy system has zero live enforcement on any
entry point while `engine.py::run_job()`'s docstring claims the opposite;
`operations/autonomy_config.py`'s Multi-Mode Selector is fully built and inert;
`integrations/mcp_server.py`'s feature-convergence direction is undecided; the
evaluator injects a fabricated neutral score into aggregation on judge failure.

---

## 3. Approach — three options, and why this plan picks the third

**A. Region depth waves** (mirror T9–T16: one wave per region cluster, read
everything). *Pro:* proven cadence, familiar artifacts. *Con:* worst
expected-yield-per-unit-effort of the three. The evidence from 17 tiers is that
line-by-line reading of large implementation files is not where the defects came
from; deep-reading 4,288 lines of pipeline code to maybe find one logic error is
a poor trade against a mechanical sweep that finds five duplicate divergences in
the same time.

**B. Pattern-exhaustive sweeps only** (waves by defect *shape*, repo-wide, to
completion). *Pro:* directly targets the demonstrated defect distribution;
mechanically checkable; results convert into permanent gates. *Con:* structurally
blind to region-specific logic bugs — a wrong formula inside `ara_pipelines.py`
has no cross-cutting signature and would never surface.

**C. Detector-first, then evidence-directed depth** *(recommended)*. Build small
AST/stdlib detectors for the known patterns, run them repo-wide, fix what they
confirm, **then** spend remaining budget depth-reading the regions the detectors
light up plus the small set of high-blast-radius regions detectors structurally
cannot reach (money, evaluation, state). *Pro:* highest expected defects per unit
effort; turns 17 tiers of manual pattern-recognition into permanent CI gates so
these classes cannot silently return; depth is spent where evidence points rather
than uniformly. *Con:* tooling work lands before the first fix — an acceptable
cost, since the detectors are each ~50 lines of stdlib AST code and the repo
already has the precedent (`scripts/check_config_drift.py`,
`scripts/check_root_module_freeze.py`, `scripts/check_test_markers.py`).

**Recommendation: C.** Waves T17–T20 are detector-driven; T21–T23 are
evidence-directed depth; T24 closes the escalation register.

---

## 4. Wave plan

Each wave is an independent, complete 8-phase V7 run with its own
`docs/hunts/tN-<name>/inventory.md` + `coverage.md`, its own test file
`tests/unit/test_hunt_tN_<name>.py`, its own RED→GREEN verification, its own full
gate-suite run, and one commit. Waves execute strictly sequentially.

### T17 — Duplicate-pair convergence sweep
**Scope:** the 65 both-sides-define candidates from §R1.
**Why first:** highest measured expected yield; the pattern with the strongest
track record across the whole programme; mechanically enumerable, so Phase 1 is
exact rather than sampled.
**Threat model:** a security or correctness fix applied to one copy and never
propagated to the live one (the T14 SQL-injection and T16 asyncio-safety shape).
**Method:** for each pair — establish canonical side via `git log` + caller greps;
`diff` both; classify (identical / cosmetic / **functional divergence**);
converge by shimming the dead side, after backporting to the canonical side any
fix that lives only on the dead side.
**Deliverable beyond fixes:** `scripts/check_duplicate_pairs.py` — fails when a
new both-sides-define pair appears. Wire into the gate suite.
**Exit:** all 65 dispositioned; every functional divergence either fixed or
recorded with a written reason.

### T18 — Silent-failure sweep at scale
**Scope:** 923 broad-except sites, ranked; the 77 immediate-`pass`/`continue`
sites audited exhaustively.
**Why here:** the pattern that produced findings in T6, T8, T9, T13 and T16 — four
independent instances of the same false-clean shape — and the one T6 explicitly
left 862 sites deep.
**Threat model:** a validator/scanner reporting clean because it silently skipped;
money already spent never charged; corrupt state read as empty.
**Ranking:** money-adjacent > security/validation-gate > persistence/state >
config-load (T16 established config-load graceful degradation is *not* the same
severity class — do not re-elevate it).
**Deliverable:** `scripts/check_silent_except.py` restricted to the money/gate
shapes, with a documented allowlist for legitimate graceful degradation.

### T19 — Subprocess & exec argument-construction sweep
**Scope:** the 70 files from §R1, minus the `safety/` subset T9 already covered
and the two T7 audited.
**Threat model:** injection via caller-supplied strings, `cwd`-relative paths
(the exact T7 finding), unvalidated URLs/hosts (the T16 SSRF-guard shape),
shell=True equivalents.
**Note:** `tools/shell_tool.py` is *by design* a shell runner — it is not a defect
and must not be reported as one; the question there is whether anything reaches
it from an untrusted path.

### T20 — Wiring-gap sweep (registered but never read)
**Scope:** repo-wide. CLI flags parsed but never consumed (T1, T10, T15, T16 each
found one), `FeatureFlags` fields never read, config keys silently dropped,
services constructed in the container and never invoked (the policy-engine shape).
**Method:** AST — collect every `add_argument` dest, every `FeatureFlags` field,
every `container.<x>` attribute; cross-reference against read sites.
**Deliverable:** `scripts/check_wiring_gaps.py`. This is the detector that would
have caught the policy-system finding automatically.

### T21 — `ide_backend/` depth (blocked region)
**Scope:** 16 files / 5,152 lines that have **never been importable** in this
environment.
**Prerequisite:** `pip install -e ".[dashboard]"` (or just `fastapi`) — see §8.
Without it this wave cannot start and must not pretend to.
**Why its own wave:** it is the only region in the entire programme where the
blocker was environmental rather than budgetary, so its coverage claim is
currently the weakest in the ledger.

### T22 — Reasoning & generation depth
**Scope:** `reasoning/` (5,450 lines, incl. `ara_pipelines.py` at 4,288) and
`generators/wf100/` (5,121 lines, incl. `checks.py` at 2,251).
**Why:** largest line-count debt, and both sit on money paths (pipeline selection
spends budget; wf100 checks gate output quality).
**Method:** these are too large for uniform reading inside one wave's budget —
prioritize by the Phase-1 taxonomy: scoring/aggregation arithmetic, budget
arithmetic, state transitions between pipeline stages, and any check that can
report pass while having skipped.

### T23 — Remaining region depth
**Scope:** `meta/`, `nash/`, `context_mgmt/`, `agents/`, `scaffold/`,
`pattern_learner/`, `dashboard_core/`, `nexus_search/agents/`,
`testing/first_generator.py`, and the ~19 `operations/` files T16 left shallow.
**Method:** coverage-ordered — reachable-from-a-live-entry-point first; dead code
gets a recorded disposition, not a deep read.

### T24 — Escalation register disposition
**Scope:** the 14 open `[REQUIRES HUMAN REVIEW]` items.
**Not a bug hunt.** For each: restate the finding, state the decision actually
required, give a recommendation with its trade-offs, and mark it
decided / deferred-with-reason / withdrawn. Produces
`docs/hunts/ESCALATION_REGISTER.md`.
**Why last:** several of these (policy enforcement, autonomy wiring, MCP
convergence) are architectural decisions whose right answer may change based on
what T17–T23 find.

---

## 5. How V7's phases map onto a depth/pattern pass

The protocol is unchanged; two phases are *instantiated* differently and this
must be stated in each wave's Phase 0, per V7's own scope-declaration rule:

- **Phase 1 (defect-surface map):** for detector waves (T17–T20) the surface is
  an enumerated *set of pattern instances*, not a file list — so the map is exact
  and the coverage claim is total *within that pattern*. For depth waves
  (T21–T23) the surface is the file list and the coverage claim is partial, ranked
  by reachability × blast radius, exactly as in T9–T16.
- **Phase 3 (trigger + innocence):** unchanged and non-negotiable. A duplicate
  pair is not a defect until the divergence is shown to matter on a reachable
  path — "these two files differ" is a Phase 2 candidate, never a Phase 4 verdict.
  The innocence attempt for a divergence is: *is the diverged side dead?* If yes,
  it is a hygiene finding (fix the drift risk, record LOW severity), not a live
  defect. T16's `deployment_feedback.py` is the worked example.

Everything else — the categorical `[VF]`/`[HYP]`/`[UNK]`/`[FALSE]` tagging, the
three independent counters (`hunt_iterations` cap 3/region, `fix_revisions` cap 1,
`budget_spent`), the mandatory Phase-8 coverage-and-residual-risk statement,
and the prohibition on numeric confidence — carries over verbatim.

---

## 6. Architecture constraints (binding on every wave)

From `CLAUDE.md`, non-negotiable:

1. **`engine.py` is the Mediator** — new logic goes in a service module, never
   into `engine.py`. This directly constrains T20 and T24: if the correct fix for
   the policy-enforcement gap is "call the policy engine from `run_job`," that is
   an architectural change requiring sign-off, not a defect fix.
2. **`models.py` is pure data** — no I/O, no asyncio, no behavior.
3. **TDD without exceptions** — RED before GREEN, every time.
4. **No new root-level modules** — all new code in existing subpackages. New
   detector scripts go in `scripts/`, which is outside the frozen 256-module root
   kernel and therefore permitted.

The V7-vs-architecture tension is already resolved and stays resolved: a fix that
must touch two functions (a shim conversion, a caller+callee import repair) is
tagged `[CONSTRAINT-FORCED ESCALATION]` — a policy artifact, not a risk signal.

---

## 7. Verification protocol (carried forward unchanged)

Per wave, before commit:

```
black --line-length=100 --check --fast <changed files>
ruff check <changed files>
lint-imports                                     # 5 contracts
python scripts/check_root_module_freeze.py       # 256-module baseline
python scripts/check_test_markers.py
mypy orchestrator/domain/ orchestrator/application/ orchestrator/container.py
      # isolated-diff method: stash fixed files, run, unstash, run,
      # normalise line/col via sed, diff both SORTED (raw order is nondeterministic)
bandit -lll -r <changed files>                   # the reported list, not the metrics footer
python -m pytest tests/unit/test_hunt_tN_*.py
python -m pytest tests/ -q -m "unit or integration"
```

**RED→GREEN is literal, not asserted:** `git add` the new test file,
`git stash push` the fixed sources, re-run and confirm each test fails *for the
predicted reason*, `git stash pop`, re-confirm green. A test that passes on both
trees is a no-regression check and must be labelled as such, never counted as
proof of the defect.

Two pre-registered environmental failures
(`tests/unit/test_openrouter_model_audit.py`, sandbox blocks `openrouter.ai`)
are expected and are not regressions.

---

## 8. Prerequisites

- **T21 is blocked** until `fastapi` is present: `pip install -e ".[dashboard]"`.
  Verified absent at plan time (`ModuleNotFoundError: No module named 'fastapi'`).
  If it cannot be installed, T21 does not run and the ledger keeps recording
  `ide_backend/` as unaudited — it must not be silently downgraded to "clean."
- No other wave has an environmental prerequisite.

---

## 9. Termination and honesty clause

The plan completes when T24 closes. It does **not** claim that completing it makes
the backend defect-free.

What it will support claiming, and nothing more:

- For T17–T20: *"Pattern class X was enumerated exhaustively across the backend
  at commit <sha>, N instances triaged, M confirmed and fixed, and a gate now
  prevents silent reintroduction."* That is a genuinely total claim — but total
  over a **pattern**, not over the code.
- For T21–T23: *"Regions R were audited for taxonomy classes C; no VERIFIED
  defect remains unfixed in what was read."* Scoped to audited surface × covered
  classes, per V7's epistemic honesty clause.
- Absence of found defects in the unaudited remainder stays `[UNK]`. It never
  becomes `[VF]` of correctness.

The prior plan's own caveat still applies to everything upstream of this one: the
T9–T16 wave decomposition was an estimate-based prioritization, not a
mathematically verified partition — which is precisely why this depth pass exists.

---

## 10. Immediate next step

**T17, Phase 0** — re-verify the 65 duplicate-pair candidates against the tree at
execution time (files move between waves), then Phase 1: canonical-side
determination for all 65 via `git log` + caller grep, before a single diff is
read for suspicion.
