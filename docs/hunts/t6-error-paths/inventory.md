# T6 — Error-Path Sweep (Broad-Except Info Loss) — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4/§7.5. Budget: 20 candidates surveyed, 3 fixed.

## Phase 0/1/2 — survey method

A background Explore agent triaged all `except Exception`, `except
BaseException`, and bare `except:` sites in `orchestrator/` (excluding
`tests/`), filtering out `# noqa: BLE001`-precleared sites and standard
"log + return failure signal" isolating handlers per the plan's §7.5
framing, then individually read full context around ~85 silent-`pass`
sites and ~30 money/security-adjacent sites, prioritizing files already
touched by T0-T5. Full ranked output (18 candidates) preserved verbatim in
this session's transcript; the 3 pursued to a fix this tier are below. See
"Residual, triaged but not fixed" for the other 15.

## Candidates fixed

### C1 — VERIFIED DEFECT — `application/verbalized_sampling.py::VerbalizedSampler.sample()` silently swallowed a real budget-charge failure

- **Property violated:** a broad except around a state-mutating call for
  money already spent must not be silent — threat class "return value
  indistinguishable from success" / "silent pass with no logging."
- **Location:** `orchestrator/application/verbalized_sampling.py:138-140`
  (pre-fix).
- **Finding:** after a real LLM call succeeds and incurs real cost
  (`resp` is non-`None`), `await self._budget.charge(cost, ...)` was
  wrapped in `except Exception: pass` — a genuine `charge()` failure (e.g.
  a malformed `cost_usd` causing `spent_usd += None` to raise `TypeError`,
  or any budget-backend error) left no trace anywhere: no log line, no
  distinguishing signal from a successful charge. The dollar amount is
  already spent (the LLM call happened) but silently never gets counted
  against the caller's budget.
- **Reachability:** live — `VerbalizedSampler.sample()` is the file's
  primary entry point; `self._budget` is caller-supplied and used whenever
  not `None`.
- **Innocence attempt:** none — a bare `except: pass` around a real
  side-effecting call with no log is the plan's own canonical "silent"
  example, not an isolating one.
- **Fix:** log a warning naming the cost and the exception; `cost` is now
  computed once, outside the narrowed `try`, so the log message can always
  reference it even if the charge call itself raises before using it.
- **Test:** `test_c1_verbalized_sampler_logs_budget_charge_failure` — a
  real trigger: a fake `budget.charge()` that raises `RuntimeError`,
  asserted via `caplog` that a warning naming the cost is emitted (none
  was, pre-fix).

### C2 — VERIFIED DEFECT — `costing/tracker.py::CostTracker._load()` silently reset cumulative cost history to empty on any read/parse failure

- **Property violated:** same "silent pass with no logging" shape, in one
  of the tier's explicitly prioritized modules (`costing/`).
- **Location:** `orchestrator/costing/tracker.py:64-65` (pre-fix).
- **Finding:** if the persisted `cost_tracker.json` exists but is
  corrupt/partially written, `except Exception: pass` left
  `self._cumulative` at its just-initialized empty dict — indistinguishable
  from "no calls tracked yet," with zero log line identifying that a
  corrupt file was silently discarded.
- **Reachability:** live — `_load()` runs unconditionally in
  `CostTracker.__init__()`.
- **Innocence attempt:** none.
- **Fix:** added `logging`/`logger` (the module had neither), log a
  warning naming the file path and the exception before falling back to
  empty cumulative totals.
- **Test:** `test_c2_cost_tracker_logs_load_failure` — writes genuinely
  invalid JSON to the tracker's storage file, constructs `CostTracker`,
  asserts via `caplog` a warning is logged (none was, pre-fix).

### C3 — VERIFIED DEFECT — `cost.py::BudgetHierarchy._load_from_db()` silently dropped unparseable per-key spend rows

- **Property violated:** same shape, in the other explicitly prioritized
  money module (`cost.py`, sibling of `budget.py`) — plus a
  security/fairness angle: a team or job whose persisted spend value fails
  to parse silently resets to 0 for the rest of the process's life, which
  could let a team/job already near its budget ceiling look like it has
  full budget again.
- **Location:** `orchestrator/cost.py:282-286` (pre-fix).
- **Finding:** `except Exception: continue` inside the per-row restore
  loop — one bad row (e.g. a hand-edited or partially-written DB value)
  is dropped with no log identifying which `key` was skipped, unlike the
  sibling `_load_from_db`/`_save_to_db` failure paths (whole-function
  try/except) which already log at `warning` level for a total failure.
- **Reachability:** live whenever `BudgetHierarchy(db_path=...)` is
  constructed — runs in `__init__` via `_load_from_db()`.
- **Innocence attempt:** none.
- **Fix:** log a warning naming the specific `key` and the exception
  before `continue`-ing past the bad row.
- **Test:** `test_c3_budget_hierarchy_logs_unparseable_spend_row` — a real
  trigger: pre-populates the SQLite `budget_hierarchy` table with one
  corrupt row (`team:alpha`) and one valid row (`team:beta`), constructs
  `BudgetHierarchy` against that DB, asserts the valid row still loads
  (`_team_spent["beta"] == 4.5`, confirms the fix doesn't break the good
  path) while the corrupt key is both absent from `_team_spent` and named
  in a warning log (neither log assertion held pre-fix).

## Residual, triaged but not fixed this tier (from the same 18-candidate survey)

Recorded here so a later tier does not need to re-survey these — cleared
candidates are inventoried, not deleted, per §4:

- **`application/evaluator.py:184-190`** — a crashed/timed-out
  self-consistency judge run is logged (unlike C1-C3, this one already
  logs) but then a **fabricated 0.5** is appended to the `scores` list
  used for aggregation — indistinguishable from a genuine judge score.
  `[REQUIRES HUMAN REVIEW]`: the correct fix (exclude the failed run from
  aggregation vs. keep a neutral-fallback for "not enough real runs left")
  is a scoring-semantics design decision, not a one-line log-and-continue
  fix like C1-C3.
- **`generators/website_validator.py:846-855`** and its duplicate
  **`quality_control.py:516-534`** / **`quality/quality_control.py:538`**
  — a hardcoded-secret scanner silently skips unreadable files
  (`except Exception: continue`, zero logging) and then reports "No
  secrets/security issues found" — a false-clean security scan result
  biased in the dangerous direction. `[REQUIRES HUMAN REVIEW]`: same
  family as T2's C3 (a security check whose failure mode is a false
  "pass"), but scoped to a different subsystem (website-output scanning,
  not the codebase-writer gate T2 already fixed) — flagged for a
  dedicated fix rather than folded in here to keep this tier's diff
  reviewable.
- **`plugin/plugin_isolation_secure.py:421-425`** — sandbox network
  syscall blocking rules installed via `try: ... except Exception: pass`
  with no logging, unlike the sibling loop for genuinely-arch-dependent
  syscalls (which has a justifying comment). `[REQUIRES HUMAN REVIEW]`:
  security-relevant (sandbox escape surface) but reachability/severity
  needs its own dedicated investigation of `plugin_isolation_secure.py`'s
  live call sites before deciding a fix.
- **`services/scorers.py:40-44`**, **`rate_limiter.py:339-364`** — both
  return a plausible "real" fallback value (`0.0` score; last-known
  cached spend) on failure with no signal distinguishing it from a
  genuine result. Lower severity (bounded blast radius: one candidate
  ranked last; one rate-limit tier calculation staying stale) — `[UNK]`
  whether either is worth a dedicated fix; not pursued this tier.
- **`engine_core/container.py:930-957`**, **`infrastructure/
  snapshot_store.py:386-394`**, **`application/decomposer.py:420-425`**
  (DEBUG-only log, same shape as C1), **`memory_bank.py:561-590`**,
  **`vcs/integration.py:154-170`**, **`vcs/sync.py:66-77`**,
  **`tenancy.py:243-273`** / **`integrations/tenancy.py:243-273`**
  (partial-load, already logs but doesn't state scope),
  **`state_mgmt/workspace.py:400-426`** (already logs, explicit
  documented tradeoff) — all individually triaged by the survey and
  recorded as `[UNK]`/lower-confidence; none independently re-verified or
  fixed this tier. See the agent's full write-up (this session's
  transcript) for per-site detail if a future tier picks these up.
- Confirmed **not** defects, already correctly isolating: `budget.py`,
  `application/budget_enforcer.py`, `generators/secrets_manager.py` (zero
  broad-except sites at all); `circuit_breaker.py:208`,
  `operations/resilience.py:288/327`,
  `application/project_runner.py:489` (re-raise/explicit-fallback
  chains); `infrastructure/state.py`'s several `except Exception: pass`
  sites (all cleanup-before-a-`raise`, primary failure still propagates).

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
