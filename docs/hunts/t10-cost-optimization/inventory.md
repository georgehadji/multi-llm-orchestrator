# T10 — Cost-Optimization Remainder — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4/§7.5 and `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`'s
wave definition. Budget: 19 files surveyed, 4 fixed.

## Phase 0/1/2/3 — survey method

A background agent read all 19 files in the wave's declared scope
(`cost_optimization/`'s 13 files, `costing/analytics.py`, `rate_limiter.py`,
`token_budget.py`, `token_optimizer.py`, `provisioned_throughput.py`),
tracing live reachability for each via repo-wide grep, with two specific
leads prioritized per the wave plan: the `token_budget.py` name collision
across two locations, and whether `cost_optimization/docker_sandbox.py`
is actually the live container-execution sandbox (it turned out not to be
— see below).

## Candidates fixed

### C1 — VERIFIED DEFECT — `orchestrator/token_budget.py` was an unshimmed duplicate of `infrastructure/token_budget.py`

- **Property violated:** an unshimmed duplicate pair — the dominant defect
  shape this entire hunt has found, here caught *before* divergence rather
  than after.
- **Location:** `orchestrator/token_budget.py` (root, 447 lines).
- **Finding:** confirmed via `diff` to be byte-for-byte identical to
  `orchestrator/infrastructure/token_budget.py` — not yet diverged, but a
  complete independent copy rather than a shim, meaning a future fix to
  either file (as already happened repeatedly to other such pairs across
  T1/T2/T3/T5/T7/T9) would silently not reach the other.
- **Reachability:** zero live callers of either copy anywhere in the repo
  (confirmed via exhaustive grep for `TokenBudgetManager`,
  `get_token_budget_manager`, `allocate_tokens`).
- **Innocence attempt:** "byte-identical today, so there's no live
  divergence to fix." True but beside the point — the defect is the
  *absence of a shim relationship*, not a current behavioral difference;
  every prior instance of this exact shape in this hunt started out
  byte-identical too.
- **Note on naming:** NOT the same feature as
  `orchestrator/cost_optimization/token_budget.py` (a different class,
  `TokenBudget`, enforcing per-*phase* output-token ceilings with its own
  pricing table) — that file shares a filename but not a purpose, and is
  separately covered by the "confirmed clean" list below.
- **Fix:** converted `orchestrator/token_budget.py` into a re-export shim
  of `orchestrator.infrastructure.token_budget`.
- **Test:** `test_c1_root_token_budget_is_canonical` — asserts class
  identity (pre-fix: two distinct class objects with identical content).

### C2 — VERIFIED DEFECT — `orchestrator/provisioned_throughput.py` was an unshimmed duplicate of `operations/provisioned_throughput.py`

- **Property violated:** same shape as C1.
- **Location:** `orchestrator/provisioned_throughput.py` (root, 459 lines).
- **Finding:** byte-for-byte identical to
  `orchestrator/operations/provisioned_throughput.py`. A third copy,
  `orchestrator/infrastructure/provisioned_throughput.py`, already
  correctly shims to the `operations/` copy — establishing `operations/`
  as the intended canonical location and leaving the root copy as the
  orphaned, un-shimmed twin.
- **Reachability:** zero live callers of any of the three copies anywhere
  in the repo.
- **Innocence attempt:** same as C1 — byte-identical today is not evidence
  against the missing-shim defect itself.
- **Fix:** converted `orchestrator/provisioned_throughput.py` into a
  re-export shim of `orchestrator.operations.provisioned_throughput`,
  matching the direction the existing `infrastructure/` shim already
  established.
- **Test:** `test_c2_root_provisioned_throughput_is_canonical`.

### C3 — VERIFIED DEFECT — `cost_optimization/cost_optimization_integration.py` had a broken relative import

- **Property violated:** the exact "one dot short/long for this file's
  actual package depth" bug this hunt has now found at least five times
  (`code_executor.py`, `integrations/tenancy.py`, three `safety/`
  duplicates in T9).
- **Location:** `Tier1OptimizationMixin`'s module-level
  `from .log_config import get_logger` (1 dot) — the file lives at
  `orchestrator/cost_optimization/`, so `log_config` (at
  `orchestrator/log_config.py`) needs 2 dots. The same off-by-one also
  affected two `TYPE_CHECKING`-only imports (`.engine`, `.models`) a few
  lines below — not a runtime crash on their own, but wrong for the same
  reason and would confuse any type checker that follows them.
- **Reachability:** confirmed by direct execution to raise
  `ModuleNotFoundError` unconditionally at import time. Zero importers of
  this module anywhere in the repo today, so currently dormant — but this
  is why `Tier1OptimizationMixin` (the file's own docstring shows it as
  the intended wiring point for prompt caching / batch API / token budget
  into `Orchestrator`) could never have been adopted as written.
- **Innocence attempt:** none needed — a plain, unambiguous relative-import
  depth mistake.
- **Fix:** corrected all three relative imports to the proper 2-dot depth.
- **Test:** `test_c3_cost_optimization_integration_imports_cleanly` — a
  real trigger: simply importing the module (pre-fix:
  `ModuleNotFoundError: No module named
  'orchestrator.cost_optimization.log_config'`).

### C4 — VERIFIED DEFECT — `cost_optimization/docker_sandbox.py::DockerSandbox.execute()` had no path-containment check on caller-supplied filenames

- **Property violated:** a sandbox execution primitive that writes files
  based on unsanitized, caller-supplied names is a path-traversal /
  arbitrary-host-file-write vector — the same class of defect this hunt
  treats with maximum scrutiny for anything execution-adjacent (T7's
  `appbuilder/verifier.py`, T9's `safety/` sandbox survey).
- **Location:** `execute()`'s file-writing loop:
  `file_path = workspace / filename; file_path.parent.mkdir(...);
  file_path.write_text(content)`.
- **Finding:** `filename` comes directly from the caller's `code_files`
  dict with no validation. Two escape vectors: (a) Python's `pathlib`
  semantics mean `Path("/tmp/sandbox_x") / "/etc/cron.d/evil"` evaluates
  to `Path("/etc/cron.d/evil")` outright — an absolute-path key completely
  discards the sandbox prefix; (b) a `"../"`-containing relative key
  escapes via normal traversal. Either writes to an arbitrary host path
  *before* the Docker container is even created.
- **Reachability:** the survey traced the only real construction site of
  `DockerSandbox` — `safety/code_executor.py::CodeExecutor.
  _execute_in_sandbox()` — and found `CodeExecutor` itself has zero live
  callers anywhere (confirmed: the one file that references it,
  `tests/unit/test_safety_code_executor_sandbox_import.py`, only inspects
  the import statement's AST, never constructs or invokes the class).
  Currently fully dead end-to-end.
- **Innocence attempt:** "dead code, so the traversal never fires." Holds
  for current reachability. Fixed anyway — cheap, self-contained, and this
  hunt has consistently fixed real security defects inside dormant
  subsystems (T1-C4, T8-C4/C6/C7, T9-C1) without wiring the subsystem
  itself live. Did **not** add the separately-flagged missing container
  hardening (`read_only`, `cap_drop`, `security_opt`, `pids_limit`) — that
  is new capability/configuration surface, not a bounds-check fix, and
  building it into code nobody currently calls is scope creep matching
  T9's treatment of `safety/sandbox.py`'s own unenforced resource limits.
  `[REQUIRES HUMAN REVIEW]` if `CodeExecutor`/`DockerSandbox` is ever wired
  live.
- **Fix:** resolves both the workspace root and the candidate file path,
  rejects (raises, caught by the existing outer `except Exception`,
  returned as a normal failed `ExecutionResult`) any path that resolves
  outside the workspace.
- **Tests:** `test_c4_docker_sandbox_rejects_path_traversal_filename` (a
  real trigger: a `"../../../etc/evil.txt"` filename, with `docker` module
  import faked via `sys.modules` injection since the real `docker` Python
  package isn't installed in this environment — the containment check
  fires before `client.containers.run()` is ever reached, so the fake
  client object never needs to support it) and
  `test_c4_docker_sandbox_accepts_normal_filename` (confirms a legitimate
  `"main.py"` filename is not rejected by the new check — it still fails
  downstream on the fake client's missing `.containers` attribute, which
  is expected and asserted against, not the containment error).

## Candidates surveyed, not fixed — cleared or `[REQUIRES HUMAN REVIEW]`

- **`cost_optimization/__init__.py`'s `get_optimization_config()`/
  `update_config()`** — the documented, argparse-registered `--tdd-first`
  CLI flag (`entrypoints/cli_dispatch.py`) mutates
  `OptimizationConfig.enable_tdd_first`, but the only code that reads it
  (`application/task_executor.py::TaskExecutor._try_tdd_generation`) has
  zero construction sites anywhere — the live executor never references
  it. A user passing `--tdd-first` gets no error and silently receives
  ordinary generation. This is the same root cause T1 already recorded as
  C3 (`[REQUIRES HUMAN REVIEW]`, `BudgetEnforcer`/`TaskExecutor` never
  wired into the live pipeline) — not a new independent bug, just a
  concrete, user-visible symptom of the same already-recorded gap. Not
  re-fixed here; making the CLI reject/warn on a flag it can't honor is a
  UX decision, not a bug-fix-scope change.
- **`cost_optimization/pricing_cache.py`** — a fourth, fully independent
  pricing mechanism (live-API + disk cache + hardcoded fallback),
  duplicating `models.py::COST_TABLE`/`costing/core.py`/
  `costing/analytics.py`. Zero callers, not even re-exported by
  `cost_optimization/__init__.py`. `[REQUIRES HUMAN REVIEW]` — consolidating
  4-5 independent pricing mechanisms is a design decision, not a bug.
- **`rate_limiter.py`** (whole module) — T5 already cleared
  `GrokRateLimiter.acquire()`'s internal concurrency safety; T8 already
  cleared `fetch_current_spend()` as dead. This tier's survey extended
  that to full-module reachability: the one construction site,
  `engine_core/service_collection.py::SearchServices.build()`, is itself
  dead (zero importers of `service_collection.py` anywhere). Recorded as a
  confirmation, not a new finding.
- **`costing/analytics.py`** — `CostAnalytics`, a fifth independent
  cost-tracking/forecasting engine, reachable via a correct shim
  (`cost_analytics.py`) but constructed nowhere live. Arithmetic checked,
  no bugs found. `[UNK]`/dead, not fixed.
- **The other 8 dead `cost_optimization/` files** (`batch_client`,
  `dependency_context`, `github_push`, `model_cascading`, `prompt_cache`,
  `speculative_gen`, `streaming_validator`, `structured_output`,
  `tier3_quality`) — individually verified (not just at category level)
  that T1's C6 finding ("imported everywhere, instantiated nowhere") holds
  with no exceptions: every primary class's only construction sites are
  its own file or the dead `Tier1OptimizationMixin`.
- **`token_optimizer.py`** — confirmed correct and live: a genuine,
  accurate shim to `infrastructure/token_optimizer.py::TokenOptimizer`,
  reachable via `orchestrator/mcp_server.py`'s `__main__` entry.

## Out-of-scope handoff findings (discovered while tracing this tier, not fixed here)

Recorded per this hunt's established handoff convention (T1 → T7's
`code_executor.py` fix; T7 → this tier's own C1/C2 discovery), so a later
wave does not need to re-discover these:

- **`application/cache_warmup.py::warm_cache_for_level()`** — called from
  `engine.py:1052`, the real parallel-task-execution hot path — does
  `from ..operations.cache_warmup import warm_prompt_cache`, but
  `orchestrator/operations/cache_warmup.py` **does not exist**. Confirmed
  by direct execution: raises `ModuleNotFoundError`, caught by the
  function's own `except Exception` and logged as "Cache warming failed
  (non-critical)". Prompt-cache warming for parallel batches silently
  never happens, on every run, misreported as a benign transient failure.
  Belongs to `application/` — `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`
  wave T12's territory, not T10's. Severity note: this is a missed
  cost/latency optimization (provider-side prompt caching never engages),
  not a correctness or security defect — the system still produces correct
  results, just without this optimization's benefit.
- **`orchestrator/integrations/mcp_server.py:64`** —
  `from .token_optimizer import TokenOptimizer` resolves to a
  non-existent `orchestrator.integrations.token_optimizer` — would raise
  `ModuleNotFoundError` if this (fuller, more-recently-extended) MCP-server
  variant is ever run via an external MCP client. The root `mcp_server.py`
  variant this tier's own `token_optimizer.py` shim correctly serves is
  unaffected. Belongs to wave T15's `integrations/` territory.

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
