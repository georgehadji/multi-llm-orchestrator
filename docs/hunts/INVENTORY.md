# Defect-Hunt Programme — Running Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §5. Appended once per tier on close. Cleared candidates are
recorded, not deleted, so a later tier does not re-raise a settled false alarm.

## T0 — Census repair (closed)

Full detail: `docs/hunts/t0-census-repair/inventory.md`, `docs/hunts/t0-census-repair/coverage.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `CLAUDE.md` cited `tests/stress_test.py` (S2/S6/S7) as an existing file with documented failures; never committed. Corrected in both locations. |
| C2 | **VERIFIED DEFECT — FIXED** | `orchestrator/quality/toml_validator.py` unconditionally imported stdlib `tomllib` (3.11+), breaking on the declared `requires-python = ">=3.10"` floor. Version-gated with a `tomli` fallback (`python_version < '3.11'` marker in `pyproject.toml`). |
| C3 | **CLEARED (innocent)** | `stress`/`load` pytest markers registered, zero usage — deliberately reserved per `projects/stress_test/README.md`, not dead config. Do not re-raise. |
| C4 | **RESIDUAL — `[REQUIRES HUMAN REVIEW]`** | CI runs Python 3.12 exclusively on every job; never tests the declared 3.10/3.11 floor. Resolves §7.1. Not fixed here — a CI/CD pipeline change needs explicit sign-off. |
| C5 | **RESIDUAL — `[UNK]`** | `.claude/skills/orchestrator-failure-archaeology/SKILL.md`'s incident table cites 4 commit hashes that do not resolve in current git history. Out of T0's declared scope; not fixed. |

**Cumulative residual-UNKNOWN set added by T0:** C4, C5 (above).

**Pre-registered known-innocents reconfirmed still failing, still environmental (§7.3):**
`tests/unit/test_openrouter_model_audit.py::test_audit_against_live_catalogue_is_clean`,
`::test_runtime_only_ids_resolve_via_endpoints` — sandbox blocks `openrouter.ai`.

**Gate status at T0 close:** black/ruff/lint-imports/root-freeze/test-markers/mypy(core)/
bandit all PASS. `pytest tests/ -m "unit or integration"`: 2460 passed, 2 failed (both
pre-registered environmental), 21 skipped.

**Next tier:** T1 (money) — `budget.py`, `cost.py`, `cost_tracker.py`, `cost_analytics.py`,
`cost_optimization/`. Its Phase 0 delta must re-verify the shared census against the tree
this tier leaves behind, per §4 Step 1.

## T1 — Money (closed)

Full detail: `docs/hunts/t1-money/inventory.md`, `docs/hunts/t1-money/coverage.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `orchestrator/costing/core.py` was an independent, hand-copied fork of `cost.py`'s `BudgetHierarchy`/`CostPredictor`/`CostForecaster` that silently diverged: `cost.py`'s `BudgetHierarchy` gained SQLite persistence never backported to the fork. Reachable via `orchestrator.costing.BudgetHierarchy`. Fixed with a re-export shim; the two class objects are now identical. |
| C2 | **VERIFIED DEFECT — FIXED** | `BudgetEnforcer.record_cost()` mutated `budget.spent_usd` directly (bypassing `Budget`'s asyncio.Lock, the exact TOCTOU protection BUG-001/FIX-001a added) and called `budget_hierarchy.record_cost(...)` — a method that does not exist (real method: `charge_job`), silenced with `# type: ignore[attr-defined]`. Same defect shape as the file's own documented prior bug, BUG-004. Fixed: delegates to `Budget.charge()`; hierarchy call removed. |
| C3 | **RESIDUAL — `[REQUIRES HUMAN REVIEW]`** | `BudgetEnforcer` is never instantiated anywhere in the live pipeline (`container.py` hardcodes `budget_enforcer = None`); `TaskExecutor`/`task_handlers.py`'s typed-dispatch system it exclusively drives is likewise never constructed (the live executor is the differently-named `ExecutorService`). `Budget`'s own reserve/commit/release concurrency-safety pattern (also BUG-001/FIX-001a) is equally unused. Not fixed — wiring either subsystem in live is an architecture decision. |
| C4 | **VERIFIED DEFECT (in the C3-dead subsystem) — FIXED** | Every typed handler in `task_handlers.py` hardcoded `TaskResult(cost_usd=0.0, tokens_used={0,0})` regardless of the real, billable LLM call made; `_call_llm()` discarded `response.cost_usd`/tokens at the source; the `budget` parameter was never referenced. Fixed: real cost/tokens now flow through, and a real `Budget` gets charged when given one. Currently dormant per C3, but one wiring change from being the default path for 4 task types. |
| C5 | **VERIFIED DEFECT — FIXED** | `ProjectRunner.run_job()`'s cross-run settlement computed `actual_spend = max_usd - remaining_usd`, which silently clamps at `max_usd` on any per-run overspend (`remaining_usd` floors at 0.0; `Budget.charge()` enforces no ceiling) before charging `BudgetHierarchy` — undercounting true cross-run spend. Fixed: charges `budget.spent_usd` directly. |
| C6 | **RESIDUAL — `[REQUIRES HUMAN REVIEW]`** | `cost_optimization/`'s entire cost-reduction machinery (batching, caching, model cascading, speculative generation, streaming validation, token budgeting) is imported everywhere it should be used and instantiated nowhere; `cost_optimization_integration.py`'s mixin has zero importers anywhere. A standing, silent excess-spend condition (the mirror of C4/C5's overspend bugs), not fixed here since wiring it in is a cost/latency/quality architecture decision. |

**Cumulative residual-UNKNOWN set added by T1:** C3, C6 (above), plus one out-of-tier finding
handed off rather than absorbed: a confirmed-broken 3-dot relative import in
`orchestrator/safety/code_executor.py:183` (found tracing C6), queued as a separate task —
T7's territory (execution/filesystem safety), not a money defect.

**Related hygiene observation, not a separate candidate:** `orchestrator/services/executor.py`
vs `orchestrator/application/executor.py` (`ExecutorService`, defined twice) — diffed in full,
no behavioral divergence found (docstring/comment/whitespace only). A fourth undocumented
duplicate pair alongside C1 and `REMEDIATION_PLAN.md`'s own tracked two, but with nothing to
fix since the two copies already agree.

**Gate status at T1 close:** black/ruff/lint-imports/root-freeze/test-markers/mypy(core)/
bandit all PASS. `pytest tests/ -m "unit or integration"`: 2476 passed (+16 over T0's 2460,
exactly the new tests), 2 failed (both pre-registered environmental, §7.3, unchanged), 21
skipped — zero regressions, verified via a retroactive `git stash`-based RED→GREEN check on
all four fixes (15/16 new tests failed for the predicted reason against the pre-fix tree, 1
correctly passed in both states as a fact-pin, all 16 pass on the fixed tree).

**Next tier:** T2 (credentials & trust boundary) — `security/`, `api_clients.py`, `gateway.py`,
`api_server.py`, `generators/secrets_*`, the 43 `api_key` modules. Its Phase 0 delta must
re-verify the shared census against the tree T1 leaves behind, per §4 Step 1 — in particular,
re-check `orchestrator/infrastructure/llm_client.py` (where `UnifiedClient`/`APIResponse`
actually live, discovered this tier) against the plan's `api_clients.py`-only assumption.

## T2 — Credentials & trust boundary (closed)

Full detail: `docs/hunts/t2-credentials/inventory.md`, `docs/hunts/t2-credentials/coverage.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `generators/secrets_generator.py` was self-referential (`from ..generators.secrets_generator import *`), resolving to itself and ending up silently empty. Fixed as a shim to the real 745-line `orchestrator/secrets_generator.py`. |
| C2 | **VERIFIED DEFECT — FIXED** | `codebase_writer.py` (root) and `codebase/writer.py` (subpackage) diverged — subpackage gained SEARCH/REPLACE patching and a pre-destructive-op snapshot safety net the root never got, and two different existing test files depended on the two different copies. Fixed: root converted to a shim of the fuller subpackage version. |
| C3 | **VERIFIED DEFECT — FIXED** | `ModificationGate._check_secrets()` (the live, wired path) appended detected hardcoded secrets to `.warnings`, which `apply()` never reads (only `.errors` blocks a write) — a detected password/api_key/secret/token never stopped the write. Also stopped embedding up to 20 raw chars of the matched secret into the (discarded) message. |
| C4 | **VERIFIED DEFECT — FIXED** | `Tenant.to_dict()` never serialized `api_key`, so every tenant's real key was lost on every restart and all restored tenants collided onto one empty-string entry. Fixed in both `tenancy.py` and `integrations/tenancy.py`. |
| C5 | **VERIFIED DEFECT — FIXED** | `integrations/tenancy.py` could not be imported at all (`from .log_config import get_logger` — correct at root depth, wrong from within `integrations/`) — found while testing C4. One-line fix, same shape as the T0 docker_sandbox bug. |

**Residual, not independently fixed (see coverage.md for full detail):** `operations/diagnostics.py`'s health check accepts OPENAI/GOOGLE/ANTHROPIC keys the live client never reads; `secrets_manager.py`'s log-masking `SecretsFilter` is fully built but never installed on any real logger; several `logger.error(..., e)` sites re-log SDK exceptions of unconfirmed content from calls built with real Bearer headers; two more hardcoded-secret scanners (`safety/security_review.py`, `safety/security_validator.py`) embed raw secret substrings into their findings and have zero live callers; `gateway.py`/`integrations/multi_tenant_gateway.py` use non-constant-time key comparisons in code of unconfirmed reachability. No live path was found writing a real provider key into generated website output.

**Gate status at T2 close:** black/ruff/lint-imports/root-freeze/test-markers/mypy(core)/bandit
all PASS (bandit's 3 Low findings are pre-existing, unrelated `subprocess` usage in
`install_dependency()`).

## T3 — Persistence & resume (closed)

Full detail: `docs/hunts/t3-persistence/inventory.md`, `docs/hunts/t3-persistence/coverage.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `state_mgmt/checkpoints.py` silently lacked `ContentCheckpointManager` (real content-restoring rollback via a SnapshotPort) that root `checkpoints.py` had — neither had a live importer, fixed as a shim to the fuller root version regardless, closing the landmine. |
| C2 | **VERIFIED DEFECT — FIXED** | `ResumeDetector.find_resumable_project()` was an acknowledged stub that always returned `None` ("would be implemented with async/await"), despite its own Jaccard/recency scoring machinery being fully built and already used correctly elsewhere (`cli_dispatch.py::_check_resume`). Completed by wiring it to the already-existing `StateManager.find_resumable()`, mirroring the already-live pattern. |

**Cleared (innocent):** `orchestrator.state`/`orchestrator.state_mgmt.state` (both correct
redundant shims to `infrastructure/state.py`), `orchestrator.async_event_store` (correct shim).

**Gate status at T3 close:** black/ruff/lint-imports/root-freeze/test-markers/mypy(core)/bandit
all PASS.

## T4 — Concurrency & resource lifecycle (closed)

Full detail: `docs/hunts/t4-concurrency/inventory.md`, `docs/hunts/t4-concurrency/coverage.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `PipelineRunner.execute_all()`'s `run_one()` let a failed task's exception propagate to `asyncio.gather(return_exceptions=True)`, which only logged it — `results[tid]` was never set, so `ProjectState.results` silently had no record the task was ever attempted. Fixed: catches the exception, records a `FAILED` `TaskResult` via the same `_build_failure_result()` convention already used in `task_executor.py`. |

**Cleared (innocent):** 5 fire-and-forget `asyncio.create_task()` sites checked for the
GC-mid-flight leak pattern, all already using the correct held-reference pattern.

**Residual, not fixed (dead code):** `A2ACoordinator.distribute_task()` has a genuine task-leak
shape on early exception/cancellation, but has zero live callers (`A2AManager` aliases a
different class, `A2AQueueManager`).

**Gate status at T4 close:** black/ruff/lint-imports/root-freeze/test-markers/mypy(core)/bandit
all PASS.

## T5 — Resilience state machines (closed)

Full detail: `docs/hunts/t5-resilience/inventory.md`, `docs/hunts/t5-resilience/coverage.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `operations/circuit_breaker.py` silently diverged from the canonical, live `circuit_breaker.py`: missing the `probe_in_flight` enforcement (BUG-001/BUG-002 fixes) that limits a HALF_OPEN breaker to exactly one concurrent probe — verified with a real concurrent-`check()` trigger, pre-fix both callers were admitted. Because `operations/resilience.py` sources its `CircuitBreakerOpen` from this same divergent copy, a real `CircuitBreakerOpen` raised by any live breaker (all constructed from the canonical module) was not an instance of the exception class `run_with_resilience()`'s `except` clause checked for — it would not have been caught. Fixed: converted to a shim of the canonical module, resolving both issues at once. |

**Cleared (innocent):** `resilience.py` and `integration_circuit_breaker.py` duplicate pairs,
both correct non-diverged shims. `rate_limiter.py::GrokRateLimiter.acquire()`'s check-then-act
RPM/TPM logic confirmed atomic (already carries prior `BUG-NEW-001`/`BUG-NEW-002` hardening).

**Residual, not investigated:** `orchestrator/adaptive_router.py` (192 lines) — not read this
tier.

**Gate status at T5 close:** black/ruff/lint-imports/root-freeze/test-markers/mypy(core)/bandit
all PASS. 61/61 existing resilience/circuit-breaker tests still pass, zero regressions.

## T6 — Error-path sweep (broad-except info loss) (closed)

Full detail: `docs/hunts/t6-error-paths/inventory.md`, `docs/hunts/t6-error-paths/coverage.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `application/verbalized_sampling.py::VerbalizedSampler.sample()` charged a caller-supplied budget for a real, already-incurred LLM cost inside a bare `except Exception: pass` — a charge failure left zero trace anywhere. Fixed: logs a warning naming the cost and the exception. |
| C2 | **VERIFIED DEFECT — FIXED** | `costing/tracker.py::CostTracker._load()` silently reset cumulative cost history to empty on any read/parse failure of its persisted JSON, indistinguishable from "no history yet." Fixed: added logging (module had none), warns naming the file and exception. |
| C3 | **VERIFIED DEFECT — FIXED** | `cost.py::BudgetHierarchy._load_from_db()` silently dropped any per-team/per-job spend row that failed to parse, with no log naming which key — could let a near-limit team/job appear to have full budget again after a restart. Fixed: warns naming the specific key. |

**Residual, triaged but not fixed (15 candidates from an 18-candidate ranked survey — see
inventory.md for full detail):** highest-severity is `application/evaluator.py`'s
self-consistency loop injecting a fabricated 0.5 score into aggregation on a judge-call failure
(logged, but the fake score is indistinguishable from real) — `[REQUIRES HUMAN REVIEW]`, a
scoring-semantics decision, not a log-line fix. Also flagged: a hardcoded-secret scanner
(`generators/website_validator.py`, duplicated in `quality_control.py` x2) silently skips
unreadable files and reports a false-clean scan result; `plugin/plugin_isolation_secure.py`'s
network-syscall sandbox rule installation swallows failures with no log (security-relevant,
reachability not independently re-verified); several lower-severity success-shaped fallbacks
(`services/scorers.py`, `rate_limiter.py::fetch_current_spend`). Confirmed genuinely clean:
`budget.py`, `application/budget_enforcer.py`, `generators/secrets_manager.py` (zero broad-except
sites at all); `circuit_breaker.py`, `operations/resilience.py`, `project_runner.py`'s checked
sites (correctly isolating); `infrastructure/state.py`'s `pass`-only sites (cleanup-before-raise).

**Gate status at T6 close:** black/ruff/lint-imports/root-freeze/test-markers/mypy(core)/bandit
all PASS. New tests 3/3 (RED→GREEN verified). 155/155 existing cost/budget/verbalized-sampling
tests still pass, zero regressions.

## T7 — Execution & filesystem surface (closed)

Full detail: `docs/hunts/t7-execution/inventory.md`, `docs/hunts/t7-execution/coverage.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `appbuilder/verifier.py` silently diverged from the canonical `app_verifier.py`, missing a fix (`str(req_file.resolve())` vs unresolved `str(req_file)`) for a pip-install `subprocess.run(..., cwd=output_dir)` call that would otherwise double-resolve a relative requirements-file path and fail. `AppBuilder` itself imports the canonical, fixed class directly and was unaffected — but `appbuilder/__init__.py`'s `from .builder import *` then `from .verifier import *` import order meant the *package's own public name*, `orchestrator.appbuilder.AppVerifier`, silently resolved to the buggy class instead. Fixed: converted to a shim of the canonical module, matching the same duplicate-pair remediation used every prior tier. |

**Reachability caveat:** no live caller does `from orchestrator.appbuilder import AppVerifier`
today (grepped) — the fix closes a real, exposed public-API landmine rather than an
actively-triggered bug, consistent with how T4/T5 handled similarly dormant divergences.

**Phase 0 census (not exhaustively pursued):** 54 files touch
`subprocess.`/`eval(`/`exec(`. Only the `app_verifier.py`/`appbuilder/verifier.py` pair was
investigated this tier (matched the by-now-established root-vs-subpackage duplicate pattern).
52 files remain unexamined, including the two whose names most directly suggest
security-relevant subprocess isolation: `safety/sandbox.py`, `safety/secure_execution.py`.
`[UNK]` whether they carry their own defects.

**Gate status at T7 close:** black/ruff/lint-imports/root-freeze/test-markers/mypy(core)/bandit
all PASS. New tests 3/3 (RED→GREEN verified); zero pre-existing tests covered this area before
this tier (confirmed via grep, not just absence of failures).

## Unrelated fix merged during this tier sequence: OpenRouter model-catalog update

Per explicit user request (separate from the defect-hunt protocol), a background agent
researched OpenRouter's current model catalog/pricing/reasoning-tokens/web-search docs and
updated `orchestrator/models.py`. Reviewed and merged onto this branch:
- **A genuine enum-aliasing corruption bug, verified independently**: `Model.QWEN_3_6_FLASH`
  was assigned the same string value as `Model.GPT_4O_MINI` (`"openai/gpt-4o-mini"`), making it
  a Python enum *alias* (`Model.QWEN_3_6_FLASH is Model.GPT_4O_MINI` was `True`). Because
  `COST_TABLE`/`CONTEXT_WINDOWS` are dict literals, the later `QWEN_3_6_FLASH` entry silently
  overwrote `GPT_4O_MINI`'s real pricing (`$0.12/$0.50` instead of the correct `$0.15/$0.60`)
  and context window (`32768` instead of `131072`) — corrupting a model used as a fallback
  target in ~7 places. Fixed by giving `QWEN_3_6_FLASH` its own correct id/pricing.
- 43 stale `COST_TABLE` prices corrected, cross-checked against `openrouter_models.json`
  (a git-tracked catalogue snapshot already in this repo, committed 2026-08-01) — a new test,
  `tests/unit/test_cost_table_pricing_sync.py`, passes against it (3/3).
- Confirmed, not fixed here (each is a real gap, out of scope for a pricing-data pass):
  `orchestrator/config/costs.json` is dead code for pricing purposes (`COST_TABLE` is a
  hardcoded dict literal that shadows the JSON-loader path entirely — verified the two never
  produce the same object or content); the `PhasePolicy`/`ReasoningEffort` reasoning-token
  request machinery is fully built but has zero callers into `llm_client.py`; prompt-cache
  discount pricing (`input_cache_read`/`input_cache_write`) isn't reflected anywhere in cost
  estimation, meaning successful caching is invisible to budget accounting (safe-direction
  overestimate, not a risk, but not accurate either).
- Full suite: 2500 passed (+3 for the new pricing-sync test), 2 pre-registered environmental
  failures unchanged, 21 skipped — zero regressions. black/ruff/mypy(core) all pass.

## T8 — Remainder, coverage-ordered (closed, explicitly PARTIAL)

Full detail: `docs/hunts/t8-remainder/inventory.md`, `docs/hunts/t8-remainder/coverage.md`.
Per §8, a remainder tier never "completes" — see `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`
for the data-backed accounting of what's still genuinely unexamined (841 of 892 backend files).

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `generators/secrets_manager.py`'s `SecretsFilter` was fully built but never attached to any real logger — `log_config.py::configure_logging()` only ever attached `CorrelationIdFilter`. Fixed: attaches `SecretsFilter()` too. `configure_logging()` itself still has zero live callers (separate, flagged architecture question). |
| C2 | **CLEARED (innocent)** | `gateway.py`/`multi_tenant_gateway.py` key comparisons — wrong threat model (hash-then-dict-lookup, not vulnerable `==`); also dead code. `gateway.py` is additionally permanently shadowed by the sibling `gateway/` package (hygiene landmine, recorded not fixed). |
| C3 | **SPLIT** | `safety/secure_execution.py` — CLEARED, no defect. `safety/sandbox.py` — dead code, bypassable denylist + unenforced resource limits, `[REQUIRES HUMAN REVIEW]` (building real enforcement into dead code is scope creep, not a bug fix). |
| C4 | **VERIFIED DEFECT — FIXED** | `plugin/plugin_isolation_secure.py`'s seccomp network-syscall-blocking loop silently swallowed every rule-install failure with no log. Fixed: logs a warning. Incidental: the whole `orchestrator/plugin/` package couldn't even be imported (`Plugin` referenced from the wrong sibling module) — fixed, since C4 was untestable without it. |
| C5 | **VERIFIED DEFECT — FIXED** | `generators/website_validator.py`'s secret scanner silently skipped unreadable files and reported a clean scan — live, gates the real `orchestrator website --min-quality`/`--require-all-checks` CLI flags. Fixed: logs the failure, unreadable files now make the check fail rather than silently pass. |
| C6 | **VERIFIED DEFECT — FIXED** | `operations/diagnostics.py`'s environment check required `OPENAI_API_KEY`/`GOOGLE_API_KEY`/`ANTHROPIC_API_KEY`/`MINIMAX_API_KEY` — none of which `infrastructure/llm_client.py`'s live `UnifiedClient` ever reads (it reads `OPENROUTER_API_KEY`/`DEEPSEEK_API_KEY`/`XAI_API_KEY`). Fixed the check. **Not fixed:** `CLAUDE.md` and `.env.example` document the same wrong keys project-wide — `[REQUIRES HUMAN REVIEW]`, ambiguous whether this reflects an intentional OpenRouter migration or a real regression. |
| C7 | **VERIFIED DEFECT — FIXED** | `adaptive_router.py::AdaptiveRouter.is_available()` returned `True` unconditionally for every model whenever any concurrent writer held the lock, ignoring already-committed DISABLED/DEGRADED state. Currently fully dead (`container.py` hardcodes `adaptive_router=None`). Fixed by deleting the optimistic fast path (the state reads need no lock per the method's own docstring). |
| C8 | **CLEARED (innocent)** | `services/scorers.py`, `rate_limiter.py::fetch_current_spend` — both confirmed fully dead by exhaustive grep; fail-safe-low design is defensible if ever wired in. |

**Gate status at T8 close:** black/ruff/lint-imports/root-freeze/test-markers/bandit all PASS.
mypy: 0 new errors from this tier's 3 core-path files (isolated diff confirmed), 1607
pre-existing errors elsewhere (container.py/meta_integration.py/meta/) untouched and unrelated.
New tests 5/5 (RED→GREEN verified, including the incidental C4 import-fix RED state). Full
suite: 2517 passed, 2 pre-registered environmental failures unchanged, 21 skipped — zero
regressions across 158 targeted regression tests plus the full unit+integration run.

**Next:** `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` defines T9 (safety/execution/plugin
surface, in progress) through T16 (final catch-all), prioritized by architectural blast radius.

## T9 — Safety/execution/plugin surface (closed)

Full detail: `docs/hunts/t9-safety-execution/inventory.md`, `docs/hunts/t9-safety-execution/coverage.md`.
Per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`'s wave definition — first of waves T9-T16.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | Three dead `safety/` duplicates (`architecture_rules.py`, `architecture_advisor.py`, `reference_monitor.py`) each carried a broken relative import, the same shape T1 already fixed once in `code_executor.py`. `architecture_advisor.py`'s copy had also fallen behind the canonical module (missing "static" project-type data). Fixed: all three converted to re-export shims of their live root canonicals. Incidental: surfaced 8 pre-existing mypy errors in the now-reachable canonical `reference_monitor.py`/`specs.py` — real, pre-existing, out of scope here. |
| C2 | **VERIFIED DEFECT — FIXED** | `plugins/discovery.py::load_plugin()` built the bundled-plugin import path as `orchestrator.plugin.plugins.<kind>.<name>` (singular, not even a package) instead of `orchestrator.plugins.<kind>.<name>` (matching `_bundled_plugin_path()`) — any bundled plugin would be discovered then silently fail to import. Dormant today (no bundled plugin dirs exist yet). Fixed. |
| C3 | **VERIFIED DEFECT — FIXED** | `safety/generated_output_scanner.py::scan_output_dir()` — live (wired via `output_organizer.py`) — silently skipped unreadable files with no count/log, the fourth known instance of the "silent scan gap" shape (T6 found it in `website_validator.py`/`quality_control.py` x2; T8 fixed the live `website_validator.py` copy). Fixed: added `files_skipped` counter + warning logs, without changing the deliberately-documented "must not crash delivery" pass/fail contract. |
| C4 | **VERIFIED DEFECT — FIXED (documentation)** | `safety/tool_guardrails.py`'s docstring falsely claimed to be "Called from engine.py._execute_task()" — verified false by repo-wide grep; the guardrail is fully dead (constructed, never read). Fixed the false claim; wiring it in is `[REQUIRES HUMAN REVIEW]`. |

**Residual, surveyed but not fixed — `[REQUIRES HUMAN REVIEW]`:** `command_guard.py`'s
unplugged shell-risk classifier (live path already has an independent `shell=False` barrier);
the entire `orchestrator/plugin/` isolation subsystem (dead, plus a self-reported-trust bypass
if ever wired in); `safety/guardrails.py`'s 589-line `ProductionGuardrails`/`KillSwitch` (zero
callers, zero tests); `gateway/run.py`'s unauthenticated `handle_message()` (unreachable today —
no real network listener exists); `AgentSafetyMonitor` (same dead-field shape as C4).
**Cleared, resolving T7's original question:** `safety/sandbox.py`/`safety/secure_execution.py`
are plain re-export shims with zero divergence risk — not independent implementations.

**Gate status at T9 close:** black/ruff/lint-imports/root-freeze/test-markers/bandit all PASS.
mypy: 0 new errors introduced by this tier's own edits (8 errors surfaced in
`reference_monitor.py`/`specs.py` are pre-existing, now-reachable via C1's shim, out of scope).
New tests 5/5 (RED→GREEN verified). Full suite: 2522 passed (+5), 2 pre-registered environmental
failures unchanged, 21 skipped — zero regressions.

## T10 — Cost-optimization remainder (closed)

Full detail: `docs/hunts/t10-cost-optimization/inventory.md`, `docs/hunts/t10-cost-optimization/coverage.md`.
Second of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `orchestrator/token_budget.py` was an unshimmed, byte-identical duplicate of `infrastructure/token_budget.py` — caught before divergence rather than after. Fixed: converted to a shim. Not the same feature as `cost_optimization/token_budget.py` (different class, different purpose, coincidental filename). |
| C2 | **VERIFIED DEFECT — FIXED** | `orchestrator/provisioned_throughput.py` was likewise an unshimmed, byte-identical duplicate of `operations/provisioned_throughput.py` — the module `infrastructure/provisioned_throughput.py` already shims to, establishing `operations/` as canonical. Fixed: converted to a shim, matching direction. |
| C3 | **VERIFIED DEFECT — FIXED** | `cost_optimization/cost_optimization_integration.py::Tier1OptimizationMixin` used a same-package-depth relative import (`from .log_config import get_logger`, one dot short), raising `ModuleNotFoundError` unconditionally — the same "wrong depth" shape found at least five times this hunt. Fixed (plus two TYPE_CHECKING-only imports with the same bug). |
| C4 | **VERIFIED DEFECT — FIXED** | `cost_optimization/docker_sandbox.py::DockerSandbox.execute()` wrote caller-supplied `code_files` filenames into the sandbox workspace with no path-containment check — an absolute path or `../` traversal could write outside the sandbox onto the host filesystem, before the container even starts. Dead today (`CodeExecutor`, its only construction site, has zero live callers). Fixed: resolves and validates containment, rejecting escapes. Did not add separately-flagged missing container hardening (`read_only`/`cap_drop`/etc.) — `[REQUIRES HUMAN REVIEW]` if ever wired live. |

**Residual, surveyed but not fixed:** the `--tdd-first` CLI flag's silent no-op (a concrete
symptom of T1's already-recorded C3, not new); `pricing_cache.py`'s architectural fragmentation
(a 4th/5th independent pricing mechanism). **Cleared:** `token_optimizer.py` (correct live shim);
`rate_limiter.py` whole-module reachability (extends T5/T8); `costing/analytics.py` + 8 other
dead `cost_optimization/` files (extends T1's C6 per-file).

**Handoff findings, out of this tier's scope, not fixed here:** `application/cache_warmup.py`
(wave T12) calls a nonexistent `operations/cache_warmup.py` module from the real parallel-task
hot path in `engine.py` — silently disables prompt-cache warming on every run, misreported as
"non-critical" (a missed cost/latency optimization, not a correctness defect). `integrations/mcp_server.py`
(wave T15) has a broken import for its `TokenOptimizer` reference.

**Gate status at T10 close:** black/ruff/lint-imports/root-freeze/test-markers/bandit all PASS.
mypy: isolated diff empty (this tier's files aren't reached by the core-path invocation). New
tests 5/5 (RED→GREEN verified, 4/5 defect-proving + 1 no-regression check). Full suite: 2527
passed (+5), 2 pre-registered environmental failures unchanged, 21 skipped — zero regressions.

## T11 — Infrastructure / engine_core / domain / events core (closed)

Full detail: `docs/hunts/t11-core-architecture/inventory.md`, `docs/hunts/t11-core-architecture/coverage.md`.
Third of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`; largest wave to date (118 files).

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `config/costs.json` had no entry for `qwen/qwen3.6-flash` — a real, actively-routed model — so `infrastructure/llm_client.py`'s live cost-computation path silently priced every real call at $0.00. The survey's initial claim of "5 affected models" was independently corrected during verification: 3 of the 5 referenced deprecated, commented-out (non-live) enum members that can never actually be constructed. Fixed the one genuine gap (plus a placeholder internal-only model for consistency). Highest-severity live defect found in this hunt to date. |
| C2 | **VERIFIED DEFECT — FIXED** | `engine_core/project_planner.py::get_execution_levels()` — the live task-scheduling method every project run goes through — silently dropped circularly-dependent tasks with zero diagnostic. Partially mitigated already (`StateCoordinator` correctly downgrades to `PARTIAL_SUCCESS`) but named nothing. Fixed: logs an ERROR naming the exact dropped task IDs, without changing the existing return behavior. A separate, broken `container.py` import that would have wired in real cycle-detection instead of this silent fallback was left unfixed (`[REQUIRES HUMAN REVIEW]`, an architecture decision). |
| C3 | **VERIFIED DEFECT — FIXED** | `engine_core/pipeline_executor.py`'s retry loop didn't recognize `"vs_retry_escape"` (set by `stages/self_consistency.py`'s own verbalized-sampling diversity-escape feature) as a retry signal, so the escape it had just configured never ran — live whenever the `vs_retry_escape` flag is on (which `search_strategy_tuner.py` can autonomously enable). Fixed: added the missing value to the recognized tuple. |

**Residual, surveyed but not fixed:** an unsynchronized circuit-breaker HALF_OPEN race in
`infrastructure/streaming_resilient.py` (same class as T5's fix, confirmed fully dead);
`unified_events/core.py::UnifiedEventBus` never calling `.start()` (dead projections, unbounded
queue growth — a bigger wiring decision than this tier's fixes); `domain/model_registry.py`'s
stale `QWEN_3_6_FLASH` entries (dead, lower priority than C1's live gap); `engine_core/dep_resolver.py`
vs `dependency_resolver.py` investigated as a possible duplicate pair, found NOT to be one (genuinely
different, unrelated responsibilities — the latter is a dead, harmless orphan shim).

**Discovered incidentally, out of scope, not fixed:** running the repo's own config-drift
diagnostic while verifying C1 surfaced a large, separate `routing.json` drift condition (many
task-type/model keys silently dropped due to enum mismatches) — large enough to warrant its own
future investigation, not folded into T11.

**Gate status at T11 close:** black/ruff/lint-imports/root-freeze/test-markers/bandit all PASS.
mypy: isolated diff empty. New tests 3/3 (RED→GREEN verified). Full suite: 2530 passed (+3), 2
pre-registered environmental failures unchanged, 21 skipped — zero regressions.

## T12 — application/ orchestration core, agents/, supervisor/, nash/, meta/ (closed)

Full detail: `docs/hunts/t12-application-core/inventory.md`, `docs/hunts/t12-application-core/coverage.md`.
Fourth of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` (93 files).

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `application/decomposer_service.py`'s Instructor fast-path decomposition made a real, billable OpenRouter call via a raw client (bypassing `UnifiedClient`'s cost tracking) but never called `charge_fn` — live whenever a project description is ≤8,000 chars, the common case. Fixed: charge a token-count estimate via the existing `models.py::estimate_cost()` helper, mirroring the defensive try/except style the fallback path already uses. Does not claim the estimate matches Instructor's real reported cost — a full fix requires restructuring `structured_outputs.py` (outside any tier's scope). |
| C2 | **VERIFIED DEFECT — FIXED** | 7 wrong-depth relative imports across `engine_core/method_selector.py` (4), `container.py` (2), `engine_deps.py` (1) made the entire ARA reasoning-pipeline subsystem (~5,000 lines, 22 reasoning strategies, enabled by default) unreachable, silently swallowed by bare `except ImportError`. Fixed: corrected all 7 to point at their actual targets (`reasoning/ara_pipelines.py`, root `ara_integration.py`/`ara_execution_strategy.py`). Verified end-to-end (real construction, `enabled=True`) and against the pre-existing 26-test container/ACR suite. Cross-tier fix: `engine_core/` was T11's closed scope, but the defect is only traceable from a T12 file and the fix is a pure dot-count correction — documented rather than artificially deferred. |
| C3 | **VERIFIED DEFECT — FIXED** | `routing/__init__.py` imported a `routing/selector.py` that has never existed in this repo's history, breaking the whole `routing/` package and, transitively, `model_routing.py` — the file CLAUDE.md's own architecture table names for LLM routing. Fixed: removed the broken import line; did not invent a replacement target. |
| C4 | **VERIFIED DEFECT — FIXED** | `meta/integration.py` had diverged from its canonical root twin `meta_integration.py`: the subpackage copy still read `state.status.value` directly (AttributeError if `status` is a plain string), a bug the root copy already fixed via `getattr`. Fixed: converted `meta/integration.py` into a shim pointing at root — the reverse direction from this codebase's other 5 `meta_*.py` shims, since root is provably canonical here. Confirmed no circular import, both empirically and by tracing. |
| C5 | **VERIFIED DEFECT — FIXED** | `application/cache_warmup.py` imported a nonexistent `operations/cache_warmup` module (T10 handoff). T10's "live hot-path" characterization is corrected here: `git log -S` confirms its only caller, `engine._warm_cache_for_level`, has never itself had a caller — fully dead code, not a missed hot-path optimization. Fixed the import to the real `cost_optimization/prompt_cache.py::warm_prompt_cache`; did not wire the dead call site live (a product decision). |

**Residual, surveyed but not fixed:** `HumanInTheLoop`'s fail-closed gate never reaching
`ProjectRunner` (`project_runner.py:232`'s `self._hitl` is never set anywhere), so
`UnattendedGuard` always reports no checkpoint present regardless of real configuration —
`[REQUIRES HUMAN REVIEW]`, a safety-gate behavior change not independently verified as
risk-free; `orchestrator/agents.py` permanently shadowed by the `orchestrator/agents/`
package (currently inert, needs a rename decision, not a mechanical fix);
`application/task_executor.py`'s missing `await` on an async call (real, but inside code
with zero live callers); `delegation/batch_runner.py`'s dead-code docstring-contract
violation (drops failed tasks from its return dict, contradicting its own docstring).

**Discovered incidentally, out of scope, not fixed:** `commands/nash.py`'s `nash backup`
subcommand crashes unconditionally (`orchestrator/nash_backup.py` has never existed) — handed
to T15 (commands/ is its declared scope); `cli.py`'s docstring names a nonexistent dispatcher
module (harmless, real import is already correct) — handed to T15 (cli* is its scope); a
broad structural pattern where roughly a third of `application/` plus most of
`agents/`/`planning/` are real, individually-tested, but never wired into production —
recorded for visibility, not assigned to any future wave.

**Gate status at T12 close:** black/ruff/lint-imports/root-freeze/test-markers/bandit all
PASS. mypy: isolated diff empty after sorting (raw diff was pure ordering noise — identical
1879-line counts both sides). New tests 5/5 (RED→GREEN verified). Pre-existing 26-test
container/ACR/engine/resilience suite: zero regressions. Full suite: 2535 passed (+5), 2
pre-registered environmental failures unchanged, 21 skipped — zero regressions.

## Out-of-band — `config/routing.json` config drift (resolved, not a numbered tier)

T11 flagged "a large routing.json config-drift condition... large enough to warrant its own
dedicated investigation" but explicitly out of scope for that tier. Resolved here, between
T13's dispatch and close, using two artifacts the user supplied directly: a live OpenRouter
`/models` catalogue snapshot (431 models) — this sandbox has no outbound network access to
`openrouter.ai`, so this let `tests/unit/test_openrouter_model_audit.py`'s snapshot-based
check run for real instead of skipping (saved to the gitignored
`scripts/openrouter_models_snapshot.json`, confirmed clean: 0 dead ids, 0 stale
replacements against 628 live ids) — and the actual V7 protocol source document, confirming
`docs/DEFECT_HUNT_PLAN.md`'s operationalization of it needed no correction. Re-running
`.claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py` (unrelated
to the snapshot — a pure JSON-vs-enum check) surfaced this finding.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `config/routing.json` had 21 dead top-level keys shaped like `{model_id: [task_type, ...]}` — the inverse of the file's real `{task_type: [model_id, ...]}` shape. `models.py::_build_routing_table`'s `if k in TaskType._value2member_map_` guard (verified directly in the production code, the exact mechanism `check_config_drift.py` simulates) silently drops any key that isn't a real `TaskType` value, so all 21 were always inert. Of the 50 (model, task_type) pairs those dead keys carried, 44 duplicated a pairing already present in the correct list — harmless clutter; 6 did not, and were the actual defect: `meituan/longcat-2.0`, `google/gemini-3.5-flash-lite`, and `anthropic/claude-opus-5-fast` silently missing from `code_review` (the latter two also missing from `creative_writing`), and `z-ai/glm-5.1` missing from `code_generation` — all four are live, routable models per the supplied catalogue. Fixed by adding the 6 pairs to their correct lists and deleting all 21 dead keys (13 clean `TaskType`-keyed entries remain). |

**Discovered incidentally, out of scope, not fixed:** `check_config_drift.py`'s Direction-B
(informational) output also lists 7 `Model` enum values with no `costs.json` entry — all
`:free`-suffixed variants (e.g. `qwen/qwen3-coder:free`), where a missing entry may be
intentional (free tier, $0 is arguably correct) rather than a defect like T11's C1. Not
investigated further; flagged for whoever next touches `costs.json`.

**Gate status:** black/ruff/root-freeze/test-markers/bandit all PASS. `check_config_drift.py`
now exits 0 (was exit 1, 71 hard-drift lines). New tests 8/8 (RED→GREEN verified — 7 failed
pre-fix for the exact predicted reason, 1 no-regression check passed both sides). Targeted
regression (`-k "routing or model_selector or planner"`) 67/67 pass, zero regressions.

## T13 — generators/, appbuilder/, design/, scaffold/, output/, quality/ (closed)

Full detail: `docs/hunts/t13-generators-design-quality/inventory.md`, `docs/hunts/t13-generators-design-quality/coverage.md`.
Fifth of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` (122 files). Framed by
the plan itself as lower architectural blast radius than T9-T12 (generated-output quality,
not orchestrator integrity/security).

**Lead verdict:** `generators/website_validator.py`'s T6-flagged false-clean secret scan is
already fixed (by T8) and reconfirmed intact — no action needed.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `quality_control.py::TestRunner._run_security_checks` silently reported "No security issues found" for files it couldn't read (`except Exception: pass`) — the third instance of this hunt's false-clean-scan pattern (after T8's `website_validator.py`, T9's `generated_output_scanner.py`). Confirmed dead today (`TestLevel.SECURITY` never requested by either real caller of `run_quality_gate`), fixed anyway per this hunt's established "cheap fix even in dead code" precedent: narrowed to `except OSError`, added a skip counter + warning log, folded into the pass/fail result. |
| C2 | **VERIFIED DEFECT — FIXED** | `orchestrator/quality/quality_control.py` was an unshimmed, byte-for-byte duplicate of the live root `quality_control.py` (import-depth comments aside), carrying C1's identical bug. Confirmed zero live importers. Converted to a shim. |

**Residual, surveyed but not fixed — reclassified from the survey's own "highest impact this
wave" claim:** `design/component_registry.py` imports `ComponentSource`/`ComponentSpec` from
`.design_system` — two classes that have **never existed anywhere in this repository's
history** (confirmed via `git log -S` across the whole repo). The survey correctly traced the
consequence (generated websites always fall back to 4 generic placeholder components instead
of the curated multi-source library) but framed it as a silent, undiscovered defect.
Independent verification of `generators/website_generator.py` found this is instead a
**previously-known, explicitly documented, deliberate workaround** (`# FIXED: ... Lazy
import — component_registry has broken dependencies`) — a prior developer already identified
this exact gap and chose graceful degradation. Completing it means designing a
compatibility-scoring algorithm from scratch, a product decision `[REQUIRES HUMAN REVIEW]`,
not a mechanical fix. Also residual: `output/organizer.py` (dead, missing two pipeline steps
the live root copy has); `docker_generator.py`'s root/subpackage divergence (hardcoded
default DB credentials vs. hardened generation, both dead today, security-adjacent); a
complete but entirely unused CSP/CSRF generation library (`design/frontend_security.py`); a
fully dead legacy import chain (root `website_generator.py` → nonexistent
`component_registry.py` → `cli_website.py`); a `CodebaseAnalyzer` naming collision between two
independent classes (`[UNK]`).

**Gate status:** black/ruff/lint-imports/root-freeze/test-markers/bandit all PASS. mypy:
isolated diff empty. New tests 5/5 (RED→GREEN verified). Targeted regression
(`-k "quality_control or quality_gate"`) 27/27 pass. Full suite: 2549 passed (+14 over the
prior commit's 2535 — 13 new hunt tests across T13 and the out-of-band routing.json fix, +1
test newly un-skipped by the user-supplied OpenRouter snapshot), 2 pre-registered
environmental failures unchanged, 20 skipped (was 21), 157 deselected — zero regressions.

## T14 — learning/, knowledge/, nexus_search/, pattern_learner/, context_mgmt/, analysis/ (closed)

Full detail: `docs/hunts/t14-learning-knowledge-search/inventory.md`, `docs/hunts/t14-learning-knowledge-search/coverage.md`.
Sixth of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` (71 files). Framed as
"auxiliary intelligence/retrieval subsystems — real but not on the critical execution path."

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `knowledge/knowledge_base.py`'s shim imported a nonexistent `KnowledgeEntry` name (root defines `KnowledgeArtifact`, not `KnowledgeEntry`). Converted to a wildcard shim. |
| C2 | **VERIFIED DEFECT — FIXED** | `knowledge/knowledge_graph.py`'s shim imported two nonexistent names (`ModelPerformanceGraph`, bare `KnowledgeGraph` — root defines `PerformanceKnowledgeGraph`). Converted to a wildcard shim; transitively fixed `knowledge/docs_generator.py`'s broken import too. |
| C3 | **VERIFIED DEFECT — FIXED** | `learning/federated_learning.py` was an unshimmed duplicate of the live root file whose stripped import (`# REMOVED: from .feedback_loop import ...`) made `contribute_insight()` raise `NameError` on `OutcomeStatus` the moment it ran. Confirmed dead (real caller uses root). Converted to a shim. |
| C4 | **VERIFIED DEFECT — FIXED** | `learning/transfer_learning.py`, same shape as C3, but the stripped import was justified by a confidently-wrong comment ("meta_orchestrator types removed (module does not exist)") — `orchestrator/meta_orchestrator.py` exists and defines all four names root still imports. Converted to a shim. |
| C5 | **VERIFIED DEFECT — FIXED** | `performance.py::QueryOptimizer.build_selective_query()` built raw SQL via unvalidated f-string interpolation of `table`/`columns`/`order_by` — the third instance this tier of "fix landed on the dead copy of an unshimmed duplicate, never backported": `analysis/performance.py`'s copy had already been hardened (allowlist + identifier validation) but the fix never reached the live root copy. Confirmed dead (`QueryOptimizer` has zero callers anywhere). Ported the hardening to root; converted `analysis/performance.py` to a shim now that it's redundant. |

**Residual, surveyed but not fixed:** 7 further duplicate pairs in `analysis/` confirmed
clean today (zero divergence) but still unshimmed — a standing structural risk given 3
confirmed divergences of exactly this shape surfaced in this tier alone; a latent split-brain
`leaderboard.py` singleton (root side currently dead, so no active bug); several
lower-priority informational findings (an unreachable duplicate `except` block in
`nexus_search/nexus_client.py`, a soft silent-skip in `memory_tier.py`'s best-effort file
scan, 4 of 7 `nexus_search/optimization/` modules unwired, an unwired generated-app
RLS-policy generator matching T13's already-documented pattern).

**Gate status:** black/ruff/lint-imports/root-freeze/test-markers/bandit all PASS. mypy:
isolated diff empty. New tests 9/9 (RED→GREEN verified — 8 failed pre-fix for the exact
predicted reason, 1 no-regression check passed both sides). Targeted regression (existing
tests exercising the fixed modules) 86/86 pass. Full suite: 2558 passed (+9), 2
pre-registered environmental failures unchanged, 20 skipped, 157 deselected — zero
regressions.

## T15 — integrations/, vcs/, ide_backend/, dashboard_core/, commands/, cli*, entrypoints/ (closed)

Full detail: `docs/hunts/t15-integrations-cli-dashboard/inventory.md`, `docs/hunts/t15-integrations-cli-dashboard/coverage.md`.
Seventh of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` (82 files). Framed as
directly user-facing (CLI/dashboard surfaces) — 7 fixes, the most of any tier so far, driven
by genuine severity distribution rather than a target.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `Orchestrator(..., verbose=...)` — a kwarg `Orchestrator.__init__` has never accepted — crashed both `entrypoints/chat_cli.py`'s `orchestrator chat` build handoff and `dashboard_core/chat_view.py`'s live `/ws/chat` websocket feature. Removed the kwarg from both call sites (it was never read or used anywhere). |
| C2 | **VERIFIED DEFECT — FIXED** | The `dashboard` console script (one of only 3 this project ships) was broken on every invocation: `dashboard.py`'s shim aliased `run_dashboard` to a zero-arg function while `cli_dashboard.py` called it with `host`/`port`/`open_browser` kwargs. Rewired the shim to the real, correctly-parameterized `dashboard_core.core.run_dashboard`; dropped the never-implemented `open_browser`/`--no-browser` option rather than leaving it as a second silently-ignored flag. |
| C3 | **VERIFIED DEFECT — FIXED** | `commands/kanban.py`'s two lazy imports used the wrong relative-import depth (`.kanban.board` instead of `..kanban.board`, since `commands/kanban.py` is a flat module, not a package), breaking all four kanban subcommands and stranding a fully-built 441-line SQLite work-queue subsystem. Fixed both imports; verified `kanban list` now runs end-to-end. |
| C4 | **VERIFIED DEFECT — FIXED** | `commands/gateway.py`, identical bug shape (`.gateway.run` → `..gateway.run`), broke both gateway subcommands. Fixed; verified `gateway status` now runs end-to-end. |
| C5 | **VERIFIED DEFECT — FIXED** | `commands/codebase.py`'s `--budget` flag was defined but never read — `modify` always charged against a hardcoded $10 regardless of user input. Threaded `args.budget` through to the actual `Budget(max_usd=...)` construction. |
| C6 | **VERIFIED DEFECT — FIXED** | `commands/nash.py`'s `nash backup` crashed with a raw `ModuleNotFoundError` traceback (`orchestrator.nash_backup` has never existed — T13 handoff, exhaustively reconfirmed nothing resembling a backup manager exists anywhere in the repo). Wrapped the import in try/except for a clean, bounded message instead of inventing the unbuilt feature — mirroring the dead sibling `cli_nash.py`'s own defensive handling of the identical broken import. |
| C7 | **VERIFIED DEFECT — FIXED** | `cli.py`'s docstring (T13 handoff) named a dispatcher module (`application.cli_dispatch`) that was renamed to `entrypoints.cli_dispatch` years ago; the real import was already correct. Corrected all 3 references. |

**Residual, surveyed but not fixed:** `integrations/mcp_server.py`'s 5 broken imports (T10
handoff) — confirmed fixable, but it's a feature-richer dead fork of the live root MCP
server, and which side should be canonical is a product decision, `[REQUIRES HUMAN REVIEW]`;
`Orchestrator.modify_codebase()` — referenced by `commands/codebase.py` but never implemented
anywhere in this repository's history (only a design document ever described it), left
unbuilt per this hunt's standing discipline against inventing features, along with the same
command's silent exit-code-0-on-failure (a repo-wide exit-code convention decision, not a
single-file patch); `integrations/compat.py` (4 broken imports, dead) and
`integrations/swiftstack_integration.py` (an unshimmed dead duplicate of the live root file);
a `git_sync.py` (root) vs `vcs/sync.py` divergence (root has a fix `vcs/`'s copy lacks — an
unusual direction, since `vcs/` is normally canonical for this codebase's git-adjacent
shims); a dead, internally-inconsistent 7-file "command center" cluster; several
cosmetic/minor findings (a stale CLAUDE.md `--analyze-codebase` example, a typo'd deprecation
warning, 3 misplaced `test_*.py` files `pytest tests/` never collects).

**Gate status:** black/ruff/lint-imports/root-freeze/test-markers/bandit all PASS. mypy:
isolated diff empty. New tests 7/7 (RED→GREEN verified — 6 failed pre-fix for the exact
predicted reason, 1 no-regression check passed both sides). No existing test in the repo
touched any of the 9 fixed files. Full suite: 2565 passed (+7), 2 pre-registered
environmental failures unchanged, 20 skipped, 157 deselected — zero regressions.

## T16 — operations/ remainder, verification/, policy*, telemetry/logging misc (closed)

Full detail: `docs/hunts/t16-operations-verification-remainder/inventory.md`,
`docs/hunts/t16-operations-verification-remainder/coverage.md`. Eighth and **final** wave of
`docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` (85 files: the plan's 75-file estimate plus
`project_mgmt/`/`workspace/`, the same duplicate-pair shape this hunt targets). Included two
mandated deep-dive leads carried from earlier tiers: the SecretsFilter-installation residual
(T8/T9/T13) and CLAUDE.md's "policy system not fully integrated" claim.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `application/cli_helpers.py::setup_logging()` — the actual live CLI logging entry point, called on every invocation — never attached `generators/secrets_manager.py`'s `SecretsFilter`, unlike `log_config.py::configure_logging()` (fixed in T8), which has zero live callers. The masking safety net was built and correctly wired to the wrong, unreachable function; ~100 files' log records (including the live LLM-call error path) propagated unmasked. Fixed by mirroring `log_config.py`'s own correct pattern on the real entry point. |
| C2 | **VERIFIED DEFECT — FIXED** | `testing/validator.py::TestValidator._generate_test()` called `self.client.call_model(...)`, a method `UnifiedClient` has never had (only `.call()` exists). The `AttributeError` was silently caught, falling back to a trivial stub that gets written to disk, run, and reported as a passing generated test — a false-positive, not just a skip. Fixed the method name; the flag gating this class's own construction remains separately, architecturally unwired (not fixed — matches this hunt's wiring-gap pattern). |
| C3 | **VERIFIED DEFECT — FIXED** | `project_mgmt/analyzer.py` was a stale, unshimmed duplicate of the live `project_analyzer.py`, missing the real `ArchitectureScorer` delegation and two whole methods (`save_report()`, `print_suggestions()`) root gained since. Converted to a shim. |
| C4 | **VERIFIED DEFECT — FIXED** | Four `operations/` duplicate pairs (`hitl_workflow.py`, `concurrency_controller.py`, `deployment_feedback.py`, `memory_tier.py`) were unshimmed, dead-on-the-`operations/`-side forks of live root modules; two diverge functionally (root carries an asyncio task-reference fix and an SSRF guard the dead copies lacked). All four converted to shims. |
| C5 | **VERIFIED DEFECT — FIXED** | `services/executor.py`, `services/generator.py`, `services/observability.py` independently redefined classes `services/__init__.py` already re-exports from `application/` as canonical — 8 existing test files importing the submodule path directly were silently exercising the shadowed copy, not the one production code runs. Converted all three to shims (aliased for `generator.py`, whose real name is `DecomposerService`); verified by running all 73 tests across the 8 affected files against the new shims — zero regressions. |
| C6 | **VERIFIED DEFECT — FIXED** | `crosscutting/config.py`'s re-export of `orchestrator/config.py` constants always silently `ImportError`'d (the names never existed there) and fell back to stale values, including a $10 default that didn't match the real $8. Fixed to import the real namespaced attributes. |
| C7 | **VERIFIED DEFECT — FIXED** | `operations/quick_self_test.py` executed a full ad-hoc integration test at import time — writing a log file into the source tree and calling `sys.exit(1)` on failure at module scope. Independently discovered twice (a stop-hook untracked-file check, and this tier's own survey). Gated behind `if __name__ == "__main__":`; log destination moved out of the source tree. The script's own self-test target (a module retired years ago) was deliberately left non-functional rather than rewired to a guessed replacement. |

**Residual, surveyed but not fixed:** the policy system (`policy_engine.py::enforce()`/
`.check()`, `ConstraintPlanner.select_model()` et al.) has **zero live enforcement on any
entry point** — `engine.py::run_job()`'s own docstring claims compliance is "enforced on
every API call," which is false; a caller can pass a real `PolicySet` and get silent no-op
enforcement with no audit trail. Escalates CLAUDE.md's current one-line "not fully
integrated" note to a plain statement that enforcement is completely dead everywhere,
`[REQUIRES HUMAN REVIEW]` (which entry point should gate it is a product decision);
`operations/autonomy_config.py`'s Multi-Mode Selector is fully built and completely inert,
bypassed by an unrelated ad-hoc mapping in 4 `cli_dispatch.py` blocks, `[REQUIRES HUMAN
REVIEW]`; `orchestrator/verification.py`'s permanent shadowing by the `verification/`
package (same shape as T2's already-recorded `gateway.py`/`agents.py` instances, left
un-modified per that precedent); `orchestrator/logging.py`'s dead structlog
`configure_logging()` fork (zero callers); `ShellTool`'s shell-execution surface (by design,
zero live construction); `services/completion_judge.py`/`autonomy_costs.py` (dead code, no
duplicate to reconcile); `memory_tier.py`'s shared silent-file-skip bug (T14 already
evaluated and declined to elevate this in an already-closed tier — not reopened here, only
the newly-found duplicate-pair hygiene issue around it was fixed).

**Gate status:** black/ruff/lint-imports/root-freeze/test-markers/bandit all PASS. mypy:
isolated diff shows only *removed* errors (1571 → 1554, zero new). New tests 7/7 (RED→GREEN
verified — all 7 failed pre-fix for the exact predicted reason). The 8 pre-existing test
files affected by C5's shim conversion (73 tests) re-run directly against the new shims:
zero regressions. Full suite: 2572 passed (+7), 2 pre-registered environmental failures
unchanged, 20 skipped, 157 deselected — zero regressions.

This closes the T9-T16 continuation plan. As stated in
`docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` from the outset, the 8-wave decomposition was a
data-backed, estimate-based prioritization of the files left with no individual disposition
after T0-T8 — not a mathematically-verified 100% line-by-line partition of every file in
scope.

## T17 — Duplicate-pair convergence sweep (closed)

Full detail: `docs/hunts/t17-duplicate-pairs/inventory.md`,
`docs/hunts/t17-duplicate-pairs/coverage.md`. First wave of
`docs/hunts/BACKEND_DEPTH_PASS_PLAN.md` (T17–T24) and the first wave in this programme
whose Phase-1 surface is **exhaustively enumerated over a pattern** rather than sampled
over files: all 188 same-name root/subpackage pairs classified by AST, of which 123 were
already resolved as shims in one direction or the other, leaving **65 with definitions on
both sides** — the full candidate set, all triaged.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `integrations/swiftstack_integration.py` was a *byte-identical* copy of the root module one package deeper, so its root-relative imports resolved against `orchestrator.integrations.*` and it raised `ModuleNotFoundError: No module named 'orchestrator.integrations.api_builder'` on every import — a duplicate that could not load. T15 recorded it as a dead duplicate; this wave establishes it was also broken. Converged to a shim, which also makes it importable. |
| C2 | **VERIFIED DEFECT — RECORDED, NOT FIXED** | Root `website_generator.py:17` imports `.component_registry`, which lives in `design/`, not root — so the module and its only importer, `cli_website.py`, are both unimportable. Deliberately not fixed: correcting the path only moves the failure, because `design/component_registry.py` needs `ComponentSource`, a name referenced 17 times across the design subsystem and **defined nowhere in the repository**. A path-only change would mask the symptom rather than break the mechanism. `[REQUIRES HUMAN REVIEW]`, consistent with T13's disposition of the same subsystem. |
| C3 | **Hygiene — FIXED** | Four further byte-identical duplicate pairs (`design/component_library.py` 879 lines, `design/frontend_security.py` 1058, `security/indesign_plugin_rules.py` 1131, `security/ios_hig_prompts.py` 474) kept two independently-editable copies reachable under two import paths. No divergence today by construction; converged to shims to remove the precondition for one. |
| C4 | **VERIFIED DEFECT (latent) — FIXED** | `design/design_system.py` was a stale 155-line fork of the 310-line canonical root module, **wildcard-exposed via `design/__init__.py`**, missing the `tone`/`font_heading`/`font_body`/`accessibility` fields and the `__post_init__` materialising `spacing`/`shadow`/`animation`/`border_radius` — precisely the attributes `website_generator.py` formats into its output. Latent rather than live: every real consumer imports root, so the fork was an `AttributeError` landmine armed for the first caller writing `from orchestrator.design import DesignSystem`. mypy independently reported `"DesignSystem" has no attribute "tone"` pre-fix; that error is gone post-fix. |
| C5 | **FALSE (innocent)** | `design/design_to_code.py`'s `VISION_MODELS` names a stale `claude-sonnet-4.6` (root names the real `claude-sonnet-5`), but the dict is read by nothing anywhere. Inert constant in a dead fork. Recorded so it is not re-raised. |
| C6 | **UNKNOWN — routed to T21** | `ide_backend/log_config.py` defines a fifth independent `configure_logging()` and a `get_logger()` returning a separate `ide_backend.*` logger hierarchy with no `SecretsFilter`; five `ide_backend/` modules import it. No executable trigger is possible here — importing it requires `fastapi`, which is not installed — so per V7 Phase 3 it stays `[UNK]` rather than being promoted on reasoning. Strengthens the case for T21's stated prerequisite. |

**Deliberately excluded from convergence:** `integrations/gateway.py` — byte-identical and
name-safe, but `orchestrator/gateway/` exists as a *package*, so a `..gateway` shim would
resolve to the package rather than root's `gateway.py`. That is a semantic change, not a
convergence. Kept in the gate baseline.

**Triaged, not converged (57):** overwhelmingly import-depth-only differences (`..X` vs
`.X`, each correct for its own location) with leftover `# FIXED:` breadcrumb comments.
Converging all 57 in one commit is the over-broad change V7 Phase 6 vector 6 warns against;
each needs its own name-superset proof. Frozen in the gate baseline so they cannot grow.

**New gate — `scripts/check_duplicate_pairs.py`:** freezes the remaining 59 pairs, fails on
any new both-sides-define pair, accepts a pair once either side becomes a shim, supports
`--list`/`--update`. Self-tested three ways (detects an injected pair, accepts it once
shimmed, `--update` idempotent and non-self-corrupting). Two defects in the gate itself were
caught by its own tests before commit.

**Gate status:** black/ruff/lint-imports/root-freeze/duplicate-pairs/test-markers/bandit all
PASS. mypy: net −17 errors (1554 → 1537); the 47 lines that appear on root `design_system.py`
are pre-existing looseness in a file this wave did not modify, which mypy simply never
reached before (before = 0 lines for it) because `component_registry` resolved `DesignSystem`
to the fork. New tests 8/8 (RED→GREEN verified — all 8 failed pre-fix for the exact predicted
reason). Full suite: 2580 passed (+8), 2 pre-registered environmental failures unchanged,
20 skipped, 157 deselected — zero regressions.

## T18 — Silent-failure sweep at scale (closed)

Full detail: `docs/hunts/t18-silent-failure/inventory.md`,
`docs/hunts/t18-silent-failure/coverage.md`. Second wave of
`docs/hunts/BACKEND_DEPTH_PASS_PLAN.md`. AST-classified **all 916** broad
exception handlers in `orchestrator/` (the plan estimated 923 from a cruder grep;
the AST count supersedes it): 627 log, 67 re-raise, **222 silent**. The 222 were
ranked money (7) / validation-gate (46) / persistence (24) / config (3, not
elevated per T16) / other (142).

**The headline is a negative result.** Two mechanical sweeps tested *every* one of
the 916 handlers for the **fail-open** shape — a handler that neither logs nor
re-raises, returning either a bare `True` or a result object built with
`passed=True`/`success=True`/`ok=True`/`valid=True`/`healthy=True`/`available=True`.
**Both returned zero.** No handler in the codebase reports success because it
failed. That is the severe variant of this pattern, and its absence is consistent
with T6/T8/T9/T13/T16 having fixed the instances that existed.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `website_validator.py::_check_rate_limiting` silently `continue`d past any of its ≤50 scanned files it could not read, then reported *"No rate limiting found. Contact forms and registration endpoints must include IP-based rate limiting."* — indistinguishable from having read the files and found nothing. Now logs the read failure and appends "N file(s) could not be read and were skipped, so this scan is incomplete", matching T8 C5's pattern in this same file. |
| C2 | **VERIFIED DEFECT — FIXED** | `website_validator.py::_check_auth_flow`, identical shape over ≤20 auth files, reported *"Auth pages found but no email verification flow detected."* Same fix. |
| — | **Severity note** | Unlike T8's C5 in this file (the secret scanner, which reported **clean** — fail-open, a security hole), C1/C2 fail **closed**: an unreadable file leaves `found=False` so the check reports failure. The harm is a misreport that sends a developer to add protection that may already exist, plus silently reduced coverage of checks gating the documented `--min-quality`/`--require-all-checks` flags. Severity LOW, and deliberately not inflated to match T8's. |
| C3 | **FALSE (innocent)** | `cost.py::_static_estimate`'s `except (KeyError, Exception): return 0.0` looked like it would make unknown-cost models appear free and therefore always "cheapest". Falsified on both counts: `estimate_cost` cannot raise (`COST_TABLE.get(model, {...})` has a default), and `cheapest_model` **filters out** zero-cost candidates rather than preferring them. The 0.0 is a documented sentinel. The redundant `(KeyError, Exception)` tuple is inert. |
| C4 | **FALSE (innocent)** | `safety/code_executor.py::_is_sandbox_available`'s silent `return False`, whose caller falls through to `_execute_local` (commented "insecure"), looked like a sandbox bypass. The guard above it either returns a blocking error result or logs `"Executing code without sandbox - security risk!"` — fail-closed by default with an explicit, logged opt-out. Swallowing a Docker-reachability error is correct. |
| C5 | **FALSE (innocent)** | `infrastructure/state.py::save_checkpoint`'s flagged `except Exception: pass` guards `await db.rollback()` inside an outer handler that **re-raises**. A rollback that fails must not mask the original exception. |
| C6 | **Recorded, not elevated** | `control_plane.py::_write_audit` (documented-deliberate, comment says "audit failures must never break the main flow", and the record is a log line not a durable store); `streaming_resilient.py::get_usage_percent` (returns a fabricated 50% when the memory probe fails — same shape as T6's fabricated-neutral-score item, treated consistently); `batch_client.py`'s bounded polling swallow. |

**New gate — `scripts/check_silent_failure.py`:** fails when a broad handler that
neither logs nor re-raises returns an affirmative value; prints the number of
handlers examined so the denominator is visible. Deliberately does *not* police
silence in general — 222 handlers would be a meaningless CI signal, and the
classifier's false-positive rate for *severity* is high (it flags correct rollback
guards, best-effort cleanup, and documented-deliberate swallows). Self-tested three
ways: catches `return True`, catches `return R(passed=True)`, and correctly does
**not** flag a handler that logs before returning True.

**Gate status:** black/ruff/lint-imports/root-freeze/duplicate-pairs/silent-failure/
test-markers/bandit all PASS. mypy: isolated diff empty. New tests 4/4 (RED→GREEN —
both defect tests failed pre-fix quoting the misleading text verbatim; the two gate
tests pass on both trees and are labelled tool tests, not defect proofs). Full
suite: 2584 passed (+4), 2 pre-registered environmental failures unchanged, 20 skipped,
157 deselected — zero regressions.

---

## T19 — Subprocess/exec argument construction (closed)

Third wave of the depth pass (`docs/hunts/BACKEND_DEPTH_PASS_PLAN.md`). Shape:
a command string **built by interpolation** and handed to a shell, so a value
from outside the file becomes shell syntax rather than data.

AST census found **108 process-spawning call sites across 43 files**
(`subprocess.*` argv-form 71, `create_subprocess_exec` 25, raw
`compile`/`exec`/`eval` 7, `create_subprocess_shell` 4, `os.system` 1). Only
the shell-interpreting ones can carry this shape, narrowing the sweep to
**8 sites**, each read.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `nexus_search/server_manager.py::start` built `f"{self._docker_compose_cmd} -f {self.compose_file} up -d"` for `create_subprocess_shell`. `compose_file` is the constructor's caller-supplied parameter. A path containing a *space* breaks the command by accident; one containing `;` executes the remainder. Now builds argv for `create_subprocess_exec`. |
| C2 | **VERIFIED DEFECT — FIXED** | `nexus_search/server_manager.py::stop`, identical shape (`... down`). Same fix. |
| C3 | **VERIFIED DEFECT (low) — FIXED** | `scripts/utils/push_to_github.py` ran `f"git push origin {branch}"` under `shell=True` with `branch` from `git branch --show-current`. `git check-ref-format` **accepts** `;`, `$()`, backticks, `&&`, `|` in branch names — only the space is rejected — so a branch named `main;id` executes `id`. Maintainer-local, requires running a dev script on a hostile branch; fixed in two lines. |
| — | **Severity note** | C1/C2 are arbitrary command execution, but reachable only by whoever supplies `compose_file` (today: the default, or an operator argument). Latent, not remotely triggerable — the same disposition as T8's fail-open finding, deliberately not ranked above it. |
| C4 | FALSE (innocent) | `commands/nash.py:117` — `os.system("cls" if os.name == "nt" else "clear")`, both branches literal. |
| C5 | FALSE (innocent) | `dev_server.py:124,152,161` — commands come from a hardcoded `ProjectType` table; the only interpolated value is a port already validated as an int in 1..65535. |
| C6 | FALSE (innocent) | `safety/sandbox_executor.py:76` and `tools/shell_tool.py:35` — running a caller-supplied command *is* each API's documented purpose. Same disposition as `ShellTool` in T18. |
| C7 | FALSE (innocent) | The four remaining `scripts/git/*.py` `shell=True` sites interpolate only hardcoded literal lists. |
| C8 | **FALSE — hypothesis falsified** | Bandit warns `Test in comment: <word> is not a test name or id` for every prose word in the repo's `# nosec B602 — reason` comments. Hypothesis: prose degrades these to *blanket* suppressions hiding unrelated findings. Probed directly — a valid-but-non-matching id is still reported **with** prose (`# nosec B324 — prose`), and only a comment with no valid id at all (`# nosec B999`) falls back to blanket. Targeting survives. Cosmetic warnings, no defect, no change made. |

**Proof.** The RED pass is the wave's strongest evidence: run against the
**unmodified production code**, docker itself reported the argument it had
received as `"compose projects/docker-compose.yml"`, the injected `touch`
created its marker file, and `stop()` left a stray file named `down` in the
working directory — the tail of its own shredded command line. The committed
test's payload was then changed to an inert `echo`, so a future revert proves
the point without touching the filesystem.

**New gate — `scripts/check_shell_injection.py`:** fails any
`create_subprocess_shell` / `shell=True` / `os.system` under `orchestrator/`
whose command is not a literal (constants, literal concatenation, `IfExp` of
literals and `{}`-free f-strings all count). Three reviewed sites allowlisted
with reasons in the source. Verified to catch the bug rather than merely to
pass: re-run with the fix stashed, it reports both `server_manager.py` lines
and exits 1.

**Scope limit (see `t19-subprocess-argv/coverage.md`):** the 100 argv-form
call sites were classified by AST, not read. That is defensible for *this*
shape only — argv cannot shell-inject by construction — and says nothing
about attacker-controlled executable paths, `cwd`, or `env`. The 7 raw
`compile`/`exec`/`eval` sites are a different shape and were not hunted.

---

## T20 — Wiring gaps: registered but never read (closed)

Fourth wave of the depth pass. Shape: something **declared** as a control
surface that no code consults, so setting it appears to work and does nothing.

| Sub-shape | Declared | Never read |
|---|---|---|
| CLI flags (`add_argument`) | 109 | **0** |
| `FeatureFlags` fields | 53 | 8 |
| `OrchestratorSettings` fields | 21 | 12 |

**Detector calibration mattered more than the sweep.** The first CLI census
said 18 flags were never read (it subtracted declaration hits from a count
that never contained them); the second said 0 by matching nearly anything. The
third scopes reads to the argparse namespace and was calibrated against seven
flags confirmed read by hand — 0 false positives — before its answer was
believed. The inverse check produced 15 "undeclared reads", **all false**:
`parsed` is a dict and a `urlparse` result, `options` a config dataclass,
`args` a list and a string; `func`/`meta_cmd`/`subcommand`/`template_action`
come from `set_defaults(func=)` and `add_subparsers(dest=)`. The CLI surface
is clean, and that negative result is reported as one.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `knowledge_rerank_enabled` was declared and read by **no module**, while the feature behind it was complete: `KnowledgeBase.find_similar(rerank=, fetch_k=)` implemented, `LLMReranker` live via `HybridSearchPipeline`, five passing tests. `implementation_plan_reranking.md` lists every step — all done except the last: *"Caller … passes `rerank=flags.knowledge_rerank_enabled`"*. That consumer, `get_recommendations`, has no in-repo callers and was nearly dismissed as dead, but `CAPABILITIES.md:247` and `USAGE_GUIDE.md:1141` document it as the public way to query the knowledge base. A user who set `ORCH_KNOWLEDGE_RERANK_ENABLED=true` got no reranking and no warning. Fixed in 7 lines by taking the plan's last step. |
| C2 | **VERIFIED — ESCALATED** | `bilevel_tabu_enabled` inert *and* `engine_core/tabu_search.py` imported by nothing — the other `tabu_search` hits are string literals in docstring examples. Flag and subsystem both dead; deletion vs. completion is a product call. |
| C3 | **VERIFIED — ESCALATED** | `bilevel_level15_enabled` inert, but its `SearchStrategyTuner` *is* injected into `BilevelAutoresearch`. The flag is redundant, not the feature. |
| C4 | **VERIFIED — ESCALATED** | 12 of 21 `OrchestratorSettings` fields read nowhere (`dashboard_port`, `mcp_port`, `audit_log_path`, `default_budget_usd`, …). With `env_prefix="ORCH_"` and `extra="ignore"`, `ORCH_DASHBOARD_PORT=9000` is accepted in silence and discarded. |
| C5 | **VERIFIED — ESCALATED (worth a second look)** | `dashboard_host: str = "127.0.0.1"` in config.py while `dashboard_core/core.py:347,372` hardcode `host="0.0.0.0"`. The config states loopback-only; the dashboard binds every interface. Also `dashboard_port: int = 8000` vs. the real `8888`. Changing the bind or the setting is a product decision with a security dimension, so nothing was changed unilaterally. |
| C6 | FALSE (innocent) | Six `use_*` flags look dead but their env vars are read directly via `os.getenv`; CLAUDE.md documents them as working, and they are. The field is redundant, not the feature. |
| C7 | FALSE (innocent) | `cache_home` — same shape; `infrastructure/path_provider.py:37` reads `ORCH_CACHE_HOME` directly. |

**New gate — `scripts/check_config_wiring.py`:** freezes the 20 unread fields
and fails on any new one, so the shape can only shrink. Verified by probe:
adding a dead field is reported as `FeatureFlags.t20_gate_probe_flag`, exit 1.

**Scope limit (see `t20-wiring-gaps/coverage.md`):** the census answers "does
anything read this", not "does the *right* thing read it", and covers only
`crosscutting/config.py` — dead settings defined elsewhere are invisible to
both census and gate. `engine_core/container.py` is hand-wired with no
name-keyed registry, so "registered but never resolved" collapses into the
settings sub-shape and was not hunted separately.

---

## T21 — `ide_backend/`, the blocked region (closed)

Fifth wave of the depth pass, and the one region whose blocker was
**environmental rather than budgetary**: 16 files / 5,152 lines that had never
been importable in this environment, leaving its coverage claim the weakest in
the ledger. Cleared with `pip install` of the `dashboard` extra; all 16 modules
then import cleanly (0 failures, verified by exit code).

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `ide_backend/test_color_regex.py` contained **no `assert`** — it counted failures, printed them, and `return failed == 0`. pytest discards a test's return value, so it passed unconditionally. Proven by sabotaging its pattern to match nothing: still `1 passed`. Made fail-closed, **it failed for real** (9 of 10 cases passing, 1 not). The T18 fail-open shape, in the test surface. |
| C2 | **VERIFIED DEFECT — FIXED** | The same file copy-pasted production's regex instead of calling it, so even green it said nothing about `ide_orchestrator_server.py`. Production's substitution extracted as `replace_accent_color()`; the test now drives it. |
| C3 | **VERIFIED DEFECT — FIXED** | `ide_backend/test_server.py` is not a test — it is a FastAPI server on port 8765, and the file `start-ide.bat` launched. Renamed `standalone_server.py`; launcher and archived doc updated. |
| C4 | **VERIFIED DEFECT — FIXED** | 695 lines of `test_*.py` inside `orchestrator/`, where `testpaths = ["tests"]` means pytest never collects them — 13 tests nobody had ever run. |
| C5 | **VERIFIED DEFECT — FIXED** | `test_ide_modifications.py`'s two async tests were written against `ide_orchestrator_server.SessionManager` (sync, has `broadcast`) but imported `session_manager.SessionManager` (async, no `broadcast`) — T17's "landed on the wrong copy" shape, kept invisible by C4. |
| C6 | **VERIFIED — ESCALATED** | Three parallel IDE entry points (`launch.py`→`server.py`; `ide_orchestrator_server.py` at 3,005 lines with its own `SessionManager`; `standalone_server.py`), the latter two both binding port **8765**. Which is canonical is a product decision. |
| C7 | **VERIFIED — ESCALATED** | The `dashboard` extra pins `websockets<13.0`; the **required** `google-genai` needs `>=13.0.0`. pip reports the conflict. Both still import under 12.0, so nothing breaks at import; the Gemini Live-API websocket path is UNKNOWN and untested. |
| C8 | **FALSE — hypothesis falsified** | `grep` showed two `app = FastAPI(` and two `__main__` blocks on different ports, reading as a second app shadowing the first and discarding its routes. AST found exactly **one** module-level `app`: line 2056 is inside a triple-quoted string — a project *template* the server emits. Checking with AST rather than grep is what saved this from being reported. |
| C9 | **FALSE — hypothesis falsified** | Production calls `update_session`/`get_session` **without `await`** and broadcasts the result — the exact shape of C5's `'coroutine' object has no attribute` failure. But Subsystem B's `SessionManager` is synchronous except `broadcast`; the calls are correct. |
| C10 | Recorded, not elevated | The deleted `test_broadcast_order_session_state_first` was **tautological**: it replaced `broadcast` with a recorder, called `broadcast` three times itself, and asserted those calls arrived in the order just made. No production code ran; it could never fail for a product reason. |

**Consolidation, stated honestly.** `test_ide_modifications.py` was deleted
rather than repaired: of its 12 tests, 9 exercised Python's own `re` module, 1
was tautological, and 2 pointed at the wrong class — its coverage of product
code was **zero**. What was worth keeping is now 12 parametrized tests in
`tests/unit/test_ide_color_regex.py` that call production, plus a file
round-trip test. No product coverage was lost. The *claimed* coverage of
broadcast ordering is gone, but it never existed; it is now recorded as an
uncovered property instead of a false pass.

**New gate — `scripts/check_test_placement.py`:** fails any file under
`orchestrator/` that pytest would collect by name **and** that defines
module-level tests. Deliberately narrow — a name alone is not a violation:
`design/slop_test.py`, `test_first_generator.py`, `test_fixer.py`,
`test_validator.py` and `operations/quick_self_test.py` are production modules
whose domain is testing and define no tests; all five are correctly ignored,
verified individually. `test_instructor_tenacity.py` (three module-level test
functions that make **live API calls**) is baselined with its reason.

**Scope limit (see `t21-ide-backend/coverage.md`):** this wave cleared the
blocker and swept the test surface. Roughly **4,000 of the 5,152 lines remain
unread**, including ~2,700 of `ide_orchestrator_server.py` and all ~1,300 lines
of Subsystem A's routes/handlers/session manager. "T21 complete" means
importable with real tests, **not** audited. No server was started and no port
was bound.

---

## T22 — Reasoning & generation depth (closed)

Sixth wave of the depth pass. Scope: `reasoning/` (5,450 lines, `ara_pipelines.py`
4,288) and `generators/wf100/` (5,121 lines, `checks.py` 2,251) — the largest
line-count debt in the programme, both on money paths.

**The probe that mattered, and the hypothesis it killed.** `wf100` gates every
check on declared evidence (`auditor.py`: `if check.requires - evidence.available
-> OUTSTANDING`). Running all 82 check implementations directly against
`SiteEvidence()` produced **24 PASSes** with self-incriminating details —
*"exactly one `<h1>` on each of 0 pages"*, *"no WCAG violations across 0 pages"*.
That looked like 24 defects; it was not. The probe called implementations
directly and bypassed the gate. But it sharpened into the real rule: **the gate
is only as good as each check's declaration**, so comparing declared `requires`
against the `ev.*` attributes each body reads gives 4 candidates — and reading
them settled 3.

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | G7 (form abuse protection) computed `ev.markup + ev.scripts` while declaring `requires = {MARKUP}`. Captcha, honeypot and rate limiting are normally wired in JavaScript, so with the scripts uncollected G7 searched an empty string for half its evidence and still returned a definite verdict. Demonstrated on identical markup: PASS *"protection in place: captcha"* with scripts collected, FAIL *"public forms with no captcha, honeypot or rate limiting — they will be found by bots"* without. Now returns OUTSTANDING in exactly that ambiguous case. |
| C2 | **VERIFIED DEFECT — FIXED** | `ara_pipelines.py:1424` fetched `state.metadata.get("meta_evaluation", {})` and **discarded it**, under the comment *"# Weight by meta-evaluation quality"*, inside `_phase_jury_weighted_ranking`. The name appears exactly twice in 4,288 lines: written at 1397, dropped at 1424. The ranking is not weighted by it. |
| — | **Cost note** | That data comes from a real `client.call(model=verifier, max_tokens=1500)` at 1388–1394 in a reachable phase, while its sibling call's output (`verifications`) *is* used at 1431/1464/1493. So the orchestrator makes **a paid LLM call on every run and throws the answer away**. The dead statement and false comment are fixed; removing the call or wiring the weighting is escalated, since no specification of the weighting exists and inventing one would fabricate intent. |
| C3–C5 | FALSE (innocent) | A9 and D9 read `ev.http` under `if ev.http is not None` with file/markup fallbacks; E8 reads `ev.record` under `if record else`, falls back to structured data, and returns OUTSTANDING when it still cannot decide — precisely the pattern C1 lacked. |
| C6 | **FALSE — hypothesis falsified** | All four `/ len(...)` sites in both regions are guarded (`if not verified: return`, `if state.scores:`, `if scores else 5.0`, `if len(words) < 50: continue`). No division-by-zero, no vacuous mean. |
| C7 | Recorded, not elevated | Two fabricated neutral defaults — `5.0` for an empty score list, `0.5` for no claims — the same shape as T6's fabricated neutral score and T18's C6, treated consistently rather than re-litigated. |

**Why the C1 fix is narrow.** Declaring SCRIPTS in `requires` would have been
wrong: a site with no JavaScript has no SCRIPTS evidence, so the auditor would
skip G7 entirely and lose a verdict it makes fine from markup. Five cases
verified — protection visible → PASS; scripts uncollected → OUTSTANDING;
scripts collected but unprotected → FAIL; static site with an HTML honeypot →
PASS; static site unprotected → FAIL. The last two are the reason.

**New gate — `scripts/check_wf100_evidence.py`:** fails any check reading
evidence it does not declare; four guarded reads allowlisted with reasons.
Verified to catch the rule rather than bless the exemption — with G7 removed
from the allowlist *and* its fix stashed, it reports `G7: reads SCRIPTS but
declares MARKUP` and exits 1.

**Scope limit (see `t22-reasoning-generation/coverage.md`):** two shapes were
swept exhaustively across both regions; the regions were not audited. **~10,000
of the 10,571 lines remain unread**, including 78 of the 82 check bodies as
logic (only their evidence declarations were verified) and ~4,100 lines of
`ara_pipelines.py`.

---

## T21-follow-up — the dependency I installed locally and never checked in CI

**Trigger:** the first real CI run after a GitHub Actions outage. Every check
had been completing in 1–3 s with `runner_id: 0`, `runner_name: ""` and 404
logs since `d9af8b4`; at 14:13 UTC runners were assigned again
(`runner_id: 1000012211` / `1000012216`, jobs 90 s and 115 s) and the Test job
failed on both the push and the pull_request run.

| ID | Disposition | Summary |
|---|---|---|
| D1 | **VERIFIED DEFECT — FIXED** | `tests/unit/test_ide_color_regex.py`, added in T21, imports `orchestrator.ide_backend.ide_orchestrator_server`. `ide_backend/__init__.py:9` eagerly imports `.server`, which imports `fastapi` at module level. CI installs only `.[dev]`, which has no fastapi, so the import raised at **collection** — and a collection error aborts the run: `3 deselected, 1 error`, with all **2,790** other tests never executed. |
| D2 | Recorded, **not fixed** | `pip install -e ".[dashboard]"` is `ResolutionImpossible`. The extra pins `websockets>=11.0,<13.0` while the core dependency `google-genai` requires `>=13.0`; `httpx` collides too (core `<0.28.0` vs google-genai `>=0.28.1`). So the setup command documented in `CLAUDE.md` — `pip install -e ".[dev,security,tracing,dashboard,docs]"` — **cannot succeed**. Repairing the pins is a dependency decision with real blast radius on the dashboard, so it is escalated rather than taken unilaterally. |

**How D1 was verified, not guessed.** A `sys.meta_path` finder that raises on
`fastapi` reproduced CI's numbers exactly: `2790/2793 tests collected
(3 deselected), 1 error`. Re-run with `--continue-on-collection-errors` to prove
nothing hid behind the abort — still exactly one error. Re-run with fastapi
available and only `uvicorn` blocked, simulating CI after the fix: **0 errors,
2802/2805 collected**. Necessary and sufficient, both directions measured.

**The fix is one line** — the Test job installs `fastapi>=0.100.0,<1.0`
alongside `.[dev]`. Loose rather than through `[dashboard]`, because of D2:
fastapi alone carries none of the conflicting pins, and `.[dev]` plus that pin
resolves cleanly. The reason is stated in `ci.yml` so the next reader does not
"tidy" it into `.[dashboard]` and re-break the run.

**Two alternatives rejected.** `pytest.importorskip("fastapi")` is one line and
would have gone green — by making the test silently not run in CI, which is
verbatim the defect T21 existed to remove ("it lived where pytest never
collected it — nothing had run it"). Trading one never-run test for another is
not a fix. Extracting `replace_accent_color` into a fastapi-free module was the
architecturally tidier option and still fails: `ide_backend/__init__.py` drags
in `server.py` on any submodule import, so it would need a lazy `__getattr__`
package init as well — ~35 lines across four files, changing import semantics
for every existing caller, and leaving the other 16 modules just as unimportable
in CI.

**The process failure worth recording.** T21 was unblocked by running
`pip install fastapi uvicorn[standard] websockets httpx` in the session
container — bare packages, which bypass the project's own pins. That is why it
worked locally and why D2 went unnoticed. The test suite was verified green
against an environment CI does not have and cannot build. No gate is being added
for this: CI's own collection step is the detector, and it caught the defect
within minutes of the runners returning. The gap was that the local environment
had drifted from CI's, and a gate running in the drifted environment would not
have seen it either.

---

## P1 — V4 precision audit, wave 1 (highest-priority region: 8 files, 13,786 LOC)

First wave of `docs/hunts/PRECISION_AUDIT_V4_WAVE_PLAN.md`'s two-speed cut, unit = file
rather than shape (orthogonal to the T0-T22 shape sweeps that preceded it — see that
plan's §1 for why the two methods don't overlap). Full elicitation, innocence, fix, and
coverage detail: `docs/hunts/p1-priority-region/{inventory,coverage}.md`.

**8 VERIFIED/well-mechanized defects fixed, 8 files touched, 11 new regression tests
(`tests/unit/test_hunt_p1_deep.py`), zero regressions** (2790 passed, 24 skipped, 3
deselected, 7 xfailed; the 2 pre-registered `test_openrouter_model_audit.py` failures are
this environment's proxy, confirmed passing on CI). mypy isolated-diff: 3898 -> 3896
errors — a pure reduction, and the two removed errors independently corroborate P1-4 and
P1-5 (mypy's own message on P1-5: `note: Maybe you forgot to use "await"?`).

| ID | Disposition | Summary |
|---|---|---|
| P1-7 | **VERIFIED DEFECT — FIXED** | `api_server.py`'s three `_dispatch_execute_*` handlers reused a constructor-injected long-running `Orchestrator` (the documented mode) and mutated its shared `_run_ctx.budget` with no lock before launching each project as a fire-and-forget background task. `_run_ctx.budget` is read live inside the pipeline (`engine.py:860,897`), so a later concurrent request's overwrite is not inert — it changes what an already-running background project reads on its next check. Fixed by adding an explicit `budget` parameter to `run_project()`/`run_project_with_tasks()` (using `RunContext.reset()`'s existing, previously-unused `budget` hook) and passing it through the call at all three sites instead of writing the shared attribute. |
| P1-3 | **VERIFIED DEFECT — FIXED (partial)** | `ide_orchestrator_server.py::SessionManager.start_server` hardcodes dev-server ports (3000/8000/3000) with no session-scoping; `is_port_available`'s bind-then-close check is TOCTOU-prone against a real uvicorn/npm process's own (slower) bind. Two concurrent same-stack sessions can both pass the check, and the loser's UI shows a false "✓ Dev server running" after its process has already exited. Fixed the false-success half: `SessionManager.is_process_alive()` (new), checked at all 3 call sites before reporting success. The port-collision-in-the-first-place half (real per-session port allocation) is a product decision on port scheme, `[REQUIRES HUMAN REVIEW]`, recorded in coverage.md rather than invented. |
| P1-6 | **VERIFIED DEFECT — FIXED** | `UnifiedClient._clients` was declared `ClassVar[dict[...]]` at class level and never shadowed in `__init__` (unlike the correctly-scoped `_provider_clients`), so every instance in the process shared one XAI-client cache; any instance's `close()` cleared and closed it out from under every other live instance. Confirmed reachable via `WebsiteFactory`'s bounded-concurrency batch mode (N concurrent `Orchestrator`/`UnifiedClient` instances, one `container.py`-constructed each). Fixed: `self._clients = {}` added to `__init__`, `ClassVar` annotation removed. |
| P1-8 | **VERIFIED DEFECT — FIXED** | `first_generator.py::_parse_pytest_output`'s `r"(\d+)\s+failed\s+in"` pattern (matches a 100%-failing run's real summary line) has one capture group — the failed count — but the shared handling code unconditionally assigned `match.group(1)` to `tests_passed`. `"3 failed in 0.52s"` parsed as `(tests_run=3, tests_passed=3)`. The `_run_pytest_locally` call site has its own pre-existing returncode cross-check that neutralizes this; the sandbox call site (`_run_tests_and_collect_results`) does not, and additionally feeds the wrong `tests_passed` into `_calculate_test_quality`, inflating the quality score for a totally-failed run. Fixed at the root (both call sites benefit): failed-only patterns now assign to `tests_failed`, not `tests_passed`. |
| P1-1 | **VERIFIED DEFECT — FIXED** | `website_generator.py::_get_registry`'s `except ImportError` (the fallback that made `component_registry` "dead since it was written" per the earlier `5acde5b` revival commit) had zero log line — the exact reason nobody noticed the registry was dead for its entire lifetime until a separate hunt tier found it by reading source. Fixed with one `logger.warning`, so any future regression of `component_registry`'s imports is visible instead of silently degrading every generated site to 4 uncurated sections. |
| P1-2 | **VERIFIED DEFECT — FIXED (partial)** | `--agent-profile` is a real, documented CLI flag (`cli_dispatch.py:194`, no `choices=`) parsed into a `quality_mode`/`iteration_cap`/`temperature` dict at 4 call sites (`_async_resume`, `_async_file_project`, `_async_new_project`, `_async_visualize` — the last makes no semantic sense as a target, evidence of mechanical copy-paste) and applied at none of them. A second, independently-built classmethod (`AutonomyConfig.from_agent_profile`) exists for exactly this purpose and also has zero callers anywhere in the repo; a third, differently-named profile vocabulary exists in `operations/autonomy.py`. Three competing implementations, none wired, no existing minimal wire-up point on the actual call path (`run_project_streaming` accepts no profile-related parameter) — full wiring is `[REQUIRES HUMAN REVIEW]`. Fixed the visible half: each site now logs that the flag has no effect, rather than silently accepting and discarding it. |
| P1-4 | **VERIFIED DEFECT — FIXED** | `streaming.py::StreamingPipeline._run_pipeline` referenced the bare name `project_description` (not a parameter, not in scope — the value is `context.description`), raising `NameError` on the first statement of every invocation. Caught by the method's own `except Exception`, converted into a generic `ERROR` event, so the entire 3-stage pipeline (Decompose/Execute/Validate) never runs even once. Reachability is DEAD in production: `engine.py`'s real streaming path uses a different class (`ProjectEventBus`) in the same file; zero test files reference `StreamingPipeline`. Fixed with a one-token correction; mypy's own `name-defined` error for this exact line is gone post-fix. |
| P1-5 | **VERIFIED DEFECT — FIXED** | `StreamingPipeline.__init__` calls `get_event_bus()` (genuinely `async def`) synchronously, binding `self.event_bus` to a bare, un-awaited coroutine object — confirmed via Python's own `RuntimeWarning: coroutine 'get_event_bus' was never awaited` firing unprompted. Every `_emit_to_bus()` call then raises `AttributeError: 'coroutine' object has no attribute 'publish'`, silently swallowed by its own `except Exception`. Same DEAD reachability as P1-4. Fixed by deferring resolution to `_emit_to_bus`'s first call (already async) rather than making `__init__` async, which would be a breaking signature change; mypy's own `union-attr` error plus its `Maybe you forgot to use "await"?` note for this exact line are gone post-fix. |

**Cleared (innocent), 6 candidates — see inventory.md for full evidence.** Two are notable
for the discipline they exercised: `ide_orchestrator_server.py:2072-2101`'s
`class Item(BaseModel)` / `@app.get("/items/{{item_id}}")` looked like a live double-brace
routing bug; AST proved it sits inside an f-string inside a code-generation template — the
same grep-vs-AST trap this hunt has hit before. `engine.py::__aexit__`'s unguarded
`await self._c.shutdown()` looked inconsistent with 3 guarded sibling calls; reading
`container.py::shutdown()` in full showed every one of its 7 steps independently
self-guards, so the call cannot propagate.

**Scope limit (see `p1-priority-region/coverage.md`):** two shapes (Concurrency,
Logic/silent-failure) accounted for all 8 findings; Injection, Memory/Resource, and
Edge-case classes were checked with no findings but not exhaustively swept. Roughly 85% of
these 8 files' combined logic remains unread, concentrated in `ide_orchestrator_server.py`'s
multi-thousand-line template-generation bodies and its FastAPI route handlers (this wave
covered only `SessionManager`'s process/port lifecycle). A second, unrelated `RunContext`
class was noticed at `application/unattended_guard.py:36` during Candidate 7's trace and
deliberately not investigated — flagged for a future wave rather than chased as a tangent.

## P2 — V4 precision audit, wave 2 (DEEP tier, priority 8–7, closed)

Full detail: `docs/hunts/p2-deep-region/{inventory,coverage}.md`. Second DEEP-tier wave
(16 files, 14,799 LOC) under `docs/hunts/PRECISION_AUDIT_V4_WAVE_PLAN.md`, same config as
P1 (`APPLY_FIXES=ON`, `TOGGLE_B=ON`, `TOGGLE_C=ON`, `K=8`). Files: `integrations/
slack_integration.py`, `models.py`, `nash/infrastructure_v2.py`, `domain/ports.py`,
`infrastructure/caching.py`, `streaming.py`, `operations/diagnostics.py`, `cli.py`,
`model_selector.py`, `architecture_rules.py`, `unified_events/core.py`,
`events/ab_testing.py`, `engine_core/container.py`, `transfer_learning.py`,
`codebase/context.py`, `state_mgmt/telemetry_store.py`.

**17 VERIFIED/well-mechanized defects fixed** (one, P2-TRANSFER1, partially — visibility
only, full fix escalated), roughly double P1's yield on files that had previously only
been shape-swept (T9-T22), never read in full — evidence for the wave plan's own §1
orthogonality argument, not just its hypothesis.

| ID | Disposition | Summary |
|---|---|---|
| P2-NASH1 | **VERIFIED DEFECT — FIXED (CRITICAL)** | `nash/infrastructure_v2.py::WriteAheadLog.append()`'s two-phase-commit path for files ≥100KB did `async with self._get_io() as io:` — `_get_io()` is an `async def` factory returning an `AsyncIOManager`, which has no `__aenter__`/`__aexit__`. Guaranteed `TypeError` on every large-file WAL write, despite the file's own extensive "TD-001..TD-012 / Round 1-3" commentary claiming this exact path was hardened multiple times. Confirmed live: `nash/__init__.py` wildcard-exports this module and `nash/monitor.py` imports it directly. mypy's isolated diff independently corroborates: `"Coroutine[Any, Any, AsyncIOManager]" has no attribute "__aenter__"/"__aexit__"` — both gone post-fix. Fixed with `io = await self._get_io()` then a plain `await io.write_file(...)`, matching how every other call site in the file uses the same accessor. |
| P2-AB1 | **VERIFIED DEFECT — FIXED (HIGH)** | `events/ab_testing.py::StatisticalAnalyzer._incomplete_beta` was a one-line placeholder (`x**a * (1-x)**b / a`) whose own docstring said "for production use scipy.special.betainc". Measured its real error: `t_cdf(1.0, df=10)` returned `0.019` against a true `0.830` — not merely imprecise, the two-tailed p-value computation (`2*(1-cdf)`) came out as `1.96`, an impossible value for a real probability. Every A/B experiment with Welch-Satterthwaite df ≤ 30 (realistic for smaller `min_samples` or unequal-variance groups; the caller can configure `min_samples` below the default 30) could **never** detect significance, silently defaulting to REJECT regardless of effect size. `scipy` is already a declared project dependency (for `networkx.pagerank`) and installed in this environment — replaced the ~20-line broken approximation with `float(stats.t.cdf(t, df))`. mypy's isolated diff independently confirms: the removed function's "Returning Any from function declared to return float" error is gone. |
| P2-ENGINE1 | **VERIFIED DEFECT — FIXED** | `engine.py::Orchestrator.run_project_streaming()` overwrote `self._event_bus` — the container-wired bus `__init__` sets for the instance's whole lifetime, read by `assert_healthy()` and passed to collaborators — with an unrelated per-call `ProjectEventBus`, then set it to `None` in its `finally` block. Corrupts shared state for any later call on a reused (long-running) Orchestrator instance — confirmed live via `supervisor/service.py` (a persistent REPL) and `cli_dispatch.py`, both of which call this method. Fixed by using a plain local variable (captured by the nested closure) instead of the instance attribute — removes the shared-state hazard entirely rather than relocating it under a new name. |
| P2-UEB1 | **VERIFIED DEFECT — FIXED (well-mechanized)** | `engine_core/container.py::build()` constructs `UnifiedEventBus()` directly (not via the async `get_instance()` singleton) but is itself a **sync** classmethod, so it structurally cannot call the bus's own `async def start()`. Nothing else in the wiring ever calls it either. `UnifiedEventBus.publish()` already warns explicitly when called before `start()`, but the queue is never drained: `_process_loop()` never runs. Confirmed live: `engine_core/sagas.py:624` calls `await self.event_bus.publish(event)` on this exact bus, so every saga-originated event is silently queued forever. Fixed by calling `await self._event_bus.start()` in `engine.py::__aenter__` — an already-existing async lifecycle hook, guarded by `hasattr` for the `NullEventBus` fallback which has no `start()`. |
| P2-TELEMETRY1 | **VERIFIED DEFECT — FIXED** | `state_mgmt/telemetry_store.py::drain_queue()` — whose own docstring says "Called at warm-start time so orphaned writes from a prior crashed session are included" — had zero callers anywhere, and its write-side counterpart `enqueue_snapshot()` also had zero callers (the live write path, `record_snapshots_batch()`, is properly awaited and doesn't appear to hit the failure mode this WAL mechanism was built for). The documented recovery step simply never ran. Fixed by calling `await self._telemetry_store.drain_queue()` in the same `__aenter__` hook as P2-UEB1 — purely additive, idempotent, zero risk to existing behavior. |
| P2-S2-2 / P2-S2-2b | **VERIFIED DEFECT — FIXED (×2, same shape)** | `streaming.py::ProjectEventBus.__init__` called the async `get_event_bus()` synchronously, binding `self._event_bus` to a bare un-awaited coroutine that nothing in the class ever reads — same shape as P1-5, but this time in a **live** class: `engine.py::run_project_streaming()` (core Mediator) and `api_server.py::_stream_via_event_bus()` (SSE endpoint) both construct it. Fixed by deleting the dead assignment (unlike P1-5's `StreamingPipeline`, nothing here reads the attribute, so deletion — not deferred resolution — is the correct minimal fix). While verifying this, found the identical bug in this file's P1-audited twin, `infrastructure/streaming.py::ProjectEventBus` (P1 only fixed `StreamingPipeline` in that file, not this second class) — fixed under this wave since it's the same one-line mechanism, discovered as a direct side effect of an Innocence Check on a P2 file. |
| P2-S2-1 | **VERIFIED DEFECT — FIXED** | `streaming.py::StreamingPipeline._run_pipeline` has the exact NameError shape already fixed in P1-4's `infrastructure/streaming.py` twin (`project_description` referenced out of scope, should be `context.description`) — this is the *other* copy of the same class, in the root file. Reachability: DEAD (no live constructor of `StreamingPipeline` found, matching its twin's fate in P1). mypy's isolated diff independently confirms: `Name "project_description" is not defined` is gone post-fix. |
| P2-TRANSFER1 | **VERIFIED DEFECT — FIXED (partial)** | `transfer_learning.py::TransferLearningEngine.find_transferable_patterns()` computes real cosine-similarity `similar_projects` but uses it only for the empty-result guard — every ACTIVE pattern from every indexed project is returned regardless of similarity, defeating the entire premise of "transfer *learning*". Root cause: `PatternMiner` hardcodes `source_projects=[]` on every mined pattern ("Would populate from archive"), so there is nothing to filter on even if the filter existed. `meta/orchestrator.py::ExecutionRecord` does carry a `project_id` field, so the data exists, but whether it's reliably populated across the whole meta-optimization pipeline was not traced (outside this wave's 16-file scope). Fixed the visible half: corrected the misleading "Collect patterns from similar projects" comment and added a warning log naming the bypass. Full similarity filtering `[REQUIRES HUMAN REVIEW]`. |
| P2-MODELS1 | **VERIFIED DEFECT — FIXED** | `models.py::vs_variant_for()`'s docstring promised three tiers (PREMIUM/STANDARD/BUDGET); the code computed a `_premium` tuple and then never read it — only two effective branches existed. The leading-underscore name meant ruff's unused-variable check (F841) never flagged it, defeating the exact safety net that should have caught this. Confirmed live via `engine_core/stages/generate.py:97`. Fixed by deleting the dead tuple and correcting the docstring to the real 2-tier behavior (zero behavior change); the "should PREMIUM models get materially different treatment" question is undecided and unescalated numerically since no spec states the intended split — noted, not invented. |
| P2-M2-3 | **VERIFIED DEFECT — FIXED (partial)** | `model_selector.py::ModelSelector.decomposition_model(project_description)` never reads its argument; `_COMPLEXITY_KEYWORDS`/`_TECH_STACK_KEYWORDS` (49 keywords) are declared and confirmed dead repo-wide. Real callers (`engine.py:1180`, `application/decomposer.py:706`) pass real, non-empty descriptions expecting some effect — same "wiring gap" shape as P1-2's `--agent-profile`. Fixed the visible half: logs when a non-empty description is silently ignored (empty-string calls, e.g. `fast_decomposition_model()`'s deliberate `""`, stay quiet). Building real complexity-aware routing is `[REQUIRES HUMAN REVIEW]`. |
| P2-ARCH1 | **VERIFIED DEFECT — FIXED** | `architecture_rules.py::ArchitectureRulesEngine._generate_rules_with_llm()`'s "Use first available model" loop did `try: model = m; break; except Exception: continue` over a 5-model priority list — the try body can never raise, so it always selected `architecture_models[0]` and the other 4 declared fallbacks (plus a final `GPT_4O` default) were unreachable, despite an explicit "Check if model is available (basic check)" comment implying otherwise. Fixed via honest simplification (`model = M.CLAUDE_SONNET_5`, kept the alternatives as a comment) — zero behavior change, since that was already the only model ever selected. A real health-aware fallback needs an `api_health` dependency this class doesn't have; not invented. |
| P2-ARCH2 | **VERIFIED DEFECT — FIXED** | The sibling method `_optimize_rules_with_llm()`'s "respond in this JSON format" example, once its f-string braces are rendered, had 3 literal `{` opens vs 4 literal `}` closes — a stray extra closing brace at the very end of instructional text shown to the LLM. Confirmed by capturing the actual rendered prompt in a test and counting braces. Low real-world impact (LLMs are generally robust to a trailing brace in an example), but a genuine, cheap, one-character fix. |
| P2-PORTS1 | **VERIFIED DEFECT — FIXED** | `domain/ports.py::NullTelemetry.record_call()` omitted the `quality_score` parameter that `TelemetryPort`'s own Protocol declares — `@runtime_checkable`'s `isinstance()` only checks method *names* exist, not signatures, so this passed every existing conformance check (confirmed: `tests/test_ports_conformance.py` only does `hasattr`-style checks) while still raising `TypeError` the moment a real caller passes the keyword. No current caller does (`grep` of every real `.record_call(` site), so this is currently dormant, not actively triggered — fixed anyway since it's a one-parameter, zero-risk, exact-Protocol-match correction. |
| P2-SLACK1 | **VERIFIED DEFECT — FIXED** | `integrations/slack_integration.py::RateLimiter.is_allowed()` computed a pruned (expired-entries-removed) timestamp list but discarded it as a local variable — only `record_request()` ever mutated the real stored dict, and it only ever appended. `self._requests[key]` grew without bound for any repeatedly-used key, for the life of the process. Confirmed the whole module (`SlashCommandHandler`/`SlackEndpointHandler`/`SlackIntegrationHooks`) is currently unwired into any live server in this repo — a complete, well-built "bring your own FastAPI app" library nothing here constructs outside its own docstrings. Fixed by writing the pruned list back to storage. |
| P2-SLACK3 | **VERIFIED DEFECT — FIXED** | Same file, `TemplateRegistry.parse_overrides()`: `"1.2.3".replace(".", "").isdigit()` is `True` (all digits/dots survive), so the code unconditionally tried `float("1.2.3")` next — an uncaught `ValueError` outside `_handle_run()`'s own `try/except` (which only wraps `run_template()`, not the earlier `parse_overrides()` call). A single malformed slash-command argument (e.g. `budget=1.2.3`) would crash the whole request handler. Fixed by guarding the conversion and skipping unparseable overrides, matching the function's own existing graceful-skip style for disallowed keys. |
| P2-TRANSFER2 | **VERIFIED DEFECT — FIXED** | `transfer_learning.py` imported `ExecutionArchive`/etc. via `from .meta_orchestrator import (...)` — a re-export shim that fires a real `DeprecationWarning` on every import ("import from orchestrator.meta.orchestrator directly"). Sibling file `events/ab_testing.py` already correctly imports the canonical path. Since T14 C4 established this root `transfer_learning.py` is itself canonical/live (`learning/transfer_learning.py` shims to it), the warning fires on every real import of a live module. One-line import-path fix, zero behavior change (the shim re-exports identically via `import *`). |

**Escalated, not fixed:** `SlashCommandHandler.verify_signature()`'s fail-open-on-missing-config
default (same "should a verification gate fail open" class of decision this hunt consistently
defers); `TransferLearningEngine`'s full similarity-filtering completion (needs an upstream
data-availability fact confirmed outside this wave's scope); `TieredModelRouter.next_tier()`/
`.escalate_tier()` — zero callers on an otherwise-live class, real fix needs invented
per-model tier rankings.

**Cleared (innocent):** the model-tier dict's literal duplicate key (same value, provable
no-op); `DiskCache.get_stats()`'s blocking I/O (initially planned as a fix, reconsidered —
`get_stats()` is deliberately **sync** across the whole `CacheBackend` ABC, so fixing it
means an architecture change, not a minimal patch, for a module with zero external callers);
a `DiskCache` class-name collision between two unrelated caching modules (real, but the
confusable module is fully dead); `AsyncIOManager`'s own deprecated-constructor path being
used internally by two of its sibling classes (real, needs an async-factory refactor);
a second, unrelated `UnifiedEventBus` class in `nash/infrastructure_v2.py` (naming
collision only, no live crash).

**Gate status:** black/ruff/lint-imports (5/5 kept)/root-freeze (256/256)/the 6 T17-T22
gates/test-markers/bandit -lll all PASS. mypy isolated-diff (clean `.mypy_cache`,
`--no-incremental` both sides): 3958 → 3954 errors, pure reduction of exactly 4, zero new —
independently corroborates 3 of the 17 fixes (P2-NASH1's two `__aenter__`/`__aexit__`
errors, P2-AB1's `no-any-return`, P2-S2-1's `name-defined`). New tests 22/22 (RED→GREEN
verified against a real `git stash`; every failure matched its predicted mechanism exactly,
including P2-AB1's raw numbers — `0.0187` vs `0.8296` expected — and P2-ARCH2's brace count
— 3 opens vs 4 closes — both computed independently before the stash-verification run and
confirmed identical). Full suite: 2812 passed (+22 over P1's 2790), 24 skipped, 3 deselected,
7 xfailed — the 2 failures are the same pre-registered environmental ones
(`test_openrouter_model_audit.py`; this sandbox's proxy blocks `openrouter.ai`) — zero
regressions.

**Next:** `docs/hunts/PRECISION_AUDIT_V4_WAVE_PLAN.md` has P3-P11 queued (9 more waves,
~120k more LOC) — not started; per the standing gate, only on explicit user request.
