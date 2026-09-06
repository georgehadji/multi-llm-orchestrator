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
