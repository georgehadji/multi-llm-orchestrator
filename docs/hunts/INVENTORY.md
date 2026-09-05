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
