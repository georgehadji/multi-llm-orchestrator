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
