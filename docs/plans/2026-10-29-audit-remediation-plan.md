# Audit Remediation Plan — the remaining fixes, in order

**Date:** 2026-10-29 · **Base:** `master` @ `ccc06f34` + PR #31 (`fix/score-price-budget`)
**Source:** the 2026-10-29 ponytail audit (20 numbered findings + 14 smaller ones). PR #31
already fixed finding 1 (score parser), finding 2 (billing price table) and the pre-call
check part of finding 3.
**Rules that bind every step:** `CLAUDE.md` rules 1–4 and skill `orchestrator-change-control`.
TDD: write the failing test, watch it fail, fix, commit. One finding per commit. Never weaken
a gate to get past it.

Tags: **VERIFIED** (read in code on 2026-10-29) · **AGENT** (found by an audit sub-agent's
read, not re-read by hand) · **DECISION** (needs the owner before code).

---

## 1. How the order was chosen

1. **Live under default settings first.** A bug that hits every user today beats a bigger
   one behind a flag.
2. **Cheap before expensive.** One-line fixes with a clear test go into Wave A, so the risk
   drops fast.
3. **Money before durability before hardening.** Wrong numbers taint every later
   measurement. Lost state wastes paid work. Most hardening needs a non-default exposure.
4. **Delete last.** Lean work (Wave F) waits until behaviour fixes land, so a deletion never
   races a fix in the same file.
5. **Waves B, C and D can run in parallel**, each on its own branch. §8 lists the files
   that two waves both touch. Merge those in the stated order.

## 2. Patterns, by layer (use what the layer already uses)

The codebase already has its patterns. Each fix uses the one already in its layer. **No new
abstractions.** Every item below fits one of these rows.

| Layer | Pattern already in use | Apply it as |
|---|---|---|
| `models.py`, `domain/` | Pure data: enums, frozen constants | New shared rules become **constants** here (e.g. `RESUMABLE_STATUSES`). No logic, no I/O. |
| `application/` services | Constructor injection of ports and callbacks (`make_state_fn`, `charge_fn`) | New side effects are **injected callbacks**. Never import a concrete adapter (contract 2). |
| `engine_core/stages/` | Pipeline of stages; a stage aborts with a named `abort_reason` | **Guard clause at stage entry** (like PR #31's budget check). Failures become named aborts, never silent passes. |
| `engine_core/container.py` | Composition root | All wiring happens here. Pipeline modules never import infrastructure (contract 4). |
| `infrastructure/` | Adapters behind `domain/ports.py` | Durable writes: **temp file + `os.replace`**. Locks: **lease** (wall clock + refresh). Per-key state: the existing **registry** (`CircuitBreakerRegistry`). |
| Servers (`api_server`, `dashboard_core`, `ide_backend`, `integrations/mcp_server`) | Middleware / handshake checks | **Validate at the trust boundary, once, in one shared helper.** Default to loopback. |
| Budget (`budget.py`, `cost.py`) | `reserve` → `commit_reservation` / `release_reservation` | **Two-phase settle in `try/finally`**: every reservation is settled on success, error and cancel. |

Inheritance, factories and new interfaces are **not** used in any step. No fix needs them.

---

## 3. Wave A — live-by-default one-liners (do first, ~1 day)

Each item is a single commit with one RED test. They touch disjoint files.

| ID | Finding | Change | Test to write first | Class |
|---|---|---|---|---|
| A1 | #5 XSS (VERIFIED) | `dashboard_core/core.py:218`: `html.escape(message)` in `_render_error` | Render `"<img src=x onerror=1>"`; assert no raw `<img` in the output | Security |
| A2 | #4 bind (VERIFIED) | Change `host` defaults from `"0.0.0.0"` to `"127.0.0.1"` in `cli_dashboard.py:18`, `dashboard_core/core.py:347,372`, `integrations/mcp_server.py:75,634`, `command_center_server.py:166`, `crosscutting/config.py:154` | One parametrized test that reads each default with `inspect.signature` or argparse | Security |
| A3 | #6 shared orch closed (VERIFIED) | `api_server.py:780`: `if orch is not self._orchestrator: await _safe_shutdown_async(orch)` | Two projects on one injected orchestrator; assert `close()` is never called | Standard |
| A4 | #9 dead SQLite conn (VERIFIED) | `infrastructure/state.py:573-580`: put `self._conn = None` in a `finally` | `close()` then `save_project()`; assert it reconnects | Standard |
| A5 | #17 FAILED_ name (VERIFIED) | `output_writer.py:283`: write to `out / f"FAILED_{filename}"` | Invalid `.py` output; assert the `FAILED_` file exists and the plain name does not | Standard |
| A6 | small: positional args (VERIFIED) | `engine_core/pipeline_runner.py:91-93,145`: pass `execution_order=` and `results=` by keyword | Spy `make_state_fn`; assert `status` is not a list | Standard |
| A7 | small: xAI provider (AGENT) | `infrastructure/llm_client.py:493`: compare to `"x-ai"`. Check xAI's base URL against xAI's docs before changing it. | `get_provider(GROK_*)` routes to the xAI key | Standard |

**Exit:** dashboard and MCP default to loopback; CLI `--file` writes output again.

---

## 4. Wave B — finish the money path (depends on PR #31)

Order inside the wave matters: B1 makes the budget real, B2–B4 make the ledger honest, B5
closes the parallel gap, B6 removes the dead table.

| ID | Finding | Change | Pattern | Test first |
|---|---|---|---|---|
| B1 | #3 rest: `run_job`/`run_project(budget=)` ignored (VERIFIED, `engine.py:1004,1094`; `run_context.py:120`) | Add `Budget.reset_limits(max_usd, max_time_seconds)` to `budget.py`. It sets the limits and clears spent, reserved and phase fields. `RunContext.reset` calls it on the **shared** container `Budget` instead of rebinding `_run_ctx.budget`. This is safe because `_job_lock` serialises runs (`engine.py:1000`). | Mutate the one shared value object in place under the existing lock. Stages keep their reference. | `Orchestrator(budget=8.0).run_job(JobSpec(budget=0.5))` with a fake client; assert it stops near $0.5 |
| B2 | #14 cumulative charge (AGENT, `project_runner.py:500`) | Save `before = self._budget.spent_usd` before the run. Charge `spent_usd - before`. | Delta settle | Two runs of $1 each; hierarchy shows $2 |
| B3 | #14 error and cancel leak (AGENT, `project_runner.py:489`) | Put the run in `try/finally: charge_job(job_id, team, delta)`. `charge_job` already releases the reservation. `finally` also covers `CancelledError`. | Two-phase settle | Raise mid-run, then cancel mid-run; assert spend is charged and the reservation is released both times |
| B4 | #14 job cap ignores reservations (AGENT, `cost.py:241-243`) | `job_spent = self._job_spent.get(job_id, 0.0) + self._reservations.get(job_id, 0.0)` | Same rule as the org and team checks | Two `can_afford_job("j", "", 0.8)` against a cap of 1.0; the second returns False |
| B5 | #3 rest: uncharged calls (AGENT) | Charge `response.cost_usd` in `engine_core/stages/design_critique.py:63`, `assumption_gate.py:108` and the except branch of `application/decomposer_service.py:104`. `design_critique` needs the budget injected via `build_kwargs`, the way `GenerateStage` gets it. | Charge at the call site | Fake client with cost 0.1; assert the budget moves |
| B6 | #3 rest: parallel overshoot (VERIFIED, `shortcut:` in `generate.py`) | **DECISION:** do it only if parallel overspend is measured to matter. Then: `reserve(models.estimate_cost(...))` before the call, `commit_reservation(est, actual)` after, `release_reservation` on error, all in `try/finally`. Remove the `shortcut:` comment. | Two-phase reserve | 3 parallel tasks against a cap of 1×est; only 1 call runs |
| B7 | #2 rest: dead `costs.json` prices | Remove `get_costs` from `ConfigPort` (`domain/ports.py:117,390`) and `JsonConfigAdapter`. Delete `config/costs.json`, the unreachable `_build_cost_table` and its `_LAZY_TABLES` entry (`models.py:881,924`). Point `tests/unit/test_vfm_routing.py:37,68` at `COST_TABLE`. Run the drift tests (change-control unwritten rule 2). | Delete the second source of truth | `test_vfm_routing` passes against `COST_TABLE` |
| B8 | #20 duplicate price tables | Replace the private tables in `cost_optimization/model_cascading.py:133`, `speculative_gen.py:128` and `streaming_validator.py:147` with `models.estimate_cost`. Delete `cost_optimization/token_budget.py` (1000× wrong and never instantiated; drop its import at `container.py:79`). | One price function | Grep test: no `"input":` price literals outside `models.py` |

**Class:** Standard. Reviews are mandatory (budget math). B1 touches `budget.py` and
`application/run_context.py`, not `engine.py`.

---

## 5. Wave C — durability and resume (parallel with B)

| ID | Finding | Change | Pattern | Test first |
|---|---|---|---|---|
| C1 | #10 lock (VERIFIED, `state.py:589-595`) | Use `time.time()` instead of `time.monotonic()`. Add `refresh_project_lock(project_id)`, called once per level from `pipeline_runner` through an injected callback. Release the lock in a `finally` around the `_job_lock` body in `engine.py` (wiring only, so rule 1 holds). | Lease with heartbeat | (a) a 40 s run keeps its lock; (b) a lock written with a larger monotonic value from "before reboot" expires; (c) re-running right after a CLI run is not refused |
| C2 | #8 no mid-run save (AGENT) | Inject `persist_fn` into `PipelineRunner.execute_all`, the same way `make_state_fn` is passed. Call it under `results_lock` after each `results[tid] = res`. `project_runner` supplies `save_project(project_id, make_state(..., status=PARTIAL_SUCCESS))`. Also save in an `except` around `project_runner.py:396`. | Injected callback, application layer | Kill the run after task 2 of 3 (raise in task 3); the DB row has 2 results |
| C3 | #8 halted runs not resumable (VERIFIED, `project_runner.py:254`) | Add `RESUMABLE_STATUSES = frozenset({PARTIAL_SUCCESS, BUDGET_EXHAUSTED, TIMEOUT})` to `models.py` (pure data). Use it at the gate and in both `find_resumable` queries (`infrastructure/state.py:523,536`). Drop the `'IN_PROGRESS'` literal, which is not a `ProjectStatus`. | Shared constant | A `TIMEOUT` project resumes and keeps its 6 results |
| C4 | resume keeps the stale budget (AGENT, `resumption_service.py:75-81`) | `state = dataclasses.replace(state, budget=self._budget)` before the return | Single owner of `Budget` | Resume spends $1 more; the saved state shows the new total |
| C5 | #18 audit log (AGENT, `infrastructure/audit.py:114`) | Add `async def flush(self)`: `flush_jsonl(path)`, then `self._records.clear()`. The `hasattr` call in `engine.py:757` then works unchanged. | Flush and clear | 3 records, flush twice; the file has 3 lines |
| C6 | small: `cache.db` grows forever (AGENT, `infrastructure/cache.py:68-80,151-157`) | After `CREATE TABLE`: `DELETE FROM cache WHERE created_at < ?` with a TTL. Fix the dead TTL block that sits inside `if self._conn is None`. **DECISION:** TTL value; 30 days proposed. | Prune on open | An old row is gone after reconnect |
| C7 | small: IDE sessions (AGENT, `ide_backend/session_manager.py:216-219,273-276,361-372`) | `get_session` falls back to `load_session`. `_save_session` writes a temp file then `os.replace`. `update_file` returns `False` when `update_in_tree` fails. | Atomic write; honest return | Restart the manager; the session loads. Missing path returns False. |
| C8 | small: `_active_project_state` grows (AGENT, `api_server.py:494,587,685`) | Drop terminal entries older than N minutes when a new project is added. `get_project_status` already falls back to `StateManager`. | Bounded map | 100 finished projects; the map stays ≤ N |

**Class:** Standard. C1 touches `engine.py` (wiring) and C3 touches `models.py` (constant
only), so both are **Architectural** and need human review.

---

## 6. Wave D — security hardening (parallel with B and C; all Security class, human review)

| ID | Finding | Change | Pattern | Test first |
|---|---|---|---|---|
| D1 | #4 Origin check | Add one helper `check_ws_origin(headers, allowed) -> bool` in `orchestrator/security/`. Call it before `accept()` in `dashboard_core/chat_view.py`, `dashboard_core/core.py:337-342` and the IDE handshake. Close with code 1008 on failure. The allowlist defaults to the server's own loopback origin. | One guard at the trust boundary | Handshake with `Origin: https://evil.example` is closed with 1008 |
| D2 | #7 IDE WS crash and session hole (VERIFIED crash) | Make `register_handler(event)` a decorator (`websocket_manager.py:118`). Call `setup_websocket_handlers` once, in `create_app`, not per connection (`server.py:102`). Every handler takes `session_id` from `connection_manager._connection_sessions[websocket]`, never from the payload. | Decorator registry; server-side identity | (a) connecting no longer raises; (b) handlers register once after 3 connections; (c) `file_update` with another `session_id` does not touch that session |
| D3 | small: IDE bind vs auth flag (AGENT, `ide_backend/server.py:158-178`, `auth.py:71`) | `validate_bind_target(host, allow_remote=remote_allowed(), auth_required=auth.auth_required())`. Remove the parameter so there is one source of truth. | Single source of truth | `0.0.0.0` with the env auth off is refused even if the caller passes `auth_required=True` |
| D4 | #11 rate limiter (AGENT, `api_server.py:73-75,285`) | Key on `request.remote` only. Evict buckets idle longer than `window_seconds` on each check. | Bounded map at the edge | Random `X-API-Key` per request is still limited; idle buckets are dropped |
| D5 | #15 `spec_dir` (AGENT, `api_server.py:630-664`) | `Path(spec_dir).resolve()` must be under `ALLOWED_SPEC_ROOT` (reuse `SecurePath`, as in `session_manager`). The 400 message no longer echoes the path. **DECISION:** what is the allowed root? Proposed: the server's working directory. | Containment at the boundary | `spec_dir` outside the root returns 400 without the path |
| D6 | #12 task-id traversal (VERIFIED) | Validate ids where LLM output enters (`application/decomposer.py:545,638`): reject anything that fails `^[A-Za-z0-9_.-]+$`. Also apply `Path(...).name` in `output_writer.py:249` as a second guard. Use the same sanitiser in `checkpoints.py:100,125`. | Validate at the trust boundary + defence in depth | Decomposition with id `"../../x"` is rejected; nothing is written outside `out/` |
| D7 | small: CORS preflight (AGENT, `api_server.py:245-274`) | In the middleware: when `request.method == "OPTIONS"`, return 204 with CORS headers before calling `handler` | Middleware short-circuit | Preflight from an allowed origin gets 204 |
| D8 | small: bad task type is a 500 (AGENT, `api_server.py:562`) | Catch `ValueError` and `AttributeError` around `TaskType(...)` and return 400 | Validate input | Unknown type returns 400 |

Run `bandit -r orchestrator/ --severity-level high -x tests,docs` after each item. No new
`#nosec`.

---

## 7. Wave E — routing and the quality loop (after A7 and B; touches `generate.py`)

| ID | Finding | Change | Pattern | Test first |
|---|---|---|---|---|
| E1 | #13 one global breaker (AGENT, `llm_client.py:217-222,335`) | In `call()`, `async with (await registry.get(model_id)).context():`, using the `CircuitBreakerRegistry` already built in `container.py:957`. Delete the single breaker. Stop counting `CancelledError` (`circuit_breaker.py:214` catches `BaseException`). | Existing registry, per key | 5 failures on model A do not block model B |
| E2 | #13 no fallback on provider error (AGENT, `generate.py:143`) | Catch the provider error in `GenerateStage`, record it through the health tracker (inject `record_failure_fn` via `build_kwargs`), take **one** hop via `FALLBACK_CHAIN.get(model)`, and retry once. A second failure becomes a named abort. | Bounded chain of responsibility (one hop) | Primary raises, fallback succeeds; task COMPLETED, `api_health[primary]` falls |
| E3 | small: `retries=` dropped (AGENT, `llm_client.py:21,328`) | `client.with_options(max_retries=kwargs.pop("retries", _MAX_RETRIES))`. Delete the unused retry stacks `operations/retry_utils.py` and the test-only `run_with_resilience` path only after grep proves zero production callers. | Use the SDK's retry | Fake transport fails twice then succeeds with `retries=2` |
| E4 | #16 retry threshold (AGENT, `self_consistency.py:75`) | Compare against `ctx.task.acceptance_threshold`, not `0.7`. | One threshold | Score 0.8 with threshold 0.85 triggers a retry |
| E5 | #16 keep the best attempt (AGENT, `pipeline_executor.py:111-116`) | Track the best `(score, output, model)` in the executor loop and restore it before `to_task_result`. Fix `pipeline.py:105`, which records score 0.0 for every attempt. | Keep the best seen | Attempt 1 scores 0.8, attempt 2 scores 0.6; the result is 0.8 |
| E6 | small: ARA (AGENT, `reasoning/ara_pipelines.py`, 20 lines listed in the audit; `ara_integration.py:269-307`) | Change `.phase_aware_models` to `..phase_aware_models`. `_fallback_execute` returns score 0.0, not 0.7. | Failure is never a pass | Import each pipeline class; fallback result scores 0.0 |
| E7 | small: `tokens_used` always 0; `candidates` unbound (AGENT, `generate.py:161-167`, `critique.py:87-99`) | Read token counts from the fields `APIResponse` really has. Initialise `candidates = []`. | — | Tokens are non-zero; the flag-on, no-sampler path runs critique |
| E8 | JSON score branch (VERIFIED, `evaluator.py:367`) | **DECISION:** `{"score": 7}` clamps to 1.0 while `"score: 7"` gives 0.7. Existing tests pin the clamp (`tests/test_evaluator_service.py`, `tests/unit/test_evaluator.py:51`). Either rescale JSON like text, or ask the evaluator prompt for 0–1 only and keep the clamp. Do not change it without the owner. | — | — |

---

## 8. Wave F — lean (last)

| ID | Finding | Change | Guard |
|---|---|---|---|
| F1 | #20 dead modules (VERIFIED by grep) | Delete `engine_core/engine_deps.py` (all 67 names are `None`), `engine_core/evaluation.py`, `engine_core/escalation.py`, `costing/tracker.py`, `costing/analytics.py`, `routing/routing.py`, and `operations/retry_utils.py` if E3 did not already. In `container.py:539-550`, replace the `_DepResolver` import dance with `dep_resolver = None` (`ProjectPlanner` already falls back). Update `tests/test_god_file_refactoring.py` (`TestEngineDeps`) and `tests/unit/test_hunt_t12_application_core.py:97`. | Grep the whole tree, including string and dynamic imports, right before each delete |
| F2 | #20 double method (VERIFIED, `engine.py:1267,1355`) | Delete the second `_filter_validators_for_task`. Then decide which implementation is authoritative: `engine_core/validator.py:121` or `quality/validators.py:994`. Make the other one call it. | Run the validator tests before and after |
| F3 | #19 20 duplicated module pairs, ~8,700 lines (VERIFIED) | For each pair: diff it, port any behaviour that exists only in the root copy into the subpackage copy (first: `PRODUCTION_READINESS` from `orchestrator/preflight.py` into `quality/preflight.py`), then turn the root file into a 3-line re-export shim. One PR per ~5 pairs. Start with the two that already split behaviour: `preflight`, `streaming`. | Before the first pair, add a test that fails if a root module with a same-name subpackage twin is longer than 40 lines (ratchet: allowlist the not-yet-collapsed pairs, shrink it each PR). This is a **new gate**, so it needs human approval (change-control §8). |
| F4 | small: `ModelSelector.fallback`, `cheapest_available` (AGENT, `model_selector.py:197,263`) | Delete them. They have no callers and would pick image models. | Grep first |

---

## 9. Conflict map (files touched by more than one wave)

| File | Waves | Merge order |
|---|---|---|
| `api_server.py` | A3, C8, D4, D5, D7, D8 | A → D → C8 |
| `application/project_runner.py` | B2, B3, C2, C3 | B → C |
| `engine_core/pipeline_runner.py` | A6, C1, C2 | A → C1 → C2 |
| `engine_core/stages/generate.py` | B6, E2, E7 | B → E |
| `infrastructure/llm_client.py` | A7, E1, E3 | A → E |
| `infrastructure/state.py` | A4, C1, C3 | A → C |
| `engine_core/container.py` | B5, B8, E2, F1 | B → E → F |

## 10. Owner decisions (answered 2026-10-29)

1. **B6:** do it. Add per-call reservations now and remove the `shortcut:` comment.
2. **C6:** cache TTL is 30 days.
3. **D5:** the owner delegated the choice. The allowed root is the server's working
   directory, resolved once at `APIServer` start (`Path.cwd().resolve()`). Any `spec_dir`
   outside it returns 400.
4. **E8:** rescale JSON scores like text scores (above 10: divide by 100; above 1: divide
   by 10). Update the tests that pin the old clamp: `{"score": 3.5}`, `{"score": 1.5}` and
   `{"score": 99}` in `tests/test_evaluator_service.py` and `tests/unit/test_evaluator.py:51`.
   Values above 100 still clamp to 1.0.
5. **F3:** approved. Add the "no root twin over 40 lines" ratchet gate.

## 11. Done means

- Every item has its RED test committed next to its fix.
- `lint-imports` shows all contracts kept, and `scripts/check_new_root_files.py --baseline origin/master` passes.
- bandit HIGH is clean, and the coverage floor is not lowered.
- CI-scope suite: no failure that is not also failing on `master` (compare as PR #31 did).
- Lean result: about −10,500 lines (Wave F), with no new dependencies.
