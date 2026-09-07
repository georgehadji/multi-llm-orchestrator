# P3 — V4 precision audit, wave 3 (DEEP tier, priority 7)

**Base:** `master` @ `f390414` (post-merge of PR #29).
**Scope:** the 25 files `docs/hunts/v4_waves.tsv` assigns to P3 (14,664 LOC).
**Config:** `APPLY_FIXES=ON`, `TOGGLE_B_INNOCENCE=ON`, `TOGGLE_C_TAIL_SWEEP=ON`, `K=8` —
unchanged from P1/P2.

Epistemic tags per `docs/plans/2026-09-06-outstanding-remediation-plan.md` §0:
**VERIFIED** (read the cited line myself against this base), **INFERENCE**,
**HYPOTHESIS**, **UNKNOWN**.

---

## Findings

### P3-COST3 — reservation leak on every `run_job()` — VERIFIED, HIGH, live money path

**Files:** `orchestrator/policy.py:250-257`, `orchestrator/application/project_runner.py:475-503`,
`orchestrator/cost.py:248-252,363-366,388-392`.

`policy.py::JobSpec` — the dataclass `Orchestrator.run_job()` is annotated to take
(`engine.py:936`) — declares **no `job_id` and no `team` field**. Its full field list is
`project_description, success_criteria, budget, policy_set, quality_targets,
preferred_regions, max_parallel_tasks, quality_mode`. So in `ProjectRunner.run_job()`:

```python
job_id = getattr(spec, "job_id", "") or ""   # always ""
team   = getattr(spec, "team",   "") or ""   # always ""
```

Traced through `BudgetHierarchy`:

1. Pre-flight `can_afford_job("", "", est)` reserves **unconditionally** at `cost.py:248`
   (`self._reserved_usd += estimated_cost`); the two guarded writes that would record
   *who* holds the reservation are both skipped, because `if team:` and `if job_id:`
   (lines 249, 251) are false for `""`.
2. Settlement `charge_job("", "", actual)` releases via
   `reserved = self._reservations.pop(job_id, 0.0)` (`cost.py:363`) → `pop("", 0.0)` →
   **`0.0`**, so line 364's `self._reserved_usd - reserved` is a no-op.
3. The failure path is identical: `release_reservation("", team)`
   (`project_runner.py:492`) pops the same absent key (`cost.py:388`).

**Consequence.** `_reserved_usd` grows by `spec.budget.max_usd` on *every* `run_job()`
call, on both the success and the failure path, and is never released.
`can_afford_job` gates on `committed_org = self._org_spent + self._reserved_usd`
(`cost.py:205`), so once accumulated phantom reservations plus real spend cross
`_org_max`, **every subsequent job is refused** with a `BudgetExceededError` whose
`spent=` field reports `_org_spent` — the real, still-low number — making the refusal
look like a bug in the caller's accounting rather than in the hierarchy's.

With the docstring's own example values (`org_max=100`, `budget.max_usd=8`) the hierarchy
hard-fails on the 13th job regardless of actual spend.

**Blast radius (VERIFIED):** `_reserved_usd` is *not* among the keys `_save_to_db()`
persists (only `org_spent`, `team:*`, `job:*`), so the leak is per-process and a restart
clears it. That bounds it to long-lived processes — which is precisely where `run_job` is
used: `api_server.py` and the supervisor REPL.

This defeats the "Cross-run budget hierarchy with pre-flight checks" capability named in
`CLAUDE.md`'s Core Capabilities.

### P3-COST2 — `reset_spend()` is silently undone by a restart — VERIFIED, MEDIUM, dormant

**File:** `orchestrator/cost.py:330-350` with `:307-328`.

`reset_spend(level, key)` mutates the in-memory dicts and then calls `_save_to_db()`.
`_save_to_db()` builds its row list *from the surviving dict entries* and issues only
`INSERT OR REPLACE` — there is **no `DELETE` statement anywhere in `cost.py`** (verified by
grep). A reset team or job key therefore keeps its old row in the
`budget_hierarchy` table, and `_load_from_db()` restores the pre-reset spend on the next
process start.

`reset_spend("all")` is the same: `org_spent` is reset durably (that row is always
rewritten), but every `team:*` and `job:*` row survives, so "all" resets only the org
level once a restart intervenes.

**Reachability:** `reset_spend` has zero callers repo-wide — real, but dormant. It is a
documented public method ("e.g. at the start of a new billing period"), so the first
caller inherits the bug.

### P3-COST1 — `remaining("job")` ignores reservations — VERIFIED, LOW, dormant

**File:** `orchestrator/cost.py:414-417`.

The org branch deducts `self._reserved_usd` (line 406) and the team branch deducts
`self._team_reserved` under an explicit `# BUG-004 FIX` comment (lines 410-413). The job
branch deducts only `_job_spent`, though `self._reservations[job_id]` holds exactly the
per-job reservation it would need (`cost.py:252`). Same bug BUG-004 fixed one level up,
left unfixed one level down.

**Reachability:** `.remaining(` has zero callers repo-wide — dormant.

### P3-TE1 — TDD-first generation reports fabricated zero cost — VERIFIED, MEDIUM, money path

**File:** `orchestrator/application/task_executor.py:297-319`.

`_try_tdd_generation()` runs `TestFirstGenerator.generate_with_tests(...)`, which is
constructed with `client=self.client` and makes real LLM calls, then returns a
`TaskResult` with `cost_usd=0.0` (its own comment concedes: `# Would need to track from
TDD`) and `tokens_used={"input": 0, "output": len(code.split())}` — the input-token count
is hardcoded to zero and the output count is a whitespace word count, not a token count.

Because `TaskResult.cost_usd` is what the per-run `Budget` and the cross-run hierarchy
both settle against (`project_runner.py:500` reads `self._budget.spent_usd`), every task
that takes the TDD-first path spends real money invisibly. Gated behind
`optim_config.enable_tdd_first`, so it is off by default — same class as the plan's B7
(`ara_pipelines.py` unweighted spend).

### P3-TE2 — semantic-cache hit fabricates a passing quality score — VERIFIED, MEDIUM

**File:** `orchestrator/application/task_executor.py:326-354` with `:213-223`.

`_build_cached_result()` hardcodes `score=0.85` ("Cached patterns meet quality threshold")
and `deterministic_check_passed=True` without running any validator. The value can come
from `semantic_cache.get_cached_pattern(task)` — a *similarity* lookup, not an exact-key
lookup — so a merely-similar prior output is admitted as if it had passed this task's
deterministic checks. `0.85` also clears the `>= 0.7` DEGRADED bar and sits just under the
`>= 0.9` COMPLETED bar in `_build_result_from_cycle`, so it is reported as COMPLETED by
`_build_cached_result`'s own hardcoded `TaskStatus.COMPLETED` regardless.

### P3-PROJ1 — `get_event_bus()` bound un-awaited, in both twins — VERIFIED, LOW, dead

**Files:** `orchestrator/projections.py:610,625` and
`orchestrator/analysis/projections.py:612,628`.

`bus = event_bus or get_event_bus()` where `unified_events/core.py:1058` declares
`async def get_event_bus()`. When the optional `event_bus` argument is omitted the name
`bus` is bound to a bare coroutine object and handed to `ModelPerformanceProjection(bus)` /
`BudgetProjection(bus)`. Third and fourth instance of the shape fixed as P1-5 and
P2-S2-2/2b.

**Reachability:** `get_model_performance_projection` and `get_budget_projection` have zero
callers repo-wide — dead, exactly as P1-4/P1-5 were. Fix is still correct and one token.

### P3-GUARD0 — the entire production-safety module is dead — VERIFIED, HIGH (wiring)

**File:** `orchestrator/safety/guardrails.py` (590 LOC).

`grep` for `get_guardrails|ProductionGuardrails|all_checks_pass|KillSwitch` across
`orchestrator/` returns **zero hits outside the file itself**. A module whose header reads
"CRITICAL: Production safety mechanisms that prevent catastrophic failures" and whose class
docstring lists five "GUARANTEES" (budget never exceeded, kill switch respected, error rate
monitored, memory monitored, drift detected) is imported by nothing. Every guarantee it
advertises is currently vacuous.

The defects below are therefore all **dormant**; they matter because they are what a caller
would inherit the moment this module is wired, and wiring it is the obvious remediation.

### P3-GUARD1 — memory guardrail fails open when `psutil` is absent — VERIFIED, MEDIUM, dormant

**File:** `orchestrator/safety/guardrails.py:364-371`, with `:335-339`.

`check_memory()` wraps its whole body in `try: import psutil` and, on `ImportError`,
returns `GuardrailStatus(passed=True, ..., message="psutil not available, memory check
skipped")`. `psutil` appears **nowhere in `pyproject.toml`** (verified by grep) — it is not
a declared dependency of any extra. So on a default install the documented
"4. Memory monitored (prevent OOM)" guarantee silently degrades to an unconditional pass,
and `all_checks_pass()` counts that unperformed check as a passing one.

The same fail-open shape appears in the throttle at `:323-333`: when called within 10s of
the last check the method returns `passed=True` with "Memory check skipped (recently
checked)". Bounded-latency throttling is defensible; returning the *permissive* value for a
check that did not run is the part worth deciding deliberately.

### P3-GUARD2 — kill switch is throttled into returning "not activated" — VERIFIED, MEDIUM, dormant

**File:** `orchestrator/safety/guardrails.py:210-214`.

`check_kill_switch()` returns `False` — the "keep running" answer — for any call within 5
seconds of the previous check, before looking at the filesystem at all. It also does not
latch: once a call returns `True`, the next call inside the throttle window returns `False`
again, so a caller that re-checks during shutdown sees the switch flip back off. The
sibling `KillSwitch` class does latch (`self._activated`, `:531-534`), so the two kill-switch
implementations in this one file disagree about the semantics.

### P3-GUARD3 — kill-switch paths are world-writable and predictable — VERIFIED, MEDIUM (security), dormant

**File:** `orchestrator/safety/guardrails.py:72-73, 519-520`.

`kill_switch_file = "/tmp/orchestrator_kill"` and `force_kill_file =
"/tmp/orchestrator_force_kill"` are the defaults for both `GuardrailConfig` and
`KillSwitch`. On a shared host any local user can create those paths, and
`KillSwitch.check_and_exit()` responds to the force file with `os._exit(1)` (`:567`) — an
immediate, no-cleanup termination. That is an unauthenticated local denial-of-service
against the orchestrator, plus the usual `/tmp` pre-created-symlink hazard on
`target.touch()`. Same class of decision as the remediation plan's A2 (`0.0.0.0` binds):
the fix is a decision about default location (e.g. under the existing
`~/.orchestrator_cache/`), not a mechanical patch.

### P3-GUARD4 — "budget never exceeded" is detection, not prevention — VERIFIED, LOW, dormant

**File:** `orchestrator/safety/guardrails.py:163-171` vs the class docstring at `:95-97`.

`check_budget()` reports `passed=False` only once `spent > max_budget` — i.e. after the
overspend has already occurred. There is no "would this next call exceed" pre-flight
anywhere in the class. The stated guarantee "1. Budget never exceeded (hard limit)" is a
documentation-vs-reality gap; the real hard limit lives in `BudgetHierarchy.can_afford_job`.

### P3-BATCH0 — the batch-API cost-optimization subsystem is dead — VERIFIED, HIGH (wiring)

**Files:** `orchestrator/cost_optimization/batch_client.py`,
`orchestrator/cost_optimization/cost_optimization_integration.py`.

`BatchClient` is constructed in exactly one place outside its own docstrings:
`cost_optimization_integration.py:68`. That module has **zero external references**
(verified by grep) — nothing imports it. `engine_core/container.py` imports the `BatchClient`
name (`:75, :87`) purely to have something to set to `None` in its `except ImportError`
fallback (`:96`) and **never constructs it** (`grep -c 'BatchClient(' container.py` → 0).
So the advertised "50% cost reduction on non-critical phases" is not reachable from any
live entry point.

### P3-BATCH1 — batch requests hang 300s unless a 10th request arrives — VERIFIED, HIGH, dormant

**File:** `orchestrator/cost_optimization/batch_client.py:214-240`.

`_batch_call()` queues a request and flushes the queue only when
`len(self._request_queue) >= 10` (`:215`). Nothing else ever flushes it: `self._batch_task`
is declared at `:142` and **never assigned**, and `BATCH_WINDOW_SECONDS = 60` (`:124`) and
`MAX_BATCH_SIZE = 1000` (`:126`) are declared and **never read** — the time-window
aggregation the module's docstring describes was never implemented. A request that is not
the 10th therefore falls into the `while time.time() - start_time < timeout` loop at `:222`
and blocks for the full **300 seconds** before raising `TimeoutError` (`:240`).
`shutdown()` (`:453`) does flush, but only if something calls it, and by then the waiters
have already timed out.

### P3-BATCH2 — real batch polling retrieves the wrong id, then swallows the error — VERIFIED, HIGH, dormant

**File:** `orchestrator/cost_optimization/batch_client.py:328-334` and `:391-398`.

`_submit_batch_job()` receives the provider's batch handle as `batch = await
self.client.batches.create(...)` and uses `batch.id` only inside a log line (`:334`) — it is
never stored on the `BatchJob`. `_poll_batch_results()` then polls
`self.client.batches.retrieve(job.id)` (`:393`) using the *locally* generated
`f"batch_{int(time.time())}"` id from `:276`. The provider cannot know that id, so every
retrieve raises; the `except Exception: pass  # Continue polling` at `:397-398` hides it,
and the loop spins until the 600s `max_wait` raises `TimeoutError`. The real (non-simulated)
batch path can therefore never complete successfully.

### P3-BATCH3 — savings metric inflated ~1000x — VERIFIED, MEDIUM, dormant

**File:** `orchestrator/cost_optimization/batch_client.py:413-433`.

`COST_PER_1K` is documented as "per 1K tokens" and applied as `(tokens / 1000) *
cost_per_1k`, but its values are the published **per-million** prices (`claude-opus: 15.0`,
`gpt-4: 30.0`, `claude-sonnet: 3.0`). Cost is therefore overstated by a factor of ~1000, and
`metrics.total_savings` (`:231`) reports that inflated figure. It feeds reporting only, not
a charge, so no money moves — but the "50% cost reduction" claim is measured with a broken
ruler.

### P3-BATCH4 — `batch_call()` silently discards four of its parameters — VERIFIED, MEDIUM, dormant

**File:** `orchestrator/cost_optimization/batch_client.py:471-496`.

The convenience wrapper accepts `system`, `max_tokens`, `temperature`, `timeout` and
`**kwargs`, then calls `batch.call(model, prompt, phase_enum)` — passing **none** of them.
A caller setting `max_tokens=8192` or a system prompt gets neither, with no warning. Same
wiring-gap shape as P1-2 (`--agent-profile`) and P2-M2-3 (`decomposition_model`).

Also `_submit_batch_job()` writes `/tmp/batch_{job.id}.jsonl` (`:322`) — predictable
world-writable path, never deleted.

### P3-SHADOW1 — seven modules are unimportable by construction — VERIFIED, HIGH (structural)

**Files:** `orchestrator/{skills,verification,gateway,agents,workspace,connectors,plugins}.py`,
each colliding with a same-named package directory.

Where `orchestrator/X.py` and `orchestrator/X/__init__.py` both exist, Python's import
system resolves `orchestrator.X` to the **package**; the sibling module is shadowed and can
never be imported by that name. Verified empirically for all seven:

| shadowed module | LOC | `orchestrator.X` actually resolves to |
|---|---|---|
| `orchestrator/gateway.py` | 477 | `orchestrator/gateway/__init__.py` |
| `orchestrator/agents.py` | 277 | `orchestrator/agents/__init__.py` |
| `orchestrator/verification.py` | 11 | `orchestrator/verification/__init__.py` |
| `orchestrator/skills.py` | 11 | `orchestrator/skills/__init__.py` |
| `orchestrator/workspace.py` | 11 | `orchestrator/workspace/__init__.py` |
| `orchestrator/plugins.py` | 11 | `orchestrator/plugins/__init__.py` |
| `orchestrator/connectors.py` | 7 | `orchestrator/connectors/__init__.py` |

Two carry real implementations (`gateway.py` 477 LOC, `agents.py` 277 LOC) that are dead by
construction rather than by neglect — no call site could reach them even if one wanted to.
The other five are deprecation re-export shims of the kind T17 introduced; each is a shim
whose deprecation warning can never fire, so any caller still using the old path gets the
package silently instead of the warning the shim was written to emit.

`orchestrator/verification.py` is one of the two files `CLAUDE.md`'s pattern table names for
the Decorator/"Optional Features" row — so the table cites a path that does not resolve.

### P3-ORPHAN0 — dead-module census — VERIFIED (measurement)

Method: AST import graph over `orchestrator/` + `scripts/` + repo-root launchers, with
`tests/` counted separately; `pyproject` `[project.scripts]`, the
`orchestrator.pipeline.stages` entry-point group, and `container.py`'s
`_FALLBACK_ENTRY_POINTS` string list all counted as real references (the last two are
dynamic and invisible to an import scan). Operational non-Python references (CI yaml,
`.bat`, Dockerfile, packaging) count as entry-point evidence; `docs/` is excluded, because
the hunt documentation names nearly every module in the repo and would otherwise mark the
whole codebase reachable.

| bucket | files | LOC | meaning |
|---|---|---|---|
| **ORPHAN** | 102 | 29,909 | no product import, no test import, no `__main__`, no operational reference |
| **TEST-ONLY** | 64 | 10,069 | imported by tests only — exercised, never wired |
| **MAIN-ONLY** | 13 | 4,471 | has `if __name__ == "__main__"`, nothing imports it |
| ENTRYPOINT | 6 | 4,935 | correctly unimported (declared entry points) |
| SHIM | 94 | 918 | re-export shims (expected, from T17) |

**179 files / 44,449 LOC of product code is unreachable from any live entry point** —
18% of the 242,921 backend LOC measured by the V4 wave plan.

Two named in `CLAUDE.md`'s own architecture table are ORPHANs:
`prompt_enhancer.py` (Decorator row — zero importers repo-wide) and `gateway.py`
(Facade/"HTTP API Gateway" row — and additionally shadowed, see P3-SHADOW1).

**Innocence check performed.** `orchestrator/preflight.py` (root) *is* referenced and is
what `CLAUDE.md`'s Validation row means; only its `quality/preflight.py` twin is orphaned —
so the table is right there. `project_mgmt/assembler.py:664`'s
`importlib.import_module(module_path)` looked like a third dynamic-import source but sits
inside an f-string code template (doubled `{{cid}}` braces) — it is generated output, not a
live import, and was excluded.

### P3-TE3 — missing `await` disables both caches and injects a coroutine repr into the prompt — VERIFIED, HIGH (shape), dormant

**File:** `orchestrator/application/task_executor.py:199-201`, consequences at
`:205, :214, :138, :229, :241-253, :284`.

```python
async def _build_execution_context(self, task, all_tasks, results) -> ExecutionContext:
    dependency_context = self.dependency_resolver.get_dependency_context(   # no await
        task, results, all_tasks
    )
```

`get_dependency_context` is `async def` (`application/dependency_resolver.py:169`) and is
the **only** definition of that name in the repo, so there is no sync overload this could be
resolving to. The enclosing method is itself `async def`, so `await` was available and
simply omitted. `dependency_context` is therefore a coroutine object — and a coroutine object
is always truthy. Five separate behaviours downstream invert:

1. `:205` `if not dependency_context and self.cache_optimizer:` → false → **the
   cache-optimizer lookup never runs**.
2. `:214` `if cached_result is None and not dependency_context and ...` → false → **the
   semantic cache never runs**. Between them, the entire caching layer of task execution is
   unreachable.
3. `:229` the coroutine is stored on `ExecutionContext.dependency_context`.
4. `:241` `if context.dependency_context:` → **always true** → `_build_full_prompt` always
   appends `"--- CONTEXT FROM PRIOR TASKS ---\n<coroutine object
   DependencyResolver.get_dependency_context at 0x...>"` to the prompt sent to the model.
   For `TaskType.CODE_REVIEW` (`:243-249`) the injected text is worse, because the template
   asserts *"The following is the actual generated source code you must review. Do NOT claim
   the code was not provided."* immediately before the coroutine repr.
5. The real dependency context — the whole point of the call — is discarded, and Python
   emits `RuntimeWarning: coroutine ... was never awaited`.

**Reachability: dormant.** `TaskExecutor` is constructed in exactly one place repo-wide —
`tests/unit/test_task_executor.py:89` — and never on a product path; `engine_core/__init__.py`
and `application/__init__.py` only re-export the name. The live pipeline uses
`engine_core/stages/*` instead.

**Why the existing test does not catch it (VERIFIED):** the fixture passes `MagicMock()`
collaborators (`tests/unit/test_task_executor.py:30-64`). A `MagicMock`'s
`get_dependency_context(...)` returns another truthy `MagicMock`, which behaves
indistinguishably from the truthy coroutine at every branch above. The test cannot fail on
this bug by construction — an argument for asserting on `AsyncMock` for async collaborators,
which the same fixture already does correctly for `cache_optimizer.get` (`:48`).

**Cleared (innocent) — `unified_events/core.py:671`.** `Projection.apply()`'s
`handler(event)` looked like the same shape, but every `on_*` handler on the projections
defined in that file (`ProjectStateProjection` et al.) is a plain `def`. The `async def`
`on_*` handlers live on the *other* `Projection` hierarchy in `projections.py`, which this
`apply()` never dispatches to. No defect.
