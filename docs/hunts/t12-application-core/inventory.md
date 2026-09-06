# T12 — application/ orchestration core, agents/, supervisor/, nash/, meta/ — Inventory

Fourth of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`, continuing the
AUTONOMOUS DEFECT-HUNT PROTOCOL V7 across backend files with no individual disposition
recorded in earlier tiers.

## Phase 0 — Scope

93 files: `application/` (42, minus `evaluator.py`/`budget_enforcer.py` already fixed/verified
in T1), `planning/` (3), `routing/` (2), `reasoning/` (6), `agents/`+`agents.py` (16),
`supervisor/` (7), `delegation/` (3), `meta/` (7), `nash/` (6).

## Phase 1-3 — Survey, candidates, trigger/innocence

A background agent (general-purpose) surveyed all 93 files plus four pre-identified leads
(the `executor.py`/`task_executor.py` cluster, a third `dependency_resolver.py` copy not
compared in T11, a cross-tier `budget_enforcer.py` comparison, and T10's `cache_warmup.py`
handoff). It ran an automated `importlib.import_module` sweep across all 93 files — the
mechanism that surfaced the ARA-pipeline and `routing/` findings below — plus targeted
reachability greps for `nash/` and `meta/`.

**Every finding below was independently re-verified from source before any fix was applied**
(reading the actual files, re-deriving diffs, and empirically re-running the exact broken
import statements), per this hunt's standing "trust but verify" discipline. One of the
agent's own claims was corrected during verification: `application/task_executor.py`'s
missing-`await` bug (see Cleared/residual below) is real but was mis-classified by the agent
as living in a class with unclear reachability — independently confirmed dead (T1 already
established `TaskExecutor` has zero live callers), so it stays residual, not a fix.

## Phase 4 — Fixes (VERIFIED DEFECT)

### C1 — Instructor fast-path decomposition bypassed the run budget entirely

**File:** `orchestrator/application/decomposer_service.py:69-84` (fix at same location).

`decompose_project()` tries a fast path first: Instructor-based structured decomposition via
`TaskDecomposer` (`orchestrator/structured_outputs.py`), falling back to the DAG-based
`Decomposer` only on `ImportError`/exception. The fallback path correctly threads
`charge_fn=lambda amount: self.budget.charge(amount, "decomposition")` (wired at
`engine.py:1011`) through to `application/decomposer.py:421-425`, which awaits it. The fast
path's success branch (`decomposer_service.py:84`, pre-fix) returned tasks directly with
**zero reference to `charge_fn`**.

Verified this is a real, billable, live gap, not theoretical:
- `structured_outputs.py::TaskDecomposer.get_client()` (lines 201-212) builds a raw
  `openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=self._get_api_key())`
  when an `api_client` was passed in — bypassing `UnifiedClient`'s cost tracking entirely, by
  the code's own comment ("UnifiedClient wraps OpenRouter internally; build a direct
  AsyncOpenAI for instructor since UnifiedClient doesn't expose a .client attribute").
- `_get_api_key()` reads `OPENROUTER_API_KEY` — confirmed via grep this is the same,
  standard, `.env.example`-documented env var the main hot path
  (`infrastructure/llm_client.py`) uses, not a differently-named or unused variable. Any
  normally-configured deployment has this set.
- Grepped the whole of `structured_outputs.py` for `cost`/`charge`/`budget`: the only match
  is a log message ("Decomposition token budget: …") describing a *token* estimate for
  `max_tokens` sizing, not a cost/charge call. Zero cost accounting exists on this path.
  independent of `decomposer_service.py`.
- The fast path fires whenever `len(enhanced_project) <= _INSTRUCTOR_MAX_CHARS` (8,000
  chars) — well under `application/validators.py`'s own `MAX_PROJECT_DESCRIPTION_LEN`
  (10,000), i.e. it is the common case for real project descriptions, not an edge case.

**Fix:** after a successful fast-path decomposition, charge an estimated cost via the
existing, already-established `models.py::estimate_cost(model, input_tokens, output_tokens)`
helper (the same COST_TABLE-based estimation convention already used elsewhere for
pre-flight sizing), wrapped in the same defensive `try/except: logger.debug(...)` style
`application/decomposer.py:421-425` already uses around its own `charge_fn` call — charging
must never crash a successful decomposition. Output tokens reuse
`TaskDecomposer._calculate_decomposition_tokens()` (the same value already sent as the real
request's `max_tokens`, for internal consistency); input tokens use the standard ~4-chars-
per-token heuristic against `enhanced_project + criteria` (a conservative undercount, since
the real prompt also includes a system message and template text not visible from
`decomposer_service.py`).

**What this fix does NOT claim:** the charged amount is an *estimate*, not the real
Instructor/OpenRouter-reported cost — computing the latter would require restructuring
`structured_outputs.py` (out of any tier's scope) to retrieve real token usage (e.g. via
`instructor`'s `create_with_completion`) and was judged too large a change, on a file outside
this tier's declared scope, to make blind in an autonomous, unreviewed pass. The estimate
converts "$0.00 tracked, real cost > $0" into "an approximation tracked" — a materially
smaller error, and consistent with how the rest of this codebase's pre-flight cost tracking
is already approximate (static per-model rates × token counts, not live provider billing).
A follow-up that threads real usage through `structured_outputs.py` remains open.

### C2 — ARA reasoning-pipeline subsystem (~5,000 lines) unreachable via 3+1 wrong-depth imports

**Files:** `orchestrator/engine_core/method_selector.py` (4 broken imports: lines 17, 18, 21,
593), `orchestrator/engine_core/container.py` (2 broken imports: lines 726, 739),
`orchestrator/engine_core/engine_deps.py` (1 broken import: line 279).

`reasoning/ara_pipelines.py` (4,288 lines implementing 22 reasoning strategies — Debate,
Jury, Socratic, Bayesian, Dialectical, Delphi, ToT, PoT, SelfDiscover, etc. — plus a
`PipelineFactory`) and its supporting `ara_integration.py`/`ara_execution_strategy.py` are
fully built, enabled-by-default (`ARAExecutionStrategy._config.enabled: bool = True`), and
completely unreachable:

- `engine_core/method_selector.py:17`: `from .ara_pipelines import ReasoningMethod` — assumes
  a sibling `engine_core/ara_pipelines.py`, which does not exist (the real one is
  `orchestrator/reasoning/ara_pipelines.py`, with a root re-export shim at
  `orchestrator/ara_pipelines.py`). This alone makes `method_selector.py` itself
  unimportable — verified via `python3 -c "import orchestrator.engine_core.method_selector"`
  raising `ModuleNotFoundError: No module named 'orchestrator.engine_core.ara_pipelines'`
  pre-fix.
- The same file has three more instances of the identical wrong-dot-count bug, found only
  after fixing line 17 and re-running the import (line 18 `from .models import Task,
  TaskType`; line 21, `TYPE_CHECKING`-only, `from .api_clients import UnifiedClient`; line
  593 `from .models import ROUTING_TABLE, Model`) — all needed a second `.` to reach the
  package root from `engine_core/`.
- `engine_core/container.py:53` already had the *correct* form
  (`from ..ara_execution_strategy import ARAExecutionStrategy`, module-level) — but a second,
  function-local import at line 739 (inside the `ServiceContainer` construction method) had
  the identical class name imported with the wrong dot-count
  (`from .ara_execution_strategy import ARAExecutionStrategy`), a strong internal
  self-contradiction confirming this was a copy/paste or refactor artifact, not a
  fundamentally different code path.
- `container.py:726` and `engine_deps.py:279` both did
  `from .ara_integration import create_ara_integration` — `engine_core/ara_integration.py`
  does not exist; the canonical implementation lives at root
  (`orchestrator/ara_integration.py`, a full independent implementation — *not* a shim,
  unlike its `ara_pipelines`/`ara_execution_strategy` siblings; `reasoning/ara_integration.py`
  is a second, correctly-depth-adjusted, behaviorally-identical copy, diffed line-for-line to
  confirm the only differences are import-path adjustments for its own depth).

All three call sites are wrapped in bare `try: ... except ImportError: X = None`
with **no logging at all** — `engine_deps.py`'s `HAS_ARA` flag is set to `False` and then
never read anywhere in the repository (confirmed via grep), making the failure completely
silent and unobservable. `ARAExecutionStrategy` is designed to fail-open on
`ara_integration=None` (its own docstring: "All decisions fail-open — errors disable ARA for
that task, not crash"), so the defect never crashed anything — it just meant an enabled-by-
default reasoning-enhancement capability has been completely inert.

**Fix:** corrected all 7 relative-import dot-counts to the importing files' actual depth
(`engine_core/` is one level below the package root, needs `..` to reach root- or
`reasoning/`-level modules). `method_selector.py:17` and the `container.py`/`engine_deps.py`
ARA-integration imports now point directly at the already-existing canonical locations
(`..reasoning.ara_pipelines`, `..ara_integration`, `..ara_execution_strategy`) rather than
adding a new indirection layer.

**Verified end-to-end, not just import-success:** with a dummy `OPENROUTER_API_KEY` set,
`create_ara_integration()` now constructs a real `ARAPipelineIntegration`, and
`ARAExecutionStrategy(ara_integration=<that object>)` now reports `enabled=True` — the
subsystem is genuinely constructible, not merely importable. Also re-ran the full pre-
existing `test_container_stage_discovery.py`/`test_container_acr_wiring.py` suites (26
tests) against the fixed `container.py`/`engine_deps.py` — all pass, confirming
`ServiceContainer.build()`'s real, non-mocked construction path is unaffected.

**Cross-tier note:** `engine_core/` was T11's declared directory scope (already closed,
committed, pushed). This defect is only traceable from a T12-scope file
(`reasoning/ara_pipelines.py`) into `engine_core/`'s import statements, and the fix is a
purely mechanical dot-count correction (this hunt's most common, best-established fix
pattern, applied 10+ times across T1-T11) with zero design-decision content — not a case for
manufacturing an artificial hand-off. Documented explicitly here rather than silently
reopening T11's scope.

### C3 — `routing/__init__.py` imports a `selector.py` that never existed

**File:** `orchestrator/routing/__init__.py:3` (removed).

`routing/__init__.py` did `from .selector import *` before `from .routing import *`.
`orchestrator/routing/` contains only `__init__.py` and `routing.py` — no `selector.py`, and
`git log --all --oneline -- orchestrator/routing/selector.py` returns nothing: this file
never existed at any point in this repository's history. Because Python must execute a
package's `__init__.py` before any of its submodules are reachable, this broke not just
`import orchestrator.routing` but, transitively, `orchestrator/model_routing.py` (root, a
re-export shim: `from orchestrator.routing.routing import *`) — the exact file
`CLAUDE.md`'s own architecture table names for the Strategy-pattern "LLM Routing" role
(`| LLM Routing | Strategy | model_routing.py, planner.py |`).

Considered wiring `.selector` to `orchestrator/model_selector.py` (root, has
`ModelSelector`/`TieredModelRouter` classes) since the name is suggestive — rejected: no
evidence connects them beyond name similarity (confirmed via `git log`, no such file ever
existed to have been renamed from), and inventing a connection would be fabricating
functionality rather than fixing a defect. Confirmed zero live callers of anything under
`orchestrator.routing` or `orchestrator.model_routing` (the one non-obvious grep hit,
`design/catalogs/__init__.py:23`'s `from .routing import ...`, resolves to a same-named but
entirely unrelated `design/catalogs/routing.py`, not this package) — this is dead code today,
but CLAUDE.md itself points at it as architecturally significant, so leaving it permanently
broken was not appropriate to defer.

**Fix:** removed the broken `from .selector import *` line. `routing/__init__.py` now only
re-exports from the real, existing `routing.py`.

### C4 — `meta/integration.py` diverged from its already-fixed root canonical copy

**File:** `orchestrator/meta/integration.py` (converted to a shim).

`orchestrator/meta_integration.py` (root) and `orchestrator/meta/integration.py` (subpackage)
are near-byte-identical implementations of `initialize_meta_optimization`/
`MetaOptimizationV2Wrapper`/`get_meta_status`. A full `diff` confirmed the *only* substantive
divergence: `meta/integration.py:216`, `_state_to_trajectory`, read
`state.status.value == "success" if hasattr(state, "status") else True` — raising
`AttributeError` whenever `state.status` is a plain string rather than an enum. The root copy
(`meta_integration.py:216-221`) already carries a fix, landed in commit `66f804c`:
`getattr(state.status, "value", state.status) == "success"`, with a comment explicitly
noting "status may be an enum (use .value) or a plain str — normalize both." Every other
difference between the two files is a relative-import depth adjustment (root uses one `.`,
the subpackage copy correctly uses `..` for the same targets) — expected and correct, not a
defect.

All four real call sites (`application/project_runner.py:409`, `application/cli_helpers.py:131`,
`commands/meta.py:12`, `engine_slimming.py:162`) import the root module exclusively; nothing
reaches the subpackage copy via `meta/__init__.py`'s `from .integration import *`, so the bug
was live-reachable in principle but not exercised by any current caller — confirmed via grep
across the whole `orchestrator/` tree.

Unlike this hunt's other 8+ duplicate-pair findings, the canonical copy here is the **root**
file, not the subpackage — the other five `meta_*.py` root files
(`meta_performance.py`, `meta_config.py`, `meta_monitoring.py`, `meta_orchestrator.py`,
`meta_v2_integration.py`) are each already clean shims pointing *into* `meta/`, the opposite
direction. Confirmed via `create_ara_integration`-style empirical re-import after the fix:
`orchestrator.meta.integration.MetaOptimizationV2Wrapper is orchestrator.meta_integration.MetaOptimizationV2Wrapper`
is `True`, and a full `python3 -c "import orchestrator.meta"` succeeds with no circular-import
error (traced the potential cycle by hand first — `meta_integration.py`'s own imports
(`meta_orchestrator`, `meta_v2_integration`, `transfer_learning`, `engine`, `state`) never
reach back into `orchestrator.meta`/`orchestrator.meta.integration` — then confirmed
empirically rather than trusting the trace alone).

**Fix:** converted `orchestrator/meta/integration.py` into a `from ..meta_integration import *`
shim, matching the plain-relative-import, no-`DeprecationWarning` style this hunt has used for
every other "convert the lesser copy to a shim" fix since T9 (e.g.
`orchestrator/safety/architecture_rules.py`, `orchestrator/engine_core/dependency_resolver.py`)
— not the older `DeprecationWarning`-carrying style the *other* `meta_*.py` root shims use, since
that style was established for a different direction of shim.

### C5 — `application/cache_warmup.py` imports a nonexistent module (T10 handoff, corrected)

**File:** `orchestrator/application/cache_warmup.py:44`.

T10 flagged this as a broken import (`from ..operations.cache_warmup import
warm_prompt_cache` — `orchestrator/operations/cache_warmup.py` does not exist) and
characterized it as "live hot-path but only a missed optimization," deferring the fix to
whichever tier reached `application/`.

**T10's "live" characterization is corrected here: this is fully dead code, not a hot path.**
`warm_cache_for_level()`'s only caller is `engine.py:1047`'s `Orchestrator._warm_cache_for_level`
method — and `grep -rn "_warm_cache_for_level(" orchestrator` (invocation, not definition)
finds **zero** call sites anywhere in the repository. `git log --all -p
-S"_warm_cache_for_level("` confirms this method has never had a caller at any point in this
repository's history, before or after the "engine dissolution" refactor T10 likely inspected.
No run has ever benefited from — or lost — this optimization; it is an unfinished feature, not
a regression.

The real `warm_prompt_cache` implementation lives at
`orchestrator/cost_optimization/prompt_cache.py:373` (`async def warm_prompt_cache(system_prompt:
str, project_context: str, client=None) -> str`) — a signature match for how
`warm_cache_for_level` already calls it (`system_prompt=`, `project_context=`, `client=`).

**Fix:** corrected the import to `from ..cost_optimization.prompt_cache import
warm_prompt_cache`. This does **not** wire `_warm_cache_for_level` into any live execution
path (that would require adding a real call site inside `pipeline_runner.py`'s level-execution
loop — a product decision about whether this optimization is wanted at all, left to a human)
— it only ensures that *if* something calls it, the intended cache-warming call now actually
happens instead of unconditionally hitting the `except Exception` branch and logging a no-op
warning forever.

## Phase 4 — Cleared (innocent) / residual, not fixed

- **`application/executor.py` vs `application/task_executor.py`** — confirmed genuinely
  different responsibilities (a thin timing/metrics wrapper vs. a full generate→dependency→
  cache→critique pipeline), not a diverged duplicate pair. `ExecutorService` (the former) is
  live (`engine_core/container.py:665`, `services/__init__.py:9`); `TaskExecutor` (the
  latter) remains dead, matching T1's C3 finding.
  - **Residual, found inside the dead code, not fixed:** `task_executor.py:199-201` calls
    `self.dependency_resolver.get_dependency_context(...)` — an `async def` method — without
    `await`, inside an `async def _build_execution_context`, with a `# type: ignore[arg-type]`
    at line 229 suppressing exactly the mypy error this produces. Currently dormant
    (`TaskExecutor` has no live callers), so `[REQUIRES HUMAN REVIEW]` rather than fixed here
    — fixing dead code that nothing exercises is lower priority than the five live/reachable
    defects above, and this tier's fix budget is a cap, not a floor.
- **`application/dependency_resolver.py`** — confirmed a genuinely different class from both
  `engine_core/dep_resolver.py` (a Python-import/pip-dependency scanner, unrelated, live via
  `appbuilder/builder.py:70`) and `engine_core/dependency_resolver.py` (a dead shim
  re-exporting *this* file, confirmed via `from ..application.dependency_resolver import *`).
  Not a third duplicate — `application/dependency_resolver.py` is the actual canonical
  implementation, just unreachable today (only consumer is the dead `TaskExecutor`).
- **`engine_core/budget_enforcer.py` vs `application/budget_enforcer.py`** — confirmed the
  former is a plain 7-line `from ..application.budget_enforcer import *` shim, unchanged
  since an unrelated commit (`66f804c`). It was never an independent implementation, so T1's
  C2 fix (making `record_cost()` async, delegating to `await self.budget.charge(...)`) is
  automatically inherited — zero divergence risk, nothing to fix.
- **`HumanInTheLoop` gate never reaches `ProjectRunner`, and `UnattendedGuard`'s checkpoint
  detection is permanently blind — `[REQUIRES HUMAN REVIEW]`:**
  `engine_core/container.py` constructs a real, fail-closed `HumanInTheLoop()` and stores it
  as `ServiceContainer.hitl` (line 918/1008). `application/project_runner.py:136-164`'s
  `ProjectRunner.__init__` has no `hitl` parameter at all; `engine.py`'s single construction
  call site (`engine.py:563-577`) never passes `self._c.hitl` through. Consequently,
  `run_project()`'s `_hitl = getattr(self, "_hitl", None)` (line 232) always evaluates `None`,
  so `_has_checkpoint` (line 233) is always `False` — `UnattendedGuard.validate()`
  (`application/unattended_guard.py:106-114`) always reports "no checkpoint reachable" for
  every unattended run, regardless of whether a real decision channel is actually configured.
  A 3-line fix (thread `hitl` through `ProjectRunner.__init__` and the `engine.py` call site)
  would very likely only ever make the gate *more* accurate (permissive only for deployments
  that have configured a real channel; unchanged for everyone else) — but this touches a
  fail-closed safety gate's actual behavior, and `HumanInTheLoop.has_real_channel()`'s own
  correctness was not independently verified in this tier. Matches this hunt's standing
  precedent (T9's plugin isolation, T11's `UnifiedEventBus.start()`): wiring a dormant
  safety-relevant subsystem into a live path is a human decision, not fixed here.
- **`orchestrator/agents.py` permanently shadowed by the `orchestrator/agents/` package —
  `[REQUIRES HUMAN REVIEW]`:** both coexist; Python's import system always resolves
  `from .agents import X` (or `orchestrator.agents`) to the *package*, never the flat module
  (empirically confirmed: `ImportError: cannot import name 'TaskChannel' from
  'orchestrator.agents'`). `engine.py:158` and `engine_core/engine_deps.py:82` both try to
  import `TaskChannel` from it, both silently caught (`except (ImportError, TimeoutError):
  TaskChannel = None`), and `TaskChannel` is never referenced again in either file afterward
  (confirmed via grep) — currently inert, not a live crash. No amount of dot-count adjustment
  fixes a same-level name collision; the real fix is a rename/restructuring decision (move
  `TaskChannel`/`AgentPool` into the package, or rename the flat module), left to a human.
  `agents.py`'s own docstring describes an `orch.get_channel(name)` feature that does not
  exist anywhere on the live `Orchestrator` class.
- **`delegation/batch_runner.py`** — `BatchRunner` is constructed for real
  (`engine_core/service_collection.py:210`, gated by `flags.batch_parallelism`), but
  `.run_batch()` has zero call sites anywhere in the repo besides its own docstring example.
  Inside it, a caught subagent exception is `continue`d past, silently dropping that
  `task_id` from the returned dict — contradicting the method's own docstring contract
  ("Returns … for all tasks"). Dormant (nothing calls `run_batch()`), same "bug inside code
  nothing calls yet" shape as T1's C4; left as residual given it has zero live impact today
  and this tier's fix count is already proportionate to what was found.
- **`nash/` package** — confirmed live and reachable (`python -m orchestrator nash
  {status,tuning,compare}`, registered via `commands/nash.py`), not orphaned. Roughly 41% of
  its own LOC (`infrastructure_v2.py`'s `AsyncIOManager`/`WriteAheadLog`/
  `TransactionalStorage`/`UnifiedEventBus`, `monitor.py`'s `NashRuntimeMonitor`) has zero
  external callers of any of its public accessors — informational, not a single fixable
  defect.
- **`supervisor/` CLI registration** — `cli.py:210`'s `_supervisor_subparsers` is defined but
  never called from `entrypoints/cli_dispatch.py`; `Supervisor` remains reachable only via
  `api_server.py`'s HTTP routes. Informational, not urgent (no crash, an alternate live path
  exists).
- **`planning/decomposer.py:16,94-117`** — imports `Goal`/`SubGoal` from `.goal` then
  redefines both as plain classes with the same field shape later in the same module,
  silently shadowing the import for the rest of the file. Currently harmless (the two shapes
  are field-identical) and the whole `planning/` package is dead (only its own dedicated
  test constructs `GoalDecomposer`) — left as a documented latent risk, not fixed.

## Phase 4 — Discovered incidentally, out of T12's scope, handed off

- **`orchestrator/commands/nash.py:45`**: `from orchestrator.nash_backup import
  get_backup_manager` — `orchestrator/nash_backup.py` has never existed in this repository's
  git history (confirmed via `git log --all`). The live, registered `nash backup
  [--list|--restore|--value]` CLI subcommand crashes unconditionally with
  `ModuleNotFoundError` (no try/except around the import) — a genuine live crash, but
  `commands/` is T15's declared scope (`docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`'s T15
  row: "integrations/, vcs/, ide_backend/, commands/, cli*"). Handed off rather than reached
  into a future tier's territory for what would need either a new stub module or removing the
  broken subcommand.
- **`orchestrator/cli.py`** module docstring and inline comments (lines 4-11, 199) claim the
  CLI dispatcher is `application/cli_dispatch.py` / `application.cli_dispatch.run()`; the
  actual, correct, working import is `from .entrypoints.cli_dispatch import run` —
  `orchestrator/application/cli_dispatch.py` does not exist. Documentation-only, zero
  behavioral impact (the real import is already correct) — `cli.py` is explicitly T15's
  scope ("cli*"), handed off rather than fixed early.
- **A large structural pattern, not a single defect:** roughly a third of `application/`'s
  files and effectively all of `agents/`+`agents.py`+`planning/` implement real,
  independently-tested classes with zero production call sites — each constructed only in its
  own docstring `Usage:` example or its own dedicated unit test
  (`bandit_method_selector.py`, `bilevel_autoresearch.py`, `recursive_research.py`,
  `orthogonal_exploration.py`, `mechanism_researcher.py`/`mechanism_injector.py`/
  `mechanism_registry.py`, `trace_analyzer.py`, `search_strategy_tuner.py`,
  `vs_selector.py::CandidateSelector`, `critique_cycle.py`, `fallback_handler.py`,
  `reasoning/brain.py`, `reasoning/brainstorming.py`, `planning/decomposer.py`/`goal.py`, and
  most of `agents/`'s role classes). `agents/`'s `AgentOrchestrator` is the one exception with
  real dedicated test coverage (`tests/test_agentic_system.py` and others), suggesting a
  genuinely maintained *alternative* execution mode rather than bit-rot. Recorded for
  visibility; not a fix, and not assigned to any specific future wave.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used (Phase 1-3 survey delegated to one background agent run). One
survey claim needed no severity correction this tier (unlike T11's C1) — the one agent
inaccuracy found (T10's cache_warmup "live" claim, actually a residual note carried forward
from a *previous* tier, not this agent's own survey) was independently corrected during
verification, documented under C5.

`fix_revisions`: C1/C3/C4/C5 correct on first pass. C2 needed one revision *during
verification, before commit* — the initial fix (line 17 of `method_selector.py` only)
appeared complete by inspection, but an empirical import-success check immediately surfaced
three more instances of the identical bug in the same file (lines 18, 21, 593), which were
then fixed in the same pass before any test was written. This is exactly why this hunt
verifies fixes empirically rather than by inspection alone.
