# T16 — operations/ remainder, verification/, policy*, telemetry/logging misc — Inventory

Final wave of T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`, closing the
AUTONOMOUS DEFECT-HUNT PROTOCOL V7 continuation across backend files with no individual
disposition recorded in earlier tiers.

## Phase 0 — Scope

75 files by the plan's estimate: `operations/` remainder (~30, excluding `circuit_breaker.py`/
`diagnostics.py`, already dispositioned), `verification/` (6), `testing/` (4), `tools/` (3),
`skills/` (4), `crosscutting/` (2), plus root cross-cutting modules and a mandated deep-dive
on two specific leads carried over from earlier tiers: the `SecretsFilter`-installation
residual (T8/T9/T13) and CLAUDE.md's "policy system... not fully integrated" claim. The
survey additionally pulled in `project_mgmt/` (6) and `workspace/` (4) beyond the plan's
literal `find -maxdepth 1` scope, since they are the same root/subpackage duplicate-pair
shape this hunt targets and the plan's own file-count reconciliation implies their inclusion;
credited as in-scope here rather than flagged as an interpretation gap.

## Phase 1-3 — Survey, candidates, trigger/innocence

A background agent surveyed the full 75-file scope plus the two mandated deep-dive leads,
giving full correctness-depth to `policy_engine.py`/`policy.py`/`policy_dsl.py` (mandated),
all 6 `verification/` files (mandated, one dependency temporarily installed to empirically
exercise `JSONSchemaVerifier`), the full `SecretsFilter`/logging-configuration chain, and
every file it found real defects in — cross-referencing callers and, for the two leads,
tracing the complete call chain from `python -m orchestrator` down to the exact line that
does or doesn't enforce/mask. I independently re-verified every finding this tier changed
before implementing: re-read each cited file/line directly, ran empirical `git diff`/`diff`
comparisons on every claimed duplicate pair, confirmed caller counts via repo-wide grep
myself rather than trusting the agent's counts, and for the `services/` shim conversion (the
highest-regression-risk fix in this tier, touching 8 pre-existing test files) ran all 8 files
against the new shims before treating the fix as safe.

One independently-discovered lead predates the agent's report: a stop-hook "untracked files"
check surfaced `orchestrator/self_test_log.txt`, traced to `operations/quick_self_test.py`
writing into the source tree at import time. The agent's own report corroborates and expands
this under its informational findings — both routes converged on the same defect (C7 below).

## Phase 4 — Fixes (VERIFIED DEFECT)

### C1 — SecretsFilter never installed on any real logging entry point

**File:** `orchestrator/application/cli_helpers.py::setup_logging()`.

Three independent `configure_logging()`-shaped functions exist in this repo:
- `log_config.py::configure_logging()` — has the `SecretsFilter` (fixed in hunt T8), but its
  only caller anywhere in the repo is T8's own regression test. Zero live callers, confirmed
  again this tier.
- `orchestrator/logging.py::configure_logging()` — a separate structlog-based fork with no
  `SecretsFilter` at all, calls `logging.basicConfig(force=True)` which would wipe any prior
  handlers. Zero live callers (only stale `scripts/utils/` debug scripts reference it) — a
  deprecated fork nobody finished retiring. Left alone: it has no callers, so a fix here
  wouldn't reach anything either, and retiring a whole module is outside this hunt's scope.
- **The actual live one:** `application/cli_helpers.py::setup_logging()`, called from
  `entrypoints/cli_dispatch.py::run()` at line 241 (every CLI invocation) and line 391 (the
  YAML/file-project run path) — confirmed by direct read of both call sites. It called plain
  `logging.basicConfig(...)` with no filter of any kind.

Since `log_config.py::get_logger(name)` is used by ~100 files (confirmed: `infrastructure/
llm_client.py`, the live `UnifiedClient` adapter, among them) and none of those loggers had
their own handlers, every log record from those ~100 files propagated unmasked to the one
real `StreamHandler` this function installs. Concrete exception-logging call sites on the
live path that could embed request context from calls built with real `Authorization: Bearer`
headers: `infrastructure/llm_client.py:441` (every provider-call failure, the single
most-traveled LLM-call error path in the system), `integrations/config_sync.py`, and
`feedback_loop.py` (confirmed live via `router_integration.py`/`engine_core/
outcome_router.py`/`federated_learning.py`/`knowledge_graph.py` importers).

**Fix:** mirror `log_config.py`'s already-correct pattern — after `logging.basicConfig(...)`,
attach a `SecretsFilter()` to every handler on the root logger. Verified: a `Bearer <token>`-
shaped string logged through a child logger after calling `setup_logging()` now comes out as
`[REDACTED_BEARER_TOKEN]` instead of the raw value (`test_c1`, mirroring T8's own C1 test
technique against the real live entry point this time).

### C2 — `testing/validator.py`'s LLM-based test generator called a nonexistent client method

**File:** `orchestrator/testing/validator.py:371` (`TestValidator._generate_test()`).

Called `self.client.call_model(model=..., prompt=..., max_tokens=..., temperature=...)`.
`UnifiedClient` has never had a `call_model` method — only `.call(model, prompt, system=,
max_tokens=, temperature=, **kwargs)` (confirmed: `hasattr(UnifiedClient, 'call_model')` is
`False`; the sibling `testing/fixer.py:144` calls `.call()` correctly with the same
parameter shape). Every real invocation raised `AttributeError`, silently caught by the
broad `except Exception as e:` three lines below, which falls back to
`_generate_fallback_test()` — a trivial `assert True`-shaped stub — that then gets written to
disk, actually run via pytest, passes (exit code 0), and is reported by
`validate_test_generation()` as a successfully generated, validated test. Not just a skip:
a confirmed false-positive on the validator's entire reason for existing ("ensures generated
tests pass BEFORE committing them").

Currently unreachable in production: `TestValidator` is imported behind
`flags.test_validation_enabled` in `engine.py`/`engine_core/engine_deps.py`, but neither the
flag nor the class is ever read/constructed again outside its own module and tests — a
second, independent "fully built, never wired" gap on top of the method-name bug (matching
this hunt's already-documented pattern #6/wiring-gap shape). Left the wiring gap as residual
(architectural, not a mechanical fix) but fixed the method-name bug itself, since it's a
real, cheap, isolated defect that would still be wrong the day someone does wire it up —
matching this hunt's standing "cheap fix even in unreached code" precedent (T14's
`analysis/performance.py`, T15's `commands/kanban.py`/`gateway.py`).

**Fix:** `call_model(` → `call(`. Verified via a mock client: pre-fix, the mock's `.call` is
never awaited (the code tries `.call_model` first, which auto-creates a Mock attribute on a
plain `AsyncMock`, whose non-string return value then fails downstream when treated as text
— a different failure, but still caught by the same broad except); post-fix, `.call` is
awaited once with the correct kwargs and the real generated code is returned instead of the
fallback stub.

### C3 — `project_mgmt/analyzer.py` was a stale, unshimmed duplicate

852 (root `project_analyzer.py`) vs 843 lines, fully independent implementations, not a shim.
Root confirmed live via `engine.py:1239` (`from .project_analyzer import ProjectAnalyzer`,
called from `_analyze_completed_project`). Root has since gained delegation to the real
`ArchitectureScorer` (replacing a crude filename-matching heuristic `project_mgmt/
analyzer.py` still has) plus two whole methods (`save_report()`, `print_suggestions()`) that
`engine.py` calls and the `project_mgmt/` copy doesn't have. Confirmed zero external callers
of `project_mgmt.analyzer` (only its own `__init__.py`'s `from .analyzer import *`) — same
"reverse-direction fork, root is canonical" shape as T12's `meta/integration.py`.

**Fix:** converted to `from ..project_analyzer import *` shim. Verified `ProjectAnalyzer` now
resolves identically from both paths.

### C4 — four `operations/` duplicate pairs, unshimmed, dead on the `operations/` side

- `operations/hitl_workflow.py` vs live root `hitl_workflow.py` (imported by `meta/
  v2_integration.py`, `application/bilevel_autoresearch.py`) — 2-line diff, both import the
  same `StrategyProposal` class via a different (but both valid) module path; confirmed by
  identity check (`is` → `True`) that this is cosmetic, not a hidden second definition.
- `operations/concurrency_controller.py` vs live root (imported by `services/executor.py`,
  `engine_core/container.py`, `application/executor.py`, 2 test files — all via the root
  path). Real functional divergence: root added asyncio fire-and-forget task-reference
  protection (`self._pending_tasks` set + `add_done_callback`) around budget-charge/release
  tasks that `operations/`'s own `TaskConcurrencyGuard` still lacks. Confirmed zero callers
  to the `operations/` copy's own class anywhere in the repo.
- `operations/deployment_feedback.py` vs root — root added a `_validate_deployment_url()`
  SSRF guard (blocks non-https schemes, private/loopback/link-local IPs, cloud metadata
  hosts) that `operations/`'s copy lacks entirely. Confirmed zero callers to
  `DeploymentFeedbackLoop` anywhere in the repo, on either side — a dormant risk on dormant
  code (same disposition class as T9's `safety/sandbox.py`, T10's `docker_sandbox.py`).
- `operations/memory_tier.py` vs root — diff is import-depth-comment lines only; both copies
  share the identical `_touch_memory`/`delete_project_memories` silent-file-skip pattern
  T14 already evaluated and explicitly declined to elevate ("best-effort cache-touch/delete,
  not a security or correctness gate... only matters if a COLD-tier file is both corrupted
  and happens to match a deletion target's id"). Respecting that prior considered
  disposition: fixed the duplicate-pair hygiene issue (shim), did not reopen the bug T14
  already triaged.

**Fix (all four):** converted the `operations/` copy to a `from ..X import *` shim. Verified
each shim resolves to the exact same class object as its root counterpart.

### C5 — `services/` submodules shadowed their own package's canonical re-export

`services/__init__.py` already correctly re-exports `ExecutorService`/`GeneratorService`
(aliased from `DecomposerService`)/`ObservabilityService` from `application/`, with an
explicit "canonical implementations now live in orchestrator/application/" docstring. But
`services/executor.py`, `services/generator.py`, `services/observability.py` each
independently defined their **own** class of the same name. Confirmed empirically:
`from orchestrator.services.executor import ExecutorService` and `from orchestrator.
application.executor import ExecutorService` were different Python objects (`is` → `False`),
while `from orchestrator.services import ExecutorService` correctly resolved to the real one.

Eight existing test files import via the submodule path and were therefore silently
exercising the superseded classes, not the ones production code runs: `tests/
test_executor_service.py`, `test_generator_service.py`, `test_service_observability.py`,
`test_concurrency_controller.py`, `test_phase6_resilience.py`, `tests/integration/
test_resume_after_crash.py`, `test_circuit_breaker_fail_fast.py`, `test_full_run.py`. A full
`diff` of `services/executor.py` vs `application/executor.py` and `services/observability.py`
vs `application/observability.py` showed only cosmetic/docstring divergence today — no
functional bug yet, but nothing was stopping the two from silently diverging further, since
the "wrong" file is what the tests exercise. `services/generator.py` (251 lines) vs
`application/decomposer.py` (713 lines, `DecomposerService`) diverges much more: the real
class gained Instructor-fast-path/JSON5 parsing, task-type-string mapping, and model
selection that the stale 251-line copy never had — the exact same "one-third-scale stale
fork of an actively-developed class" shape T12 already documented for `application/
decomposer_service.py`.

**Fix:** `services/executor.py` and `services/observability.py` converted to plain
`from ..application.X import *` shims (names match exactly). `services/generator.py`
converted to an explicit aliased shim (`DecomposerResult as GeneratorResult`, etc.) — the
same aliasing `services/__init__.py` already does, applied at the submodule level so
`services.generator` stops shadowing it. **Verified empirically, not just by inspection**:
ran all 8 affected test files against the shimmed classes — all 73 tests across the 8 files
passed unchanged, confirming the "cosmetic-only" diff assessment for executor/observability
and that `DecomposerService`'s broader interface remains backward-compatible with everything
the 3 generator-dependent test files actually exercise.

### C6 — `crosscutting/config.py`'s re-exported "static defaults" always silently fell back to stale, wrong values

`orchestrator/config.py` has never defined `TIMEOUT_DEFAULT_SECONDS`/`TOKENS_MAX_OUTPUT`/
`BUDGET_DEFAULT_USD` (real values are namespaced under `Timeout`/`TokenLimits`/
`BudgetDefaults` classes) — so the `try/except ImportError` guarding this re-export always
took the except branch, every time, silently. One of the three hardcoded fallbacks
(`DEFAULT_BUDGET_USD = 10.0`) didn't even match the real live default
(`BudgetDefaults.MAX_USD_DEFAULT = 8.0`) it was evidently meant to mirror — the other two
fallback values (120, 4096) already matched `Timeout.API_CALL_LONG` and
`TokenLimits.CODE_STANDARD` exactly, confirming those are the intended real sources.
Confirmed nothing in the repo currently imports these three re-exported names, so today's
blast radius is zero — but the bug is real and would silently misinform any future caller.

**Fix:** import the real `Timeout`/`TokenLimits`/`BudgetDefaults` classes and read the
matching attributes; corrected the stale fallback to `8.0` to match. mypy's isolated diff
confirms this closes 3 real `attr-defined` errors that existed against the pre-fix import.

### C7 — `operations/quick_self_test.py` executed a full test at import time, writing into the source tree

Not gated behind `if __name__ == "__main__":` — importing (or, per `operations/__init__.py`'s
own docstring, `from orchestrator.operations.diagnostics import ...`-style direct submodule
access to this one) would run a live integration check immediately: open
`os.path.join(os.path.dirname(os.path.dirname(__file__)), "self_test_log.txt")` (i.e.
`orchestrator/self_test_log.txt`, inside the source tree) for writing, attempt `from
dashboard_mission_control import MissionControlServer` (a module retired years ago per
`dashboard_core/mission_control.py`'s own docstring — confirmed no compatible replacement
class exists there either), and call `sys.exit(1)` at module scope on failure. Independently
discovered twice: once via a stop-hook untracked-file check that found the stray log file
this session, and once by the T16 survey agent under its informational findings.
`operations/__init__.py` explicitly avoids wildcard-importing the package specifically to
avoid this class of problem for other submodules — this file is the one landmine of that
exact shape actually sitting in the package.

**Fix, deliberately narrow:** wrapped the body in `def main(): ...` behind `if __name__ ==
"__main__":`, and changed the log destination to `tempfile.gettempdir()` instead of the
source tree. Did **not** attempt to rewire the `dashboard_mission_control` import to a
guessed modern equivalent — no compatible `MissionControlServer`-shaped replacement was
found anywhere in the codebase, and inventing one would mean guessing at behavior this hunt
has consistently declined to invent (matching T13's `nash_backup`, T15's `Orchestrator.
modify_codebase()`). The script's own self-test remains non-functional against the current
architecture; the two structural defects (import-time execution, source-tree write) are
what's fixed here.

## Phase 0 mandated deep-dive — investigated, `[REQUIRES HUMAN REVIEW]`, not fixed

### Policy system: CLAUDE.md's "not fully integrated" is a significant understatement

Verdict: **zero live enforcement, on every entry point**, including the one whose own
docstring claims otherwise. Full chain, independently spot-checked against the agent's
citations rather than taken on trust:

- `engine.py::Orchestrator.run_job()`'s docstring claims policy is "enforced on every API
  call." `spec.policy_set` is written to `self._run_ctx.active_policies` and never read again
  anywhere in the repo (`application/run_context.py`'s `active_policies` field is write-only).
- `engine.py::_get_active_policies()` reads a *different*, never-set attribute
  (`self._active_policies`, not `self._run_ctx.active_policies`) and is itself never called
  anywhere — a leftover from a pre-`RunContext`-extraction refactor.
- Real model selection goes through `ModelSelector` (`container.py`), constructed with no
  `policy_engine` parameter at all — architecturally incapable of consulting policy.
- The only code calling `PolicyEngine.check()` is `ConstraintPlanner._apply_filters()`,
  reached only via `.select_model()`/`.select_reviewer()`/`.replan()` — which repo-wide grep
  confirms have zero production callers (test-only), despite `container.planner` being a
  real, fully-wired `ConstraintPlanner` that `engine.py` only ever reads a data attribute
  from, never its selection methods.
- `PolicyEngine.enforce()` — docstring: "used as a hard gate before each API call in
  engine.py" — has zero callers anywhere in the repository, full stop.

**Practical consequence:** a caller can build a `JobSpec` with a real `PolicySet` (region
restrictions, PII/training-consent flags), call `run_job()` believing per its own docstring
that compliance is enforced on every API call, and get zero actual enforcement with no error
and no audit trail. **Not fixed**: closing this gap means deciding *where* enforcement should
gate (inside `ModelSelector`? A new pre-flight check in `ProjectRunner.run_job()`? Reviving
`_get_active_policies()`'s intended call site?) — an architectural/product decision, not a
mechanical fix, matching this hunt's standing treatment of "wire a dormant subsystem"
findings (T10/T15's `mcp_server.py` feature-convergence, T12's HITL wiring).
`[REQUIRES HUMAN REVIEW]` — escalated past CLAUDE.md's current one-line note; recommend
updating CLAUDE.md's "Known Limitations" section to state this plainly rather than as
"enforcement mode selection... not fully integrated," which understates a live, silent,
security/compliance-relevant no-op.

### `operations/autonomy_config.py`'s inert Multi-Mode Selector

`engine.py` hardcodes `AutonomyConfig.for_level(AutonomyLevel.STANDARD)` and never reads
`self._autonomy` again anywhere. The real `--agent-profile` CLI flag handling
(`entrypoints/cli_dispatch.py`, four near-identical inline blocks) uses its own ad-hoc
`profile_map` dict that never touches `AutonomyConfig`/`from_agent_profile()` at all — the
well-designed Lite/Standard/Auto/Max autonomy system is completely inert. **Not fixed**:
correcting this means rewriting all four `cli_dispatch.py` blocks to route through
`AutonomyConfig.from_agent_profile()` instead of the ad-hoc dict, a multi-block behavioral
change to a live, user-facing CLI flag's semantics — a product decision about which
mapping should win, not a mechanical fix. `[REQUIRES HUMAN REVIEW]`.

## Cleared (innocent) / residual, not fixed

- **`orchestrator/verification.py`** permanently shadowed by the `orchestrator/verification/`
  package (its own re-export can never be reached via its own declared import path).
  Confirmed nothing imports it today. Same shape T2 already recorded for `gateway.py`/
  `gateway/` and `orchestrator/agents.py`/`agents/` — both left as documented, harmless
  hygiene landmines rather than "fixed," since there is no code change that un-shadows a file
  from a same-named package without a product decision to delete one side. Following that
  established precedent: recorded, not modified.
- **`orchestrator/logging.py`**'s separate structlog-based `configure_logging()` fork — zero
  live callers, self-evidently a deprecated fork (a stale migration script exists specifically
  to rewrite old callers away from it). Not fixed: adding a `SecretsFilter` here wouldn't
  reach anything, and retiring a whole module is outside a defect-hunt fix's scope.
- **`ShellTool.execute()`** (`tools/shell_tool.py`) runs `asyncio.create_subprocess_shell` on
  caller-supplied `params["command"]` — a real shell-injection-shaped surface, but this is a
  tool explicitly built to run arbitrary shell commands on an agent's behalf; that is its
  designed function, not a defect, and it has zero production construction (test-only) today.
  Not fixed, not elevated.
- **`services/completion_judge.py`**, **`services/autonomy_costs.py`** — dead code, no
  `application/` counterpart, zero callers anywhere. Not a shadowed duplicate (nothing to
  reconcile); left as residual, consistent with T15's treatment of similar zero-caller dead
  files (`integrations/compat.py`).
- **`crosscutting/config.py:24-34`, `file_scope.py`, `module_system.py`, `multi_context.py`,
  `i18n.py`**'s silent `except Exception: pass` fallbacks on malformed local config — the
  survey itself distinguished these from the T6/T8/T9/T13 false-clean-scan pattern (config
  graceful-degradation, not a security/quality gate reporting false success) and declined to
  elevate them; respecting that same-tier reasoned downgrade rather than relitigating it.
- **`operations/remediation.py`** — self-documents as deprecated in favor of
  `domain/resilience_policy.py`; both sides zero live callers. Informational only.
- **`tools/`, `skills/skills.py`** — fully built, zero production construction (test-only),
  one more data point for this hunt's already-documented "roughly a third of `application/`
  plus most of `agents/`/`planning/` real but never wired" pattern. Not fixed (nothing broken
  to fix — these are complete and correct, just unused).

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used. `fix_revisions`: all 12 fixes correct on first pass, RED→GREEN
verified on the first attempt for all 7 test cases (C4's four-pair shim and C5's three-pair
shim are each covered by one combined test per pattern, matching the established convention
of one test per *defect shape* rather than one per file when the fix is mechanically
identical across a group). The `services/` shim conversion (C5) was treated with the most
caution given its 8-test-file blast radius — verified by actually running all 73 tests in
those files against the new shims before treating the fix as final, not by diff inspection
alone.
