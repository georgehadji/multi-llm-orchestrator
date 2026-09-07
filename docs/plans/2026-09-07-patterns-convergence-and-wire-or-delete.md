# Patterns, Convergence, and Wire-or-Delete — the three remaining items

**Date:** 2026-09-07 · **Base:** PR #30 head `71782f8` (all 16 checks green)
**Scope:** the three items left open by `2026-09-07-root-cause-analysis-and-fix-plan.md`:
(1) the module→paradigm/pattern mapping, (2) the 3 genuinely-diverged duplicate pairs,
(3) wire-or-delete for the batch API, projections, and leaderboard/cross-project-learning.
**Directive being executed:** *"wire them safely and optimally if possible. otherwise delete."*
**Ground rules:** WORM; every claim tagged **VERIFIED** / **INFERENCE** / **UNKNOWN**; every
VERIFIED claim names a file:line or a command in §7 that reproduces it. Layer placement is
per `orchestrator-architecture-contract` §2 and the five import-linter contracts, not
invented here.

---

## 0. Three calls that resolved differently from how they were framed

Stated first so the rest of the document is read against the evidence, not the framing.

1. **"Converge the 3 diverged pairs" collapses to deletion.** In all three, one copy is a
   strict superset or the *both* copies are superseded by a live third implementation
   (§2). There is nothing to merge — only a worse copy to remove.
2. **The leaderboard is live and must be kept, not deleted.** It is reachable from a CLI
   command through a chain the census could not see (§4.3). It gets the "fix" branch of
   wire-or-delete, not the "delete" branch.
3. **The batch API cannot be wired safely *or* optimally through this architecture** —
   not because of its defects, which are fixable, but because the live provider adapter
   points at OpenRouter, which has no batch endpoint, so the client can only ever fall to
   its "simulate" path: sequential realtime calls with a 300-second stall bolted on (§4.1).
   Deletion is the *only* correct answer, and the honest route to batching is a new
   native-provider adapter, not this module.

---

## 1. Module → paradigm → pattern mapping

### 1.1 How this table was built, and what it is not

It is **not** a list of patterns I think each module should adopt. Inventing a pattern per
module is exactly the over-engineering this repo's `ponytail` discipline forbids. Three
evidence sources only:

- **Prescribed** — `CLAUDE.md`'s pattern table and the architecture contract's six-layer
  map (§2 of that skill). These are binding.
- **Declared** — 40 modules carry a `Pattern:` line in their own docstring. Census
  (VERIFIED, §7): Strategy ×8, Observer ×6, Facade ×3, Repository ×2, and a long tail of
  Builder / Template Method / Factory Method concentrated in `generators/` (6 declaring
  modules), `safety/` (4), `operations/` (4), `engine_core/` (4).
- **Measured** — per-package import edges into `infrastructure` / `engine` /
  `application` / `domain` (VERIFIED, §7). This is what turns "should be in layer X" into
  "is in layer X" or "leaks out of it".

Where a package has no prescribed or declared pattern and no evident one, the table says
**none** rather than guessing.

### 1.2 The core (contract-governed)

| package | layer (contract §2) | paradigm | pattern | measured | action |
|---|---|---|---|---|---|
| `models.py`, `exceptions.py` | domain data | pure dataclasses + enums | — | 0 outward edges ✓ | none |
| `domain/` | domain | `typing.Protocol` ports; pure value objects | Ports & Adapters (declared: "Structural subtyping via typing.Protocol") | 0 infra/app/engine edges ✓ | none |
| `application/` | application | async use-case services typed against ports | Strategy (routing), Chain of Responsibility (`validators`, `preflight`), Decorator (`verification`) | 0 infra, 0 engine ✓ — 12 domain, 45 models | none; **this is the reference layer** |
| `engine_core/` | engine core | composition root + entry-point-discovered pipeline stages | Mediator's collaborators; Pipeline; Plugin (stages via `orchestrator.pipeline.stages` entry points) | 10 infra edges, **all in `container.py`** ✓ (contract 4 permits exactly that) | none |
| `engine.py` | mediator | thin delegation | Mediator | 1,284 LOC vs ≤300 target (contract §3) | out of scope; Track A of the hardest-problems campaign |
| `infrastructure/` | infrastructure | concrete async adapters | Adapter (`llm_client` → OpenRouter via `AsyncOpenAI`), Repository + Memento (`state.py`) | **2 edges into `application/`** — `verification_checks.py:23,373` import `application.verification_gate` | **flag**: an inward layer importing an outward one. Not contract-covered, but it is the inversion contract 2 exists to prevent in the other direction. Fix: move `VerificationCheck` to `domain/` (it is a value type) — a 3-file port move per contract §5 |
| `commands/`, `entrypoints/`, `api_server.py` | driving adapters | one module per CLI command (`register`/`execute`); FastAPI | Command; Facade | may import anything ✓ | none |
| `unified_events/` | cross-cutting | event-sourced bus with in-process read models | Observer + CQRS projections (`Projection`, `ProjectStateProjection`, `MetricsProjection`, wired at `core.py:872-873`) | 0 outward edges ✓ | none — **and this is why `projections.py` is redundant (§4.2)** |

### 1.3 Application-tier packages that leak downward (measured smell, not a contract breach)

These packages hold use-case logic and are therefore application-tier by the §2 decision
rule, but the contracts only fence `orchestrator/application/` itself — so nothing stops
them importing infrastructure directly, and some do:

| package | infra edges | what leaks | paradigm / declared pattern | action |
|---|---|---|---|---|
| `quality/` | 4 | validators reaching concrete adapters | Chain of Responsibility (declared: "verification levels") | route through a port, or move the adapter-touching piece to `infrastructure/` |
| `reasoning/` | 3 | `ara_pipelines` talks to the client directly | Pipeline (declared: "Pipeline, Strategy, Factory Method, Immutable Data") | inject `LLMClient` port; also the home of B7's unweighted paid call |
| `generators/` | 2 | codegen touching disk/adapters | Builder + Template Method + Factory Method (6 declaring modules — the most pattern-literate package in the repo) | route through `output/` or a port |
| `nash/` | 2 (+1 engine) | `infrastructure_v2` is itself an adapter living outside `infrastructure/` | State Machine (WAL, 2PC) | either rename the package's intent (it is infrastructure) or move `infrastructure_v2` |

**Recommendation:** extend contract 2's *spirit* with one new import-linter contract —
`application-tier-packages-no-concrete-infra` with `source_modules = quality, reasoning,
generators, nash` — seeded with today's 11 edges as `ignore_imports` and treated as a
shrink-only allowlist, the same shape as the shadowing gate. That converts a measured smell
into an enforced boundary without a big-bang refactor. Classification: **Architectural**,
human review required, per change-control.

### 1.4 Everything else — paradigm by evidence, pattern only where declared

| package | LOC | paradigm (evident) | pattern (declared, else "none") | liveness / note |
|---|---|---|---|---|
| `agents/` | 2,450 | injected-collaborator agent objects; role-keyed dispatch | Registry + Abstract Factory (added this session, `registry.py`); Strategy (`persona_modes`) | live; 10/10 roles now reachable |
| `safety/` | 8,018 | monitors, sandboxes, kill switches | 4 declaring modules; Template Method | `guardrails.py` now wired at `__aenter__` |
| `operations/` | 10,858 | ops services (rollout, diagnostics, export) | 4 declaring; Observer, Strategy | 13/32 files unreachable — largest dead concentration |
| `cost_optimization/` | 6,059 | per-phase cost levers | Strategy | **batch trio deleted (§4.1)**; `token_budget` keeps `OptimizationPhase` |
| `analysis/` | 4,971 | read-only analytics over run data | Composite (`pareto_frontier` declared) | `leaderboard` **kept and fixed** (§4.3); `projections` deleted (§4.2) |
| `learning/` | 2,058 | cross-run learning stores | Observer (`learning_aggregator` declared) | `cross_project_learning` deleted (§4.4); `transfer_learning` + `learning_aggregator` are the live pair |
| `integrations/` | 5,505 | external-service adapters | Facade ×2 (declared) | correct layer for adapters; `mcp_server` canonical copy lives here |
| `state_mgmt/` | 3,072 | persistence + session lifecycle | Memento (declared); Repository | 1 infra edge — this package *is* infrastructure and could live there |
| `design/`, `output/`, `scaffold/`, `appbuilder/`, `project_mgmt/` | ~17k | template-driven generation | Builder / Template Method (declared in `design`) | application-tier codegen |
| `events/`, `hitl/`, `plugin(s)/`, `skills/`, `tools/`, `workspace/`, `memory/`, `kanban/` | small | registries and channels | Plugin registry; Observer | mostly live, small |
| `meta/`, `supervisor/`, `nexus_search/`, `context_mgmt/`, `pattern_learner/`, `dashboard_core/`, `testing/` | — | — | none declared | fully referenced (T23) |
| root kernel `orchestrator/*.py` | 246 files, 61,863 LOC | frozen; 78 re-export shims + legacy | Mediator (`engine`), Adapter (`api_clients`), Composite (`cost.py`), State Machine (`resilience`) | Rule 4 freezes growth; shrinking it is RC1 Stage B |

**Where this mapping is weak (stated):** the "paradigm (evident)" column for the long tail
is read from `__init__` docstrings and file names, not from a per-file read — the same
depth limit as P4–P11. The table is reliable for the core (§1.2, contract-measured) and
for the 40 self-declaring modules; elsewhere it is orientation.

---

## 2. The three diverged pairs — converge by deleting the worse copy

All three: **0 importers of the copy to be removed** (VERIFIED, product + tests + scripts).

### 2.1 `docker_generator.py` (root, 740 LOC) vs `generators/docker_generator.py` (761)

The sub-package copy is a strict security superset (VERIFIED by diff, non-import lines):
- adds `add_nonroot_user()` and calls it from every `for_*_app()` — the root copy runs
  containers as root;
- pins base images (`3.12-slim`, `20-slim`, `1.23-alpine`, `1.82-slim`) where the root copy
  uses `latest`;
- reads database credentials from `${VAR}` interpolation where the root copy **hardcodes
  `POSTGRES_PASSWORD: postgres`**;
- `postgres:16-alpine` vs `15-alpine`.

Nothing in the root copy is absent from the sub-package copy. **Delete the root copy.**
Deleting it is itself a security improvement: it removes the only generator in the tree
that emits hardcoded credentials.

### 2.2 `mcp_server.py` (root, 573) vs `integrations/mcp_server.py` (653)

The sub-package copy is a functional superset: adds a `GitSnapshotStore`/`NullSnapshotStore`
seam and two tools (`orch_project_status`, `orch_project_snapshots`). Its relative imports
were repaired in Phase 1. **Delete the root copy.** (The MCP server as a whole remains an
unwired capability — `ESCALATION_REGISTER.md` B3 — but that decision is now about one file,
not two.)

### 2.3 `remediation.py` (root, 117) vs `operations/remediation.py` (125)

The sub-package copy's own header: *"⚠️ DEPRECATED (Phase 1 — Resilience Unification). The
strategy-ordering logic here should use the canonical `UnifiedResiliencePolicy` and
`ResiliencePolicyConfig.fallback_strategies` from `domain/resilience_policy.py`."*
`domain/resilience_policy.py` is live (1 importer). **Delete both copies** — the
deprecation notice already names their replacement, and neither has a caller.

**Convergence work required: none.** Three deletions, one commit each, root-freeze baseline
updated via `--update`.

---

## 3. Wire-or-delete: the decision procedure

"Wire safely and optimally if possible" needs a test that can actually fail. Used here:

1. **Is there a live consumer for the capability, or a clean seam where one would attach?**
   No consumer and no seam ⇒ wiring means inventing a feature, not remediation.
2. **Does a live module already provide the capability?** If yes, wiring creates a second
   source of truth — RC1's duplicate-drift problem at subsystem scale.
3. **Can the module deliver its stated purpose through the live adapters at all?** If the
   architecture makes the purpose unreachable, no amount of fixing helps.
4. Only if 1 is yes, 2 is no, and 3 is yes: fix its known defects **first**, then wire.

Every subsystem below was run through all four.

## 4. The decisions

### 4.1 Batch API — DELETE (test 3 fails: unreachable purpose)

**Files:** `cost_optimization/batch_client.py` (506), `cost_optimization_integration.py`
(428, TEST-ONLY), `pricing_cache.py` (339, ORPHAN).

VERIFIED chain:
- The live LLM adapter is `infrastructure/llm_client.py:540-541` —
  `AsyncOpenAI(base_url="https://openrouter.ai/api/v1/")`. OpenRouter is a routing proxy;
  it exposes no `/v1/batches` endpoint.
- **No live object anywhere exposes `.batches`** (`grep -rn "\.batches\." orchestrator/`
  outside the batch client → nothing).
- `_submit_batch_job` gates on `hasattr(self.client, "batches")` (`batch_client.py:309`),
  so it **always** falls to `_simulate_batch_processing` (`:345`) — a sequential loop of
  ordinary `client.call()`s. Zero discount, by construction.
- Around that loop sit the defects already recorded: a 300-second stall for any request
  that is not the 10th (P3-BATCH1), provider polling against a locally-generated id
  (P3-BATCH2), a ~1000× inflated savings metric (P3-BATCH3), and four dropped parameters
  (P3-BATCH4).

So the wired outcome would be *strictly worse than not wiring*: every evaluation/critique
call pays the stall and gets the same price. Fixing the four defects does not change that;
the provider does.

**Optimal alternative, recorded for whoever wants batching:** it requires a native
provider adapter (Anthropic Message Batches or OpenAI Batch) — a new port in
`domain/ports.py`, an adapter in `infrastructure/`, wiring in `container.py` — the
three-file rule. Not a resurrection of this module, whose `client.batches.create(
input_file=...)` shape is the OpenAI file-upload API and matches neither Anthropic's nor
OpenRouter's surface.

**Blast radius (VERIFIED):**
- `tests/unit/test_hunt_t10_cost_optimization.py::test_c3_cost_optimization_integration_imports_cleanly`
  — asserts only that the module imports; delete that one test (its C1/C2/C4 siblings are
  unrelated and stay).
- `BatchClient` name scrubbed from the try/except-to-`None` blocks in
  `cost_optimization/__init__.py` (4 refs), `engine.py` (3), `engine_core/container.py` (3),
  `engine_core/engine_deps.py` (2), and the name list in `engine_flags.py:77` (1).
- `OptimizationMetrics.batch_calls` / `batch_savings` (`cost_optimization/__init__.py:54-57,
  75-77, 95`) removed.
- **Keep `OptimizationPhase`** — `token_budget.py` uses it.

### 4.2 Projections — DELETE both twins (tests 1 and 2 fail)

**Files:** `projections.py` (636, ORPHAN), `analysis/projections.py` (639, "referenced"
only by `analysis/__init__.py:15`'s star-import).

- Test 1: zero real consumers of `ModelPerformanceProjection`, `BudgetProjection`, or
  either accessor outside the module, its twin, and the dead router cluster (VERIFIED).
- Test 2: `unified_events/core.py` **already has a live CQRS projection system** — its own
  `Projection` ABC (`:664`), `ProjectStateProjection` (`:674`), `MetricsProjection` (`:747`),
  registered on the bus at `:872-873` and exposed via `get_project_state` / `get_metrics`
  (`:1024-1032`). `projections.py` defines a *second, incompatible* `Projection` hierarchy
  over the same events. Wiring it would put two read-model frameworks on one bus.

The one thing the dead copy has that the live one lacks — `get_best_model_for_task` — is a
routing signal already provided three ways: `planner.select_model`,
`adaptive_router.preferred_model`, `learning_aggregator.get_routing_recommendations`.

**Seam, recorded:** `UnifiedEventBus.add_projection()` is the clean attachment point if a
model-performance read model is ever wanted. It should be a subclass of the *live*
`Projection`, not a port of this one.

**Also delete:** `analysis/__init__.py:15` (`from .projections import *`).

### 4.3 Leaderboard — KEEP `analysis/leaderboard.py`, FIX it, delete the root twin (test 1 passes)

This is the call that flipped. The census marked both copies as referenced-only-by-
`__init__`; the transitive chain is (VERIFIED, every hop):

```
orchestrator/cli_nash.py:58        from .nash_stable_orchestrator import get_nash_stable_orchestrator
orchestrator/nash/stable_orchestrator.py:135   self.pareto_frontier = get_cost_quality_frontier()
                                    :181,:295  await self.pareto_frontier.get_pareto_frontier(...)
orchestrator/analysis/pareto_frontier.py:203   self.leaderboard = leaderboard or get_leaderboard()
                                    :449,:478  self.leaderboard._summaries.get(model)
```

A CLI command reaches it. **Deleting the leaderboard breaks `cli_nash`.** So it is live,
and gets the fix branch:

1. **Delete the root twin** `orchestrator/leaderboard.py`. Its only importer is
   `engine_core/outcome_router.py:35` (`from ..leaderboard import …`), which is deleted in
   §4.5, so no re-pointing is needed. `pareto_frontier` already imports the `analysis/`
   copy (`:47`, relative).
2. **Fix the two unread parameters** (found by the P3 detector, never actioned):
   - `get_leaderboard(self, task_type: TaskType | None = None)` (`analysis/leaderboard.py:681`)
     never reads `task_type`. **VERIFIED: the method has zero external callers** — the two
     `get_leaderboard()` hits at `pareto_frontier.py:203` and `outcome_router.py:154` are the
     module-level singleton *factory*, not this method, and `pareto_frontier` bypasses it by
     reading `_summaries` directly (which item 3 fixes). So there is no caller to preserve a
     filter for: **drop the parameter.** A parameter that does nothing is RC5's
     silent-absence shape.
   - `_get_recommended_tasks(self, model, summary)` (`:719`) never reads `model`. Private,
     exactly one caller (`:706`, which passes both): drop the parameter and the argument.
3. **Add a public accessor for `_summaries`.** `pareto_frontier.py:449,478` reads a private
   attribute across a module boundary — the same encapsulation break that hid P3's
   `agent._profiles` bug. Add `ModelLeaderboard.summary_for(model) -> ModelBenchmarkSummary
   | None` and point both call sites at it.
4. Fix the docstring at `analysis/leaderboard.py:17`, which still shows
   `from orchestrator.leaderboard import …` — the path being deleted.

### 4.4 Cross-project learning — DELETE both twins (tests 1 and 2 fail)

**Files:** `cross_project_learning.py` (652, ORPHAN), `learning/cross_project_learning.py`
(652, referenced only by `learning/__init__.py:2`).

Zero real consumers of `CrossProjectLearning` / `get_cross_project_learning` (VERIFIED).
The live pair covers the capability: `transfer_learning.py` (4 importers — similarity and
pattern transfer) and `learning/learning_aggregator.py` (persistent cross-run stats with
`get_routing_recommendations`). `CrossProjectLearning.inject_into_routing(router)` is the
same signal `learning_aggregator` already emits.

**Also delete:** `learning/__init__.py:2`. Leave `:1` (`cross_project`, a different module)
alone.

### 4.5 The outcome-router cluster — DELETE (tests 1 and 2 fail); record the real seam

Not on the directed list, but it is the leaderboard's *other* importer and the projections'
sibling, so the plan has to say what happens to it.

**Files:** `engine_core/outcome_router.py` (580), `router_integration.py` (254, ORPHAN).
`outcome_router` is "referenced" only by `router_integration` (`:27`), which nothing
imports (VERIFIED). Phase 1 made both importable and fixed PX-ROUTER1's two un-awaited
calls — deliberately, so the reachability gate could land green; that commit's own message
left the wire-or-delete call to this phase.

- Test 1: no consumer; its data source (`feedback_loop.py`, both twins ORPHAN) is dead.
- Test 2: three live routing signals exist (above).
- **Seam, recorded:** `engine_core/container.py::_wire_acr_backend` already swaps
  `ConstraintPlanner`'s backend on `ORCH_ACR_BACKEND`. A future outcome-weighted router
  belongs there, shaped as a planner backend — not as this standalone class.

**Consequences:** remove `get_adaptive_router` / `reset_adaptive_router` from
`adaptive_router.py` — added in Phase 1 solely for `router_integration`, they would
otherwise become the newest dead functions in the tree. The two `feedback_loop.py` twins
lose their last importer; they were already ORPHAN and are a follow-on candidate, not in
this plan's scope.

---

## 5. Sequencing

All items are **Standard** class except §1.3's new contract (**Architectural**). Every
commit: RED where a behaviour changes (§4.3), full gate suite, one item per commit,
`check_root_module_freeze.py --update` on every root deletion.

| step | contents | needs | LOC removed |
|---|---|---|---|
| S1 | §2 — delete `docker_generator.py`, `mcp_server.py`, both `remediation.py` | nothing | ~1,555 |
| S2 | §4.4 — cross-project-learning twins + `learning/__init__.py:2` | nothing | ~1,304 |
| S3 | §4.2 — projections twins + `analysis/__init__.py:15` | nothing | ~1,275 |
| S4 | §4.5 — outcome-router cluster + the two Phase-1 factory functions | nothing | ~860 |
| S5 | §4.3 — root `leaderboard.py` deleted; canonical copy fixed (RED tests for the two parameters and the accessor) | S4 (its importer) | ~820, +~15 |
| S6 | §4.1 — batch trio, the C3 test, the five name scrubs, two metric fields | nothing | ~1,273 |
| S7 | §1.3 — new import-linter contract with shrink-only allowlist | human review | 0 |

Net: **~7,100 LOC of unreachable code removed**, one live module fixed, one new boundary
enforced. Reachability census after S1–S6 (INFERENCE from the file list): ORPHAN 93 → ~84
files.

---

## 6. Where this plan is weak

- **The leaderboard chain was found by chance.** The census's "referenced" bucket hid it
  behind an `__init__` star-import, and only a manual importer trace surfaced
  `pareto_frontier`. **INFERENCE:** other "referenced-by-`__init__`-only" modules may be
  live the same way; §4.2 and §4.4 were traced by hand (zero real consumers) precisely
  because of this, but the general lesson — treat star-imports as *not* evidence of
  liveness — belongs in `reachability_census.py`.
- **§1.4's long tail is orientation, not audit.** Same coverage caveat as P4–P11.
- **§1.3's contract proposal is a judgment.** The 11 leaking edges are measured; whether
  fencing them is worth the review cost is the maintainer's call.
- **UNKNOWN: whether OpenRouter will add a batch surface.** §4.1's conclusion holds for
  the adapter as written today; the three-file alternative is the right shape regardless.
- **Correction (found executing S4): §4.5's "both twins ORPHAN" claim for `feedback_loop.py`
  is wrong for the root copy.** VERIFIED by grep before deletion: `orchestrator/feedback_loop.py`
  is imported by `federated_learning.py:49`, `knowledge_graph.py:38`,
  `nash/stable_orchestrator.py:40`, and `analysis/pareto_frontier.py:46` — the same live chain
  that makes the leaderboard live (§4.3). It was not touched by S4 (outcome_router.py and
  router_integration.py both independently verified at 0 importers, so their deletion is
  unaffected), and it must **not** be treated as a delete candidate in any follow-on cleanup.
  `orchestrator/operations/feedback_loop.py` — the other twin — genuinely has 0 importers
  (VERIFIED) and remains what the plan said: an ORPHAN follow-on candidate, out of this
  plan's scope. Unlike the leaderboard chain, this one was not found by a deliberate manual
  trace — it was an unverified assumption in the original write-up that a routine
  before-you-delete grep caught. Same lesson as the bullet above, now confirmed twice: don't
  assert liveness/deadness for a module adjacent to the one actually being changed without
  running the grep.

---

## 7. Reproducing the measurements

| claim | command |
|---|---|
| diverged pairs' non-import diffs | `diff orchestrator/X.py orchestrator/<pkg>/X.py \| grep -vE '^[<>]\s*(#\|from \.\|import \|$)'` |
| 0 importers of each delete candidate | `grep -rnE "(from\|import) orchestrator\.X\b\|from \.\.?X import" orchestrator/ tests/ scripts/` |
| live provider is OpenRouter; no `.batches` | `grep -n base_url orchestrator/infrastructure/llm_client.py`; `grep -rn "\.batches\." orchestrator/` |
| leaderboard chain | `grep -n get_leaderboard orchestrator/analysis/pareto_frontier.py`; `grep -n pareto_frontier orchestrator/nash/stable_orchestrator.py`; `grep -n nash_stable orchestrator/cli_nash.py` |
| live projection system | `grep -nE "^class .*Projection\|add_projection" orchestrator/unified_events/core.py` |
| declared-pattern census | `grep -rhoE "^Pattern:\s*.*" --include=*.py orchestrator/ \| sort \| uniq -c` |
| per-package layer edges | the loop in this session's transcript; re-run per package with `grep -cE "^\s*from (orchestrator\|\.\.)\.infrastructure"` |
| engine_core's 10 infra edges all in container.py | `grep -rlE "^\s*from (orchestrator\|\.\.)\.infrastructure" orchestrator/engine_core/` |
