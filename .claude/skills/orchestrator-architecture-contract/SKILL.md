---
name: orchestrator-architecture-contract
description: The architectural constitution of the Multi-LLM Orchestrator — load this BEFORE deciding WHERE new code goes, before touching engine.py, models.py, orchestrator/engine_core/container.py, or orchestrator/domain/ports.py, and whenever you see symptoms like "lint-imports contract failure", "check_new_root_files.py violation", "ImportError: circular import", "cannot import Orchestrator from engine", "partially initialized module", or you are tempted to add a method to engine.py or a helper file at orchestrator/*.py root. Explains the hexagonal layering, the Mediator/pure-data/no-new-root rules and WHY they exist, the DI container wiring, the dual-budget design, state persistence paths, root backward-compat shims, the 5 import-linter contracts as executable architecture, and the honestly-stated known-weak points (engine↔container circular imports, telemetry TODO stub, ~256-file root kernel, unwired semantic cache, 29-module mypy ignore baseline).
---

# Orchestrator Architecture Contract

The load-bearing design decisions of the Multi-LLM Orchestrator, why they exist, the
invariants that must hold, and the known-weak points — stated plainly so you neither
break the good parts nor trust the broken parts.

**Jargon, defined once:**
- **Hexagonal architecture (ports & adapters):** business logic in the center depends
  only on abstract interfaces ("ports"); concrete implementations ("adapters", e.g. a
  SQLite state store) plug in from outside. Dependencies point inward only.
- **Mediator:** an object that coordinates other services without containing their
  logic. `engine.py::Orchestrator` is the Mediator — it wires and delegates, nothing more.
- **Composition root / DI container:** the one place where concrete objects are
  constructed and injected: `orchestrator/engine_core/container.py::ServiceContainer`.
- **Shim:** a thin root-level module that only re-exports from a subpackage, kept so
  old `from orchestrator.state import StateManager`-style imports keep working.

## When NOT to use this skill

| You actually want | Use instead |
|---|---|
| Permission/process to make a change; a gate is blocking you | `orchestrator-change-control` |
| The full module-by-module map of the codebase | `mindmap` skill (loads `docs/CODEBASE_MINDMAP.md`) |
| Automated boundary checking during edits | `architecture-guard` skill |
| History of a specific bug/incident (BOM, HITL auto-approve, _aggregate…) | `orchestrator-failure-archaeology` |
| Feature-flag and config-file reference | `orchestrator-config-and-flags` |
| Environment setup, grimp pin, Windows-vs-CI differences | `orchestrator-build-and-env` |
| Attacking the engine demolition / circular-import problem itself | `orchestrator-hardest-problems-campaign` |
| Debugging a runtime failure | `orchestrator-debugging-playbook` |

---

## 1. The Four Unbreakable Rules (compact restatement)

These come from `CLAUDE.md` and are enforced by CI. Full incident history, per-rule
narratives, and the commits that motivated each rule: **`orchestrator-change-control` §2**
(this skill does not duplicate that content — see it there).

| # | Rule |
|---|------|
| 1 | **`engine.py` = Mediator.** New logic → new service module; engine only wires and delegates. |
| 2 | **`models.py` = pure data.** Dataclasses + enums only. No I/O, no asyncio, no behavior. |
| 3 | **TDD without exceptions.** Failing test first (RED), then impl (GREEN). |
| 4 | **No new root-level `orchestrator/*.py` modules.** All new code goes in existing subpackages. |

**Note on historical `engine.py` size figures (two numbers, two dates, both true):**
`docs/ARCHITECTURAL_AUDIT_V5.md` records `engine.py` at 5,036 lines / 104 methods at the V5
audit (earlier snapshot). `orchestrator-change-control` cites ~1,867 lines as of the later
ARCH-AUDIT-V2 (commit `431fc89c`, 2026-06-24) — i.e. the V5→V2 gap already reflects partial
demolition before V2 was even taken. Current live count: **1,284 lines** (2026-07-11), still
well above the cited `<= 300 lines` target (`docs/ARCHITECTURE_REMEDIATION_PLAN.md:38`; see
§3 below). Don't treat either historical figure as "the" size — both were true at their
respective dates; use `wc -l orchestrator/engine.py` for the current truth.

If any rule blocks you and you think it is wrong: escalate per `orchestrator-change-control`.
Never route around it.

---

## 2. Layer map and the placement decision rule

Verified package tree (2026-07-07). The six layers that matter for placement, innermost first:

| Layer | Package | What lives there | May import |
|---|---|---|---|
| Domain | `orchestrator/domain/` | Ports (Protocols), value objects, phase policy. Pure. | stdlib + `orchestrator.models` only |
| Domain data | `orchestrator/models.py`, `orchestrator/exceptions.py` | Model/TaskType enums, Task/ProjectState/TaskResult dataclasses, ROUTING_TABLE | stdlib only |
| Application | `orchestrator/application/` | Use-case services: `decomposer.py`, `evaluator.py`, `executor.py`, `task_executor.py`, `project_runner.py`, `fallback_handler.py`, `validators.py`… | domain, models — **never** `orchestrator.infrastructure`, **never** `orchestrator.engine` |
| Engine core | `orchestrator/engine_core/` | `container.py` (composition root), `pipeline.py`, `pipeline_executor.py`, `pipeline_runner.py`, `project_planner.py`, `state_coordinator.py`, `stages.py` | Pipeline modules: domain + models only. `container.py` alone may import infrastructure — it IS the composition root. |
| Infrastructure | `orchestrator/infrastructure/` | Concrete adapters: `state.py` (StateManager), `telemetry.py`, `llm_client.py`, `caching.py`, `path_provider.py`… | anything inward |
| Driving adapters | `orchestrator/commands/` (one file per CLI command), `orchestrator/entrypoints/` (`cli_dispatch.py`, `chat_cli.py`), `orchestrator/api_server.py` | CLI/HTTP entry points | anything |

Beyond these, ~55 other subpackages exist (`safety/`, `hitl/`, `quality/`, `generators/`,
`output/`, `crosscutting/`, …) — see the `mindmap` skill for the full map.

**Decision rule for new code** (apply in order):

1. Is it a data shape or enum? → `orchestrator/models.py` **only if** pure data; else a
   dataclass module in the owning subpackage.
2. Is it an abstract capability the core needs (interface)? → add a `Protocol` to
   `orchestrator/domain/ports.py`.
3. Is it business/use-case logic (how to decompose, evaluate, retry)? →
   `orchestrator/application/<service>.py`, depending only on ports.
4. Is it a concrete adapter (talks to disk, network, SDKs)? →
   `orchestrator/infrastructure/<adapter>.py`, implementing a port.
5. Is it wiring (constructing and connecting the above)? →
   `orchestrator/engine_core/container.py` (`ServiceContainer.build`).
6. Is it a CLI command? → `orchestrator/commands/<name>.py` registered via
   `orchestrator/entrypoints/cli_dispatch.py`.
7. **Never**: a new `orchestrator/<name>.py` at root (Rule 4), a new method on
   `Orchestrator` in `engine.py` (Rule 1), or behavior in `models.py` (Rule 2).

---

## 3. engine.py — the Mediator under demolition

- **Current size: 1,284 lines** (2026-07-11, `wc -l orchestrator/engine.py`; was 1,250 on
  2026-07-07 — `docs/ARCHITECTURE_AUDIT_V2.md` recorded 1,245 at an earlier point). LOC has
  drifted up slightly since 2026-07-07 despite the demolition campaign — re-verify before
  assuming it is monotonically shrinking.
- **Target: ≤ 300 lines.** Definition of done, quoted from
  `docs/ARCHITECTURE_REMEDIATION_PLAN.md:38`: "`engine.py` ≤ 300 lines, contains only
  construction + delegation; zero direct `orchestrator.infrastructure` imports; all
  logic lives in `application/` services wired by `engine_core/container.py`." This target
  is sourced and binding, not folklore — see `orchestrator-hardest-problems-campaign` Track A
  Phase 0 for a worked correction of an earlier claim to the contrary.
- History: 5,036 lines at audit V5 → 1,867 (ARCH-AUDIT-V2, `431fc89c`) → 1,250 → 1,284
  via phased demolition, still ~4x the target (e.g. commit `4ac9b4f1` "Phase C.1 - Remove 5
  dead methods + fix 2 critical bugs").

**What legitimately remains in `Orchestrator`:** `__init__` (builds or accepts a
`ServiceContainer`, copies handles onto `self`), the public entry points
(`run_project`, `run_project_streaming`, `dry_run`, `run_job`), and thin delegation
stubs that forward to container-held services.

**What does NOT belong:** any new `_helper` method, any inline business logic, any
direct infrastructure import. If your diff grows `engine.py`, you are doing it wrong —
extract a service and wire it in `container.py`.

`Orchestrator`'s documented runtime invariants (docstring, `engine.py` ~line 403):
cross-review uses a different provider than the generator; deterministic validators
override LLM scores; budget ceiling never exceeded (checked mid-task per iteration);
state checkpointed after each task; plateau detection prevents runaway iteration.

---

## 4. models.py purity

`orchestrator/models.py` (1,273 lines) holds `Model`, `TaskType`, `ProjectStatus`
enums, the `Task` / `TaskResult` / `ProjectState` dataclasses, `ROUTING_TABLE`,
`FALLBACK_CHAIN`, and `get_provider()`. It is covered by the `domain-purity`
import-linter contract (§7) — it must never import `orchestrator.infrastructure`,
`orchestrator.application`, `orchestrator.engine_core`, or `orchestrator.engine`.

**Critical coupled invariant (2026-06 drift incidents):** every key in
`orchestrator/config/{costs,routing,fallbacks}.json` must equal a `Model` enum
*value* exactly. The config builders do not resolve aliases; mismatched entries
**silently drop**. Details and the permanent-fix campaign:
`orchestrator-config-and-flags` and `orchestrator-failure-archaeology`.

---

## 5. Ports & adapters

`orchestrator/domain/ports.py` (752 lines) defines the port Protocols. Verified list
(2026-07-07):

`CachePort`, `StatePort`, `EventPort`, `ConfigPort`, `LLMClient`, `PlannerPort`,
`TelemetryPort`, `TracingPort`, `PolicyEnginePort`, `HookRegistryPort`,
`ValidatorPort`, `TaskExecutorPort`, `TaskQueuePort`, `SkillStorePort`,
`LSPValidatorPort`, `SnapshotPort`, `VSSamplerPort`, `QualityScorer`, `Reranker`.

Null adapters `NullEventBus` and `NullHookRegistry` also live in `ports.py` so the
container can wire safe no-ops without importing infrastructure.

Rules of the pattern here:
- Application services type-hint against ports, never concrete adapters
  (enforced by contract 2, §7).
- `api_clients.py::UnifiedClient` is the LLM provider **Adapter** (normalizes
  OpenAI/Google/Anthropic/DeepSeek SDKs into one `call_model()` returning
  `APIResponse` with `text`, `input_tokens`, `output_tokens`, `cost_usd`).
- New external dependency → new port in `domain/ports.py` + adapter in
  `infrastructure/` + wiring in `container.py`. Three files, always.

---

## 6. The DI container (`orchestrator/engine_core/container.py`)

`ServiceContainer` (759 lines) is a dataclass holding ~40 collaborator fields
(`budget`, `client`, `cache`, `state_mgr`, `selector`, `pipeline`, `evaluator`,
`semantic_cache`, `cb_registry`, …). The classmethod **`ServiceContainer.build(...)`**
(line ~289) is the single composition root: it constructs adapters (DiskCache,
StateManager, TelemetryCollector, PolicyEngine, ConstraintPlanner, pipeline stages…)
and returns a fully wired container.

`Orchestrator.__init__` then either accepts a pre-built container or calls
`ServiceContainer.build(...)` itself, and copies handles onto `self`
(`self._c = container; self.budget = container.budget; …`).

Wiring quirks you must not "fix" casually:
- Most imports inside `build()` are **deliberately lazy** (function-local) to dodge
  import cycles. Moving them to module top level will reintroduce circular imports.
- Some services need engine-level callables and are therefore **late-bound after**
  `Orchestrator.__init__` (e.g. `wire_pipeline_executor`; health tracker and
  resumption service are rebuilt by the engine — comments at container.py ~lines
  528–560 document this).
- Several subsystems import under `try/except ImportError` and degrade to `None`
  (cost_optimization suite, ModelCascader, GeneratorService). Callers must
  None-guard.

### The known circular-import weakness (engine ↔ container)

`engine.py` imports `ServiceContainer` lazily inside `Orchestrator.__init__`;
`container.py` needs engine-level callables for some services. The cycle is managed,
not solved. **Evidence — skip markers in `tests/test_phase6_10_comprehensive.py`
(2026-07-07):**

- 6 × `@pytest.mark.skip(reason="Importing Orchestrator from engine.py has circular imports")` (lines ~819–876)
- 4 × `@pytest.mark.skip(reason="Relies on container.py imports which have circular deps")` / `"Relies on container.py imports"` (lines ~498–536)

These skips are the honest cost ledger of Rule 1 being incomplete. Do not delete the
skips to make numbers look better; do not add new imports that widen the cycle.
Fixing this properly is **Track A** of `orchestrator-hardest-problems-campaign` — that
skill's Track A Phase 0.5 is the executable reproduction; read it before touching this cycle.

**Caveat (verified 2026-07-11, re-verify before trusting):** direct instantiation with a
dummy API key succeeds today with no `ImportError`:
```bash
OPENROUTER_API_KEY=sk-test-dummy python -c "from orchestrator.engine import Orchestrator; Orchestrator()"
```
The cycle described above is the *documented mechanism* (lazy import at `engine.py:432`
managing a real structural dependency), but as of this date it is **not reproducible as a
live `ImportError`** — the 10 skip markers in `tests/test_phase6_10_comprehensive.py` fail on
an eager `AuthenticationError` when `OPENROUTER_API_KEY` is unset, not a circular import. The
skip markers may be misdiagnosing their own failure. See `orchestrator-hardest-problems-campaign`
Track A Phase 0.5 for the full reproduction before assuming this is still a live blocker.

---

## 7. The 5 import-linter contracts — executable architecture (summary; owner: `orchestrator-change-control`)

`.importlinter` at repo root; run in CI as a **blocking** step (`lint-imports`) and in
pre-commit. The contracts *are* the architecture — the prose above is commentary. The full
table (verbatim contract text, `ignore_imports` entries, violation examples, and the
rules-of-engagement doctrine) is owned by **`orchestrator-change-control` §3** — this skill
does not duplicate it. One-line summary of what each contract protects, for orientation only:

1. `domain-purity` — domain/models/exceptions stay import-free of everything outward.
2. `application-no-concrete-infra` — application layer depends on ports, not concrete adapters.
3. `application-services-no-engine` — application never reaches back up into `engine.py`.
4. `engine-core-no-loose-infra` — pipeline modules stay infra-free (`container.py` excepted, it's the composition root).
5. `root-modules-no-infra` — root shims don't smuggle in infrastructure imports.

Never add `ignore_imports` entries to pass a build; a contract failure means your code is
in the wrong layer, not that the contract is wrong. For the full contract text and current
`ignore_imports` state: `orchestrator-change-control` §3.

**Windows note (2026-07-07):** running `lint-imports` locally on this Windows machine
can crash with a grimp `PanicException: range end index ... out of range` — this is
the known grimp Rust panic (hence the `grimp>=3.3,<3.4` pin in `pyproject.toml`:
"3.4+ has a Rust panic on Windows with this codebase"). CI (ubuntu-latest) is the
authority. Environment fixes: `orchestrator-build-and-env`.

---

## 8. Dual-budget design — why there are two

| Component | File | Scope | Purpose |
|---|---|---|---|
| `Budget` dataclass | `orchestrator/budget.py` (line 36) | **Per-run** | Tracks spend/time inside one `run_project()` call. Has atomic reserve/commit/release (FIX-001a) to prevent concurrent-task race conditions, and phase partitions (`decomposition`, `generation`, `cross_review`, `evaluation`, `reserve`). Defaults: `max_usd=8.0`, `max_time_seconds=5400`. |
| `BudgetHierarchy` | `orchestrator/cost.py` (line 136) | **Cross-run** | Org → Team → Job caps persisting across runs and restarts (pass `db_path`, e.g. `~/.orchestrator_cache/budget.db`). Composite pattern. |

They are deliberately separate: per-run enforcement must be lock-cheap and in-memory
inside the hot loop; organizational caps must survive process death. Both can be
active simultaneously — `BudgetHierarchy` is passed into `Orchestrator` (via
`ServiceContainer.build(budget_hierarchy=...)`) alongside the per-run `Budget`. Do
not merge them, and do not assume checking one covers the other.

---

## 9. State persistence

- **Canonical class:** `orchestrator/infrastructure/state.py::StateManager`
  (async SQLite via `aiosqlite`, connection timeout guarded).
- **DB path:** `StateManager.__init__(db_path=DEFAULT_STATE_PATH)` where
  `DEFAULT_STATE_PATH = CachePathProvider().state_db` →
  **`~/.orchestrator_cache/state.db`**, overridable via env var **`ORCH_CACHE_HOME`**.
- `orchestrator/infrastructure/path_provider.py::CachePathProvider` is the single
  source of truth for all cache paths (`state.db`, `cache.db`, `cache_l2.db`,
  `secure_cache.db`) — it replaced 21+ hardcoded `Path.home() / ".orchestrator_cache"`
  constructions. Never hardcode that path again; inject the provider.
- Pattern: Repository + Memento. State is checkpointed after each task → crash
  recovery / `--resume <project_id>`.

---

## 10. Root shims — why 250+ files still sit at root

`ls orchestrator/*.py` counts **256 files** (2026-07-07). Many are thin
backward-compatibility shims created during remediation, e.g.:

- `orchestrator/state.py` → re-exports from `infrastructure/state.py`
- `orchestrator/telemetry.py` → re-exports `TelemetryCollector` from `infrastructure/telemetry.py`
- `orchestrator/ports.py` → `from .domain.ports import *`
- `orchestrator/container.py` → re-exports `ServiceContainer` from `engine_core/container.py`

**Why they exist:** the remediation moved canonical code into subpackages without
breaking hundreds of existing `from orchestrator.state import StateManager`-style
call sites in one big-bang change. Shims carry explicit `__all__` exports (Phase A).

**Rules:** never import through a shim in *new* code — import the canonical
subpackage path. Never add logic to a shim. Deleting a shim requires migrating all
its callers first (grep before you cut). The root-file **freeze** (Rule 4) stops the
pile growing; shrinking it is ongoing remediation work.

---

## 11. Known-weak points (honest ledger, 2026-07-07)

Do not build on these as if they were solid; do not "discover" them as new bugs.

1. **engine ↔ container circular imports.** Managed via lazy imports; 6 + 4 skipped
   tests in `tests/test_phase6_10_comprehensive.py` are the standing evidence (§6).
2. **Telemetry validator-failure tracking is a stub.**
   `orchestrator/infrastructure/telemetry.py::record_validator_failure` (line ~167)
   literally says: `# TODO: Implement proper tracking` / `# For now, just degrade
   trust factor slightly`. Trust-factor math derived from validator failures is
   approximate — don't cite it as precise signal.
3. **Root kernel still ~256 files.** Frozen, not fixed. Placement rule (§2) plus
   `scripts/check_new_root_files.py` prevent regression only.
4. **Semantic cache is NOT wired into the call path.** `ServiceContainer.build`
   hardcodes `semantic_cache = None` (container.py ~line 620-621, with a comment
   requiring callers to None-guard). The field, the engine attribute
   (`self._semantic_cache`), and `orchestrator/semantic_cache.py` all exist —
   dead wiring as of the 2026-06-25 cost audit. The real cost win is the response
   DiskCache; note the cache can defeat evaluation self-consistency (see
   `orchestrator-config-and-flags`).
5. **mypy legacy baseline.** `pyproject.toml` carries a 29-module override list
   (including `orchestrator.engine`, `orchestrator.cli`,
   `orchestrator.ide_backend.ide_orchestrator_server`, …) with
   `ignore_errors = true` and the instruction "New code must NOT be added to this
   list. Remove entries as they are cleaned up." mypy is strict and blocking only
   for `domain/`, `application/`, `engine_core/` + `container.py` in CI.
6. **Contract 5 has an empty `ignore_imports` list** while root shims like
   `orchestrator/state.py` visibly import `orchestrator.infrastructure` — the
   contract-vs-shim tension is resolved in CI configuration, not in your local
   reasoning. If contract 5 fires on your diff, treat it as real; if it behaves
   surprisingly, check CI output, not local runs (local grimp panics on Windows, §7).
7. **`lint-imports` unrunnable locally on this Windows setup** (grimp panic) —
   verified 2026-07-07. Push and let CI arbitrate, or see `orchestrator-build-and-env`.

---

## 12. Pre-flight checklist before any structural change

- [ ] Placed new code per the §2 decision rule (no root file, no engine method, no models.py behavior)?
- [ ] New external capability = port + adapter + container wiring (three files)?
- [ ] No new import edge that violates a §7 contract? (Mentally trace: does my module's layer permit importing that target's layer?)
- [ ] No new top-level import in `engine.py` or `container.py` that could widen the circular-import cycle?
- [ ] Config keys touched? Then keys ≡ `Model` enum values, and drift tests run (`orchestrator-config-and-flags`).
- [ ] Failing test written first (Rule 3)?
- [ ] If a gate blocks you: `orchestrator-change-control`, not a workaround.

---

## Provenance and maintenance

All facts verified against the repo on 2026-07-07 (branch `feat/response-healing`).
Re-verify before relying on volatile numbers:

| Fact | Re-verify with |
|---|---|
| engine.py line count (1,250) | `(Get-Content orchestrator/engine.py).Count` |
| container.py line count (759) / build() location | `Select-String -Path orchestrator/engine_core/container.py -Pattern "def build"` |
| Port protocol list | `Select-String -Path orchestrator/domain/ports.py -Pattern "class \w+\(Protocol\)"` |
| 5 contracts and their ignore lists | `Get-Content .importlinter` |
| Skipped circular-import tests | `Select-String -Path tests/test_phase6_10_comprehensive.py -Pattern "circular"` |
| Telemetry TODO stub | `Select-String -Path orchestrator/infrastructure/telemetry.py -Pattern "TODO" -Context 2` |
| Root file count (~256) | `(Get-ChildItem orchestrator/*.py).Count` |
| Semantic cache unwired | `Select-String -Path orchestrator/engine_core/container.py -Pattern "semantic_cache = None"` |
| mypy legacy override list (29 modules) | `Select-String -Path pyproject.toml -Pattern "Legacy modules" -Context 0,35` |
| State DB path | `Select-String -Path orchestrator/infrastructure/path_provider.py -Pattern "state_db" -Context 2` |
| Demolition target ≤300 | `Select-String -Path docs/ARCHITECTURE_REMEDIATION_PLAN.md -Pattern "300 lines"` |
| Cited commits exist | `git log --oneline -1 431fc89c; git log --oneline -1 4ac9b4f1; git log --oneline -1 e863f0c8` |
