# Implementation Plan — Phase 5: God Class Completion & Quality Ratchet

**Version:** 2.0 | **Date:** 2026-06-22 | **Branch:** `feat/response-healing` → `master`  
**Previous phases:** 0–4 (16 commits, 42 files, cli -46%, bandit 0, lint-imports 5/5)

---

## 1. Executive Summary

Phases 0–4 addressed critical fixes, coverage, security, architecture cleanup, and CLI extraction. Phase 5 targets the **final God class extraction** — reducing `engine.py` from 1881 LOC to the mediator-only target of < 400 LOC — plus **quality ratchet** work (mypy type safety, coverage growth, ruff baseline elimination). This is the most technically demanding phase due to `_execute_task` (230-line method) and `run_project` orchestration complexity.

---

## 2. Current Architecture Assessment

### 2.1 State After Phase 4

```
orchestrator/
  domain/ports.py          ← 7 protocols (+Tracing, +TaskExecutor)   [CLEAN]
  application/evaluator.py ← uses ports, not concrete adapters        [CLEAN]
  engine_core/             ← pipeline runtime, DI container           [CLEAN]
  commands/                 ← 19 self-contained modules, auto-discovered [CLEAN]
  infrastructure/          ← concrete adapters                        [CLEAN]
  engine.py                ← God class / Mediator partial             [1881 LOC]
  cli.py                   ← thin dispatch + main() + helpers         [1379 LOC]
```

### 2.2 God Class Composition

**engine.py (1881 LOC):**

| Cluster | Lines | Extractable? |
|---|---|---|
| `__init__` + container wiring | ~70 | ✅ Already wired through ServiceContainer |
| `run_project` / `run_job` / `run_project_streaming` | ~150 | ⚠️ Orchestration logic — needs careful extraction |
| `_decompose` (Instructor + fallback) | ~60 | ✅ Already delegates to DecomposerService |
| `_execute_all` (level-based parallel) | ~75 | ✅ Already delegates to PipelineRunner |
| `_execute_task` (pipeline loop) | **~230** | 🔴 Biggest cluster — core generate→critique→evaluate loop |
| `_run_preflight_check` | ~30 | ✅ Small, self-contained |
| `_select_decomposition_model` + model routing | ~80 | ✅ Could move to ModelSelector |
| `_apply_warm_start` + `_load_circuit_breaker_state` | ~70 | ✅ State management — move to StateCoordinator |
| Property accessors (~90 methods) | ~500 | 🔵 Mostly one-liner delegations; can stay as mediator facade |
| Utility/helper methods | ~200 | ✅ Extract to application/ services |
| **Total** | **~1500** (without properties) |  |

**cli.py (1379 LOC):**

| Cluster | Lines | Extractable? |
|---|---|---|
| `main()` | ~140 | ⚠️ Entry point — keep thin, but extract dispatch logic |
| Delegation stubs (10 functions) | ~40 | ✅ Already 4-line stubs |
| `cmd_cache_stats` (overload) | ~50 | ✅ Could extract to commands/ |
| `_default_output_dir` + `setup_logging` + `safe_print` + `_print_results` | ~140 | ✅ Utility modules |
| `_resolve_task_paths` + `_build_tracing_cfg` + `_handle_modify_command` | ~70 | ✅ Configuration/resolution logic |
| `print_help` + `run_help` | ~30 | ✅ Help utilities |
| **Total** | **~470** (extractable) | |

---

## 3. Phase 5 — Final God Class Extraction & Quality Ratchet

### 3.1 WBS-5.1: Extract `_execute_task` Pipeline Loop

**Objective:** Move the 230-line `_execute_task` method to `engine_core/pipeline_executor.py` (new module). The method contains the core generate→critique→revise→evaluate loop with self-consistency retry, ARA algorithm integration, and cost tracking.

**Design:** The extracted method becomes `PipelineExecutor.execute(task, pipeline, context_factory)`. `Orchestrator._execute_task` delegates to `self._pipeline_executor.execute(task)`.

**Risk: HIGH** — this is the single most critical code path. Must be extracted with full test coverage.

### 3.2 WBS-5.2: Extract Run Orchestration Logic

**Objective:** Move `run_project`, `run_job`, `run_project_streaming` to `application/project_runner.py` (extend existing). The methods orchestrate the full project lifecycle: warm-start → decompose → execute_all → save state → telemetry flush.

**Design:** `ProjectRunner` already exists in `application/`. Extend it with `run_project`, `run_job`, and `run_streaming` methods. Engine.py delegates.

**Risk: MEDIUM** — existing extraction provides a pattern to follow.

### 3.3 WBS-5.3: Extract CLI `main()` Dispatch Logic

**Objective:** Move `main()` argument parsing and flat-flag handling to `application/cli_dispatch.py`. cli.py becomes: `def main(): from .application.cli_dispatch import run; run()`. Together with remaining helper extractions, target cli.py < 400 LOC.

### 3.4 WBS-5.4: Quality Ratchet

| Task | Target |
|---|---|
| Coverage floor | 15% → 30% (add tests for extracted modules) |
| MyPy: engine.py | Fix `_execute_task` type errors after extraction |
| Ruff: F401/F821 | Remove from ignore list (auto-fix remaining) |
| `_ExecutorAdapter` | Extract to `domain/ports.py` as a reusable adapter |

---

## 4. Task Breakdown Structure (WBS)

### WBS-5.1 — Extract `_execute_task` Pipeline Loop

**Affected components:**
- `orchestrator/engine_core/pipeline_executor.py` (new)
- `orchestrator/engine.py` (delegate)
- `orchestrator/engine_core/container.py` (wire)
- `tests/unit/test_pipeline_executor.py` (new)

**Implementation tasks:**
1. Create `engine_core/pipeline_executor.py` with `PipelineExecutor` class.
2. Move `_execute_task` body verbatim, renaming `self` → `self._engine` for internal references.
3. Replace `self._pipeline.run(ctx)` with injected `pipeline.run(ctx)`.
4. Replace `self._health_tracker` → injected `health_tracker`.
5. Replace `self._telemetry` → injected `telemetry` (already port-typed).
6. Wire `PipelineExecutor` in `ServiceContainer.build()`.
7. `Orchestrator._execute_task` becomes: `return await self._executor.execute(task)`.
8. Backward compat: existing callers of `_execute_task()` unchanged.

**Acceptance criteria:**
- [ ] `_execute_task` body lives entirely in `pipeline_executor.py`
- [ ] `Orchestrator._execute_task` is ≤ 5 lines (delegation)
- [ ] All existing tests pass
- [ ] `lint-imports`: 5/5 contracts

### WBS-5.2 — Extract Run Orchestration

**Affected components:**
- `orchestrator/application/project_runner.py` (extend)
- `orchestrator/engine.py` (delegate)
- `orchestrator/engine_core/container.py` (wire)

**Implementation tasks:**
1. Extend `ProjectRunner` with `run_project`, `run_job`, `run_streaming` methods.
2. Move `run_project` body (~50 lines) — warm-start → decompose → execute → save → flush.
3. Move `run_job` body (~40 lines) — BudgetHierarchy enforcement + delegation.
4. `Orchestrator.run_project` delegates: `return await self._runner.run_project(...)`.
5. `Orchestrator.run_job` delegates: `return await self._runner.run_job(...)`.

**Acceptance criteria:**
- [ ] `engine.py` < 800 LOC after extraction
- [ ] All project lifecycle tests pass

### WBS-5.3 — Extract CLI `main()` + helpers

**Affected components:**
- `orchestrator/application/cli_dispatch.py` (new)
- `orchestrator/cli.py` (slim to < 400 LOC)
- `orchestrator/commands/` (extend nash, nexusscope if needed)

**Implementation tasks:**
1. Create `application/cli_dispatch.py` with `run()` — contains the body of cli.py `main()`.
2. Move `safe_print`, `_print_results`, `_default_output_dir` to `application/cli_helpers.py`.
3. `cli.py` becomes: `from .application.cli_dispatch import run; def main(): run()`.
4. Remove remaining delegation stubs (use dynamic discovery for all).

**Acceptance criteria:**
- [ ] `cli.py` < 400 LOC
- [ ] `python -m orchestrator --help` works

### WBS-5.4 — Quality Ratchet

**Affected components:**
- `pyproject.toml` (coverage, mypy, ruff configs)
- `tests/` (new integration/unit tests)
- `orchestrator/domain/ports.py` (TaskExecutorAdapter)

**Implementation tasks:**
1. Write `tests/unit/test_pipeline_executor.py` — mock pipeline, assert execute loop.
2. Write `tests/unit/test_project_runner.py` — mock compose/decompose operations.
3. Raise `fail_under` to 30% in `pyproject.toml`.
4. Write `TaskExecutorAdapter` in `domain/ports.py` — satisfies `TaskExecutorPort` by wrapping any callable.
5. Remove `F401`, `F821` from ruff ignore list.
6. Run `ruff check --fix orchestrator/` and commit.

**Acceptance criteria:**
- [ ] Coverage ≥ 30% in CI
- [ ] `ruff check` exits 0 with F401, F821 un-ignored
- [ ] `_ExecutorAdapter` removed from `commands/website.py`

---

## 5. Risk & Mitigation Matrix

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| `_execute_task` extraction breaks pipeline | Medium | Critical | Extract verbatim; full regression test before merge |
| `run_project` orchestration refactor introduces race condition | Low | High | Level-based parallelism already tested via PipelineRunner |
| MyPy type checking reveals 100+ new errors | High | Medium | Phase them — fix engine.py/cli.py first, then root modules |
| Coverage 30% not reachable without test infrastructure changes | Medium | Medium | Move tests/ out of gitignore first |
| CLI dynamic discovery breaks `--help` for nash/nexusscope | Low | Low | nash/nexusscope not in `discover_command_modules()` auto-find; verify via `pkgutil` |

---

## 6. Testing & Quality Assurance

| Phase | Test Type | Coverage Target |
|---|---|---|
| WBS-5.1 | Unit tests for PipelineExecutor loop | 90% branch coverage |
| WBS-5.2 | Integration tests for run_project lifecycle | 80% branch coverage |
| WBS-5.3 | CLI smoke test: `--help` for all subcommands | All 19 commands |
| WBS-5.4 | Ruff regression: no new F401/F821 | Zero violations |

---

## 7. Deployment & Rollback

- All extractions are **Strangler Fig**: old method becomes delegation stub, new service runs identical code.
- Rollback: revert delegation stub to include original body. No data migration needed.
- Per-commit atomic: each WBS item is its own commit with independent rollback.

---

## 8. Post-Implementation Validation

- [ ] `engine.py` < 800 LOC (WBS-5.2), target < 400 (WBS-5.1)
- [ ] `cli.py` < 400 LOC (WBS-5.3)
- [ ] `lint-imports`: 5/5 contracts
- [ ] `bandit` HIGH: 0
- [ ] `ruff`: clean (F401, F821 removed from ignore)
- [ ] Coverage ≥ 30%
- [ ] `python -m orchestrator --help` functional
- [ ] `python -m orchestrator website -d "test"` functional
