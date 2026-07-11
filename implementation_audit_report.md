# Implementation Audit Report — Spec-Kit Integration & Governance Hardening
<!--
  Source: E:\Documents\Vibe-Coding\Ai Orchestrator\spec-kit-integration-implementation-plan.md
  Review scope: Tracks A, B, E (Milestone 1 — recommended first cut)
  Date: 2026-07-11
  Reviewer: Automated review (3 parallel subagents + architecture verification)
-->
## 1. Executive Summary

The implementation delivers the three planned tracks — **A** (Spec-Kit ingestion), **B** (Constitutional enforcement), and **E** (Shell safety) — with the core functionality fully operational across all three. No architectural contracts are violated, no root-level `.py` files were created, and all import-linter contracts currently evaluated `KEEP`.

However, the review identifies **six medium-severity plan-compliance gaps** and several code-quality refinements that should be addressed before merging Tracks A+B. The most important gaps are: **(1)** the `constitution_enforcement_enabled` feature flag from the plan's rollback strategy was not implemented, and **(2)** `required_validators` are not merged into task hard-validators at container wiring time. Neither gap is runtime-blocking for Track A, but both must be resolved before Track B's acceptance criteria can be signed off.

**Verdict:** APPROVED WITH CHANGES — the six plan-compliance gaps must be addressed. All code-quality issues are non-blocking improvements.

---

## 2. Plan Compliance Matrix

### Track A — Spec-Kit Mode A Ingestion

| Plan Item (WBS) | Status | Evidence | Notes |
|:---|:---:|:---|:---|
| 1.1.1 — `SpecArtifacts` + `FileReaderPort` | ✅ Complete | `speckit_adapter.py:26-46`, `ports.py:705-728` | `_EXPECTED_FILES` defined but never used (dead code) |
| 1.1.2 — `parse_tasks_md` + tests | ✅ Complete | `speckit_adapter.py:79-150` | Story-group extracted but dependency edges not computed from `[Story]` groups. `[P]` marker captured but discarded. Plan says "dependency edges from `[Story]` + explicit 'depends on'" — partially implemented |
| 1.1.3 — `parse_constitution_md` + tests | ✅ Complete | `speckit_adapter.py:156-259` | Best-effort; fragile forbidden-import extraction (picks up arbitrary lowercase words) |
| 1.1.4 — `SpecKitAdapter.load` + tests | ✅ Complete | `speckit_adapter.py:310-444` | File reads via injected `FileReaderPort` ✅ |
| 1.1.5 — CLI `--from-speckit` + Orchestrator branch | ✅ Complete | `cli_dispatch.py:227-232` (flag), `project_runner.py:88-89` (branch) | `Orchestrator.run_project_with_tasks()` delegates correctly |
| 1.1.6 — Integration test (decompose-skipped spy) | ❌ Missing | N/A | No integration test that asserts `decompose_project` is **not** called on the `--from-speckit` path |
| 1.1.7 — Docs + USAGE_GUIDE | ❌ Missing | N/A | No `USAGE_GUIDE.md` update |
| Rollback — feature-flagged behind `--from-speckit` | ✅ Complete | Flag defaults to `""`, only activates when explicitly provided | Opt-in, no override of default path |

### Track B — Constitutional Enforcement Gate

| Plan Item (WBS) | Status | Evidence | Notes |
|:---|:---:|:---|:---|
| 1.2.1 — `ProjectConstitution.check_task()` + tests | ✅ Complete | `constitution.py:85-131`, `test_constitution_gate.py:27-107` | Goes beyond plan: deep-scans `task.id`, `task.prompt`, `task.context` for path references (valuable but uses hardcoded path-prefix list) |
| 1.2.2 — `ConstitutionGate` stage + tests | ✅ Complete | `constitution_gate.py:1-77`, `test_constitution_gate.py:115-180` | Has unused `__constitution_gate_marker` property (remove it) |
| 1.2.3 — `ServiceContainer` wiring + validator merge | ⚠️ Partial | `container.py:585-603` | Gate is wired and inserted at phase -1 ✅. But `required_validators` are NOT merged into task hard-validators ❌. Plan says: "append required_validators to each task's hard validators at wiring time so the existing ValidateStage enforces them" |
| 1.2.4 — Pipeline registration + abort integration | ✅ Complete | Stage inserted via `stages.insert(0, ...)` before `GenerateStage` | Uses `ctx.should_abort` / `ctx.abort_reason` — no new plumbing |
| 1.2.5 — Compliance event via `EventPort` | ❌ Missing | N/A | Plan: "Emit a compliance event via existing EventPort (observability) on abort." Gate logs a warning but does not emit an event |
| Rollback — `constitution_enforcement_enabled` flag | ❌ Missing | N/A | Plan: "Wiring guarded by `constitution_enforcement_enabled` flag (default on for `--from-speckit`, off otherwise)". Not implemented — gate activates based on constitution content only |

### Track E — Shell-Command Safety Guard

| Plan Item (WBS) | Status | Evidence | Notes |
|:---|:---:|:---|:---|
| 1.3.1 — `RiskLevel` + rule table + tests | ✅ Complete | `command_guard.py:1-215` | 18 rules covering all 4 tiers + default-deny on unknown |
| 1.3.2 — `command_guard` classifier | ✅ Complete | `command_guard.py:220-268` | Secure-by-default ✅, `BLOCKED` raises attention ✅, `DANGEROUS` requires `allow_dangerous=True` ✅ |

---

## 3. Architecture Compliance Assessment

### 3.1 Import-Linter Contracts

All 5 contracts verified manually (AST-based import checker):

| Contract | Verdict | New files checked |
|:---|:---:|:---|
| 1. `domain-purity` | ✅ KEEP | `ports.py` (FileReaderPort), `constitution.py` (check_task) — stdlib only |
| 2. `application-no-concrete-infra` | ✅ KEEP | `project_runner.py` — no infrastructure imports |
| 3. `application-services-no-engine` | ✅ KEEP | No new imports of `orchestrator.engine` |
| 4. `engine-core-no-loose-infra` | ✅ KEEP | `constitution_gate.py` — imports only domain + pipeline (sibling) |
| 5. `root-modules-no-infra` | ✅ KEEP | No new depth-1 `.py` files created |

**⚠️ Gap:** `orchestrator/ingest/` is a new root-level subpackage (not under `orchestrator/application/`). None of the 5 contracts cover it. The plan labels it "application subpackage" but it lives at `orchestrator/ingest/`, escaping `application-no-concrete-infra`. The ingest adapter currently imports only domain types — safe today — but no contract prevents a future contributor from importing infrastructure directly.

**Recommendation:** Add an import-linter contract `ingest-no-concrete-infra` in `.importlinter` or move the package to `orchestrator/application/ingest/`.

### 3.2 Layer Placement

| Component | Layer | Correct? |
|:---|:---|:---:|
| `FileReaderPort` | Domain | ✅ Follows `CachePort`, `StatePort` pattern |
| `NullFileReader` | Domain | ✅ All NullAdapters live in `ports.py` |
| `SpecArtifacts` | Ingest (application boundary) | ✅ Pure dataclass, no I/O |
| Parser functions (`parse_*_md`) | Ingest | ✅ Pure functions, testable in isolation |
| `SpecKitAdapter` | Ingest | ✅ Depends on `FileReaderPort` (injected), never opens files |
| `FileReader` | Infrastructure | ✅ Satisfies `FileReaderPort` protocol |
| `ConstitutionGate` | Engine Core / Stages | ✅ Sibling to `ValidateStage`, `PreflightStage` |
| `command_guard.py` | Safety | ✅ Existing subpackage, classifier only (no execution) |

### 3.3 Pattern Compliance

| Pattern | Applied? |
|:---|:---:|
| Hexagonal (Ports & Adapters) | ✅ `FileReaderPort` ↔ `FileReader` |
| PipelineStage Protocol | ✅ `ConstitutionGate.process(ctx) -> ctx` — structural match |
| NullAdapters for testing | ✅ `NullFileReader` in ports, `_InMemoryReader` in tests |
| Feature flags for rollback | ⚠️ Missing for Track B (`constitution_enforcement_enabled`) |
| Strangler Fig (engine extraction) | ✅ `ProjectRunner.run_project()` branches — no logic added to `engine.py` |

---

## 4. Code Quality Findings

### 4.1 Critical (none)

No security vulnerabilities, no architectural regressions, no data corruption risks found.

### 4.2 High (none)

### 4.3 Medium

| # | File | Issue | Recommendation |
|:---|:---|:---|:---|
| M1 | `speckit_adapter.py:279-282` | `parse_plan_md` substring matching causes false positives: `"go"` matches "Django", "mongoose", "good"; `"dart"` matches "dartboard"; `"rust"` matches "trust" | Use `\b` word-boundary regex: `re.search(r'\b' + re.escape(lang_key) + r'\b', lower)` |
| M2 | `speckit_adapter.py:209-215` | `parse_constitution_md` forbidden-import extraction uses brittle word-splitting — picks up arbitrary lowercase English words as "packages" | Guard with a known-package dictionary OR only extract words in backticks/quotes/blocks |
| M3 | `speckit_adapter.py:111-114` | `parse_tasks_md` path regex requires 1 of 9 hardcoded prefixes (`src/`, `tests/`, etc.) AND a file extension — misses `README.md`, `Makefile`, `Dockerfile`, `package.json`, `docker-compose.yml` | Broaden the regex: `r'(?:\S+\/)*\S+\.\w+'` for any path-like string ending in an extension |
| M4 | `speckit_adapter.py:356-359` | `_EXPECTED_FILES` is dead code — defined on the class but never referenced | Remove it or use it in `load()` for validation |
| M5 | `constitution_gate.py:40-42` | `__constitution_gate_marker` property — no other stage has this pattern; Protocol conformance is verified structurally | Remove the unused marker property |
| M6 | `cli_dispatch.py:731-736` | `_async_speckit_project` exception handler does `return` (success exit) after catching execution errors | Add `sys.exit(1)` before `return` to ensure non-zero exit on failure |

### 4.4 Low

| # | File | Issue | Recommendation |
|:---|:---|:---|:---|
| L1 | `speckit_adapter.py:262` | `tech_context` used for story group annotation — `tech_context` is documented as "tech stack note", not a story-group field | Add a dedicated `story_group` attribute or rename to `story`; document the semantic shift |
| L2 | `speckit_adapter.py:112` | Inline `import re` in `check_task()` — no benefit from lazy import | Move to module top-level |
| L3 | `speckit_adapter.py:419` | Hardcoded `.specify/memory/constitution.md` path — no override | Accept an optional `constitution_path` parameter in `load()` |
| L4 | `container.py:591` | Gate activation checks `protect_paths or forbidden_imports or require_tests` but omits `required_validators` | Add `or _constitution.required_validators` to the condition |
| L5 | `project_runner.py:89` | `constitution: Any` type — should be `ProjectConstitution | None` | Tighten the type annotation (with `TYPE_CHECKING` guard if needed) |
| L6 | `cli_dispatch.py:741` | `if not state` truthiness check — safer as `if state is None` | Use identity check |
| L7 | `cli_dispatch.py` | ~30 lines of post-execution boilerplate duplicated between `_async_new_project` and `_async_speckit_project` | Extract `_post_execution_reporting()` helper |
| L8 | `speckit_adapter.py:85-86` | `parse_tasks_md` captures `[P]` marker but discards it; `Task` has no `parallel` field | Either add a `parallel` attribute or document that it's ignored |
| L9 | `constitution.py:112` | `import re` inside `check_task` — should be module-level |

---

## 5. Testing & Coverage Assessment

### 5.1 Test Files Created

| File | Tests | Coverage |
|:---|:---|:---|
| `tests/test_speckit_adapter.py` | 20 test functions across 4 test classes | Parsers + adapter load — all green |
| `tests/test_constitution_gate.py` | 11 tests across 2 classes | `check_task` + `ConstitutionGate` stage — all green |
| `tests/test_command_guard.py` | 31 parametrized tests across 2 classes | All 4 risk tiers + edge cases + approval gating — all green |

### 5.2 Gaps

| Gap | Impact |
|:---|:---|
| No integration test that asserts `decompose_project` is NOT called on `--from-speckit` path | Plan acceptance criterion — unverified |
| No integration test that asserts zero generation tokens after constitution abort | Plan acceptance criterion — unverified |
| No test for `required_validators` enforcement via ValidateStage | Plan task 1.2.3 not implemented, so untestable |
| No test for `constitution_enforcement_enabled` feature flag | Plan rollback guard not implemented |
| Coverage ratchet not raised (`fail_under` still at 7 in pyproject.toml) | Plan requires Track A raise to 8, B to 9, E to 10 |
| No edge-case test for `parse_plan_md` substring false-positives | Risk of misrouting for "Django" → "go" match |

### 5.3 Test Quality

- ✅ RED-first approach evident (tests exist alongside implementation)
- ✅ Parametrized tests cover multiple inputs per category
- ✅ Edge cases covered: empty input, malformed input, no-constitution, whitespace-only
- ✅ Protocol conformance verified structurally (`inspect.signature`)
- ⚠️ Integration-style tests are simulated via `asyncio.run()` in unit test files rather than `pytest.mark.integration` + `@pytest.mark.asyncio`

---

## 6. Risk & Regression Analysis

### 6.1 Architectural Regressions

| Risk | Likelihood | Impact | Mitigation |
|:---|:---:|:---:|:---|
| `ingest/` package grows without import-linter protection | Medium | Medium | Add `ingest-no-concrete-infra` contract in `.importlinter` |
| `ConstitutionGate` blocks legitimate tasks due to overzealous path scanning in `check_task` | Low | Medium | Empty constitution = no-op ✅; default-off (`--from-speckit` only mechanism missing the `--constitution` flag) |

### 6.2 Backward Compatibility

| Change | Compatible? | Evidence |
|:---|:---:|:---|
| `FileReaderPort` added to `ports.py` | ✅ | Additive only; `Protocol` is structural |
| `check_task()` added to `ProjectConstitution` | ✅ | New method, no signature changes to existing |
| `ProjectRunner.run_project()` new parameters | ✅ | Both `precomposed_tasks` and `constitution` default to `None` |
| `TaskPipeline` stage insertion | ✅ | `insert(0, ...)` prepends — all existing stages remain in order |
| `ServiceContainer` dataclass | ✅ | `constitution_gate: Any = None` has a default |
| CLI `--from-speckit` flag | ✅ | Defaults to `""`; legacy paths unchanged |
| `command_guard.py` | ✅ | New module, imported by nothing |

### 6.3 Technical Debt Introduced

| Debt | Severity | Cleanup |
|:---|:---:|:---|
| `_EXPECTED_FILES` dead code | Low | Remove or activate in `load()` |
| `__constitution_gate_marker` dead property | Low | Remove |
| Duplicate post-execution reporting in `cli_dispatch.py` | Low | Extract helper before Track C |
| `constitution` parameter typed as `Any` in `ProjectRunner` | Low | Tighten when Track B consumes it |
| `aiofiles` import not listed as production dependency | Low | Either add to `pyproject.toml` or simplify `FileReaderAsync` |

### 6.4 Security Considerations

| Area | Assessment |
|:---|:---|
| `command_guard.py` | ✅ Pure classifier — no `subprocess`, `eval`, `exec`, or any I/O. Default-deny on unknown. |
| `spec-kit` adapter | ✅ No network, no shell, no `eval`. File reads only via injected port. |
| Path traversal | No path sanitization in `FileReader` — acceptable for current use case (user-specified directory via `--from-speckit`). May need `Path.relative_to()` validation if reused for other contexts. |
| Constitution enforcement | ✅ Protects via `fnmatch` — standard and battle-tested. No regex injection possible (user patterns are globs). |

---

## 7. Required Corrections

### Before Merge (Blocking)

| Severity | File | Issue | Recommendation |
|:---|:---|:---|:---|
| **BLOCKER** | `container.py` | `required_validators` not merged into task hard-validators (plan task 1.2.3) | At wiring time, after creating the gate, append `_constitution.required_validators` to each task's `hard_validators` |
| **BLOCKER** | Integration test | No spy test verifying `decompose_project` is not called on `--from-speckit` path | Add `tests/integration/test_from_speckit_flow.py` |
| **HIGH** | Rollback safety | `constitution_enforcement_enabled` feature flag not implemented | Add flag defaulting `True` for `--from-speckit`, `False` otherwise |
| **HIGH** | `constitution_gate.py` | Plan task 1.2.5: compliance `EventPort` event on abort not emitted | Inject `EventPort` into `ConstitutionGate.__init__` and call `await self._event_bus.publish(ConstitutionAbortEvent(...))` |
| **MEDIUM** | `cli_dispatch.py:736` | Non-zero exit not enforced on execution failure in `_async_speckit_project` | Add `sys.exit(1)` after `traceback.print_exc()` |
| **MEDIUM** | `speckit_adapter.py:279-282` | `parse_plan_md` substring false-positives (M1) | Use `\b` word-boundary regex |

### Before Default-On (Non-Blocking Improvements)

| Severity | File | Issue | Recommendation |
|:---|:---|:---|:---|
| **MEDIUM** | `speckit_adapter.py:356` | `_EXPECTED_FILES` dead code (M4) | Remove or use in `load()` |
| **MEDIUM** | `constitution_gate.py:41` | `__constitution_gate_marker` dead property (M5) | Remove |
| **MEDIUM** | `.importlinter` | `orchestrator.ingest` not covered by any contract | Add `ingest-no-concrete-infra` contract |
| **LOW** | `speckit_adapter.py:209` | Fragile forbidden-import extraction (M2) | Guard with known-package dictionary |
| **LOW** | `speckit_adapter.py:111` | Path regex too narrow (M3) | Broaden extension matching |
| **LOW** | `container.py:591` | Gate activation condition omits `required_validators` (L4) | Add to condition |
| **LOW** | `project_runner.py:89` | `constitution: Any` type (L5) | Tighten to `ProjectConstitution | None` |
| **LOW** | `pyproject.toml` | Coverage ratchet not raised (still at 7) | Raise to 8 (Track A), 9 (B), 10 (E) |
| **LOW** | `cli_dispatch.py` | Duplicate post-execution boilerplate (L7) | Extract `_post_execution_reporting()` |

---

## 8. Final Verdict

**APPROVED WITH CHANGES** — merge deferred until the six plan-compliance blockers are resolved.

The core engineering is sound: architecture contracts are respected, the adapter contract matches `decompose_project()` exactly, the gate uses existing abort channels, and the shell classifier is a clean pure-function design. The six blockers (outlined in §7) are straightforward — all are additions to existing plumbing, not redesigns. Estimated fix effort: ~3 days.

**Pre-merge checklist:**

- [ ] `required_validators` merged into task hard-validators in container wiring
- [ ] `constitution_enforcement_enabled` feature flag with default-off for non-spec-kit paths
- [ ] Compliance `EventPort` event emitted on constitution abort
- [ ] Integration test: spy assert `decompose_project` not called on `--from-speckit`
- [ ] Non-zero exit on execution failure in `_async_speckit_project`
- [ ] `parse_plan_md` word-boundary fix for false-positives
- [ ] `_EXPECTED_FILES` dead code removed
- [ ] `__constitution_gate_marker` property removed
- [ ] Coverage ratchet raised: 7 → 8 (minimum)
- [ ] `.importlinter` contract added for `orchestrator.ingest`
